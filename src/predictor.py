"""
Inference Pipeline for Career Predictor.

Singleton pattern for thread-safe, lazy-loaded model serving.
Strictly separated from training logic.
"""

import json
import threading
from typing import Dict, List, Tuple, Optional

import numpy as np
import joblib
from xgboost import XGBClassifier

from src.config import Config
from src.processor import encode_single_input
from src.utils import get_logger
from src.validator import CareerValidator

logger = get_logger(__name__, Config.LOG_DIR)


class CareerPredictor:
    """
    Singleton inference engine for career prediction.
    Lazily loads model, encoders, and schema on first use.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._model: Optional[XGBClassifier] = None
        self._label_encoders: Optional[Dict] = None
        self._target_encoder = None
        self._feature_schema: Optional[Dict] = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> 'CareerPredictor':
        """Get or create the singleton predictor instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Load all model artifacts from disk."""
        logger.info("Loading prediction artifacts...")
        required = {
            'model': Config.MODEL_PATH,
            'encoders': Config.ENCODERS_PATH,
            'target_encoder': Config.TARGET_ENCODER_PATH,
            'schema': Config.FEATURE_SCHEMA_PATH,
        }
        for name, path in required.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {name} at {path}. "
                    "Run the training pipeline first."
                )

        self._model = joblib.load(Config.MODEL_PATH)
        self._label_encoders = joblib.load(Config.ENCODERS_PATH)
        self._target_encoder = joblib.load(Config.TARGET_ENCODER_PATH)
        with open(Config.FEATURE_SCHEMA_PATH, 'r') as f:
            self._feature_schema = json.load(f)

        self._loaded = True
        logger.info(
            f"Artifacts loaded. Features: {self._feature_schema['feature_count']}, "
            f"Classes: {len(self._target_encoder.classes_)}"
        )

    def preprocess_input(self, form_data: Dict) -> np.ndarray:
        """Encode a single form input into a feature vector."""
        if not self._loaded:
            raise RuntimeError("Predictor not initialized.")
        encoded = encode_single_input(
            input_dict=form_data,
            label_encoders=self._label_encoders,
            feature_schema=self._feature_schema,
        )
        return encoded.reshape(1, -1)

    def predict(self, form_data: Dict) -> Tuple[str, np.ndarray]:
        """Predict career from form data. Returns (label, probabilities)."""
        X = self.preprocess_input(form_data)
        proba = self._model.predict_proba(X)[0]
        
        # Apply sharpening to boost top choice confidence for UX calibration
        # This makes the top prediction more dominant as requested
        sharpening = getattr(Config, 'PREDICTION_SHARPENING', 1.0)
        if sharpening != 1.0:
            proba = np.power(proba, sharpening)
            proba = proba / (np.sum(proba) + 1e-15)
            
        predicted_class = np.argmax(proba)
        career_label = self._target_encoder.inverse_transform([predicted_class])[0]
        logger.info(f"Prediction: {career_label} (confidence: {proba[predicted_class]:.2%})")
        return career_label, proba

    def get_top_predictions(self, form_data: Dict, n: int = 3) -> List[Dict]:
        """
        Get top-N career predictions with confidence scores, SHAP explanations,
        and validation alignment scores.
        
        Args:
            form_data: User input data dictionary
            n: Number of top predictions to return (default: 3)
            
        Returns:
            List of prediction dictionaries with confidence, factors, and alignment scores
        """
        _, proba = self.predict(form_data)
        top_indices = np.argsort(proba)[::-1][:n]
        
        from src.explain import SHAPExplainer
        X = self.preprocess_input(form_data)
        explainer = SHAPExplainer.get_instance()
        
        results = []
        for idx in top_indices:
            label = self._target_encoder.inverse_transform([idx])[0]
            confidence = float(proba[idx])
            interpretations = explainer.get_local_interpretations(X, class_idx=idx)
            
            # Format top 3 positive factors
            positive_factors = [
                item for item in interpretations if item['impact'] > 0
            ][:3]
            factors_summary = [
                f"{item['feature']} (+{item['impact']:.1f}%)"
                for item in positive_factors
            ]
            
            # Get validation scores for this prediction
            validation_report = CareerValidator.generate_validation_report(
                form_data, label
            )
            
            results.append({
                'career': label,
                'probability': round(confidence * 100, 2),
                'factors': factors_summary,
                'interpretations': interpretations,
                'alignment_scores': validation_report.get('alignment_scores', {}),
                'warnings': validation_report.get('warnings', []),
                'suggestions': validation_report.get('suggestions', []),
                'is_aligned': validation_report.get('is_aligned', True),
            })
        return results

    def get_class_names(self) -> List[str]:
        if self._target_encoder is None:
            return []
        return list(self._target_encoder.classes_)

    def get_feature_names(self) -> List[str]:
        if self._feature_schema is None:
            return []
        return self._feature_schema['feature_names']

    def get_prediction_validation(self, form_data: Dict, career: str) -> Dict:
        """
        Get comprehensive validation report for a specific career prediction.
        
        Args:
            form_data: User input data
            career: Career label to validate
            
        Returns:
            Validation report with alignment scores and recommendations
        """
        return CareerValidator.generate_validation_report(form_data, career)

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None
