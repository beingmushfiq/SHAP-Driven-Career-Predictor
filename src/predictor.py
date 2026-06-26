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

        # Validate alignment mapping against model classes
        model_classes = set(self._target_encoder.classes_)
        for field, careers in Config.FIELD_CAREER_ALIGNMENT.items():
            valid_careers = [c for c in careers if c in model_classes]
            invalid_careers = [c for c in careers if c not in model_classes]
            if invalid_careers:
                logger.warning(
                    f"Field {field} references invalid careers (not in model): {invalid_careers}"
                )
            if not valid_careers:
                logger.warning(
                    f"Field {field} has NO valid careers! Predictions will be problematic."
                )

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
        """Predict career from form data. Returns (label, probabilities).
        Filters out unsupported career domains and field-mismatched careers."""
        X = self.preprocess_input(form_data)
        proba = self._model.predict_proba(X)[0]
        
        # Apply sharpening to boost top choice confidence for UX calibration
        sharpening = getattr(Config, 'PREDICTION_SHARPENING', 1.0)
        if sharpening != 1.0:
            proba = np.power(proba, sharpening)
            proba = proba / (np.sum(proba) + 1e-15)
        
        # Determine allowed careers for the user's field
        user_field = str(form_data.get('field', '')).strip()
        aligned_careers = set(Config.FIELD_CAREER_ALIGNMENT.get(user_field, []))
        
        # Zero out unsupported and field-mismatched career clusters
        for idx in range(len(proba)):
            label = self._target_encoder.inverse_transform([idx])[0]
            if label in Config.UNSUPPORTED_CAREER_CLUSTERS:
                proba[idx] = 0.0
            elif aligned_careers and label not in aligned_careers:
                proba[idx] = 0.0
        
        # Re-normalize after zeroing
        total = np.sum(proba)
        if total > 0:
            proba = proba / total
            
        predicted_class = np.argmax(proba)
        career_label = self._target_encoder.inverse_transform([predicted_class])[0]
        logger.info(f"Prediction: {career_label} (confidence: {proba[predicted_class]:.2%})")
        return career_label, proba

    def get_top_predictions(self, form_data: Dict, n: int = 3) -> List[Dict]:
        """
        Get top-N career predictions with filtering, alignment scoring, and re-ranking.
        
        Strategy:
            1. Generate top 5+ predictions from the model
            2. Filter out unsupported career domains
            3. Compute Career Alignment Score for each
            4. Remove predictions below minimum alignment threshold
            5. Re-rank by combined score (model confidence + alignment)
            6. Return top N valid recommendations
        
        Args:
            form_data: User input data dictionary
            n: Number of top predictions to return (default: 3)
            
        Returns:
            List of prediction dictionaries with confidence, alignment, and SHAP data
        """
        _, proba = self.predict(form_data)
        
        # Step 1: Get more candidates than needed (top 5+ from model)
        candidate_count = max(n * 3, 12)
        top_indices = np.argsort(proba)[::-1][:candidate_count]
        
        from src.explain import SHAPExplainer
        X = self.preprocess_input(form_data)
        explainer = SHAPExplainer.get_instance()
        
        # Determine allowed careers for the user's field (hard filter)
        user_field = str(form_data.get('field', '')).strip()
        aligned_careers = set(Config.FIELD_CAREER_ALIGNMENT.get(user_field, []))
        has_field_filter = bool(aligned_careers)
        if has_field_filter:
            logger.info(f"Field alignment filter active for '{user_field}': {aligned_careers}")
        
        # Step 2-4: Filter and score each candidate
        scored_candidates = []
        for idx in top_indices:
            label = self._target_encoder.inverse_transform([idx])[0]
            confidence = float(proba[idx])
            
            # Skip unsupported career clusters
            if label in Config.UNSUPPORTED_CAREER_CLUSTERS:
                logger.info(f"Filtered unsupported career: {label}")
                continue
            
            # Hard filter: skip careers not aligned with user's field
            if has_field_filter and label not in aligned_careers:
                logger.info(f"Filtered field-mismatched career: {label} (field: {user_field})")
                continue
            
            # Skip zero-confidence predictions
            if confidence <= 0:
                continue
            
            # Compute Career Alignment Score
            alignment_score = CareerValidator.compute_career_alignment_score(form_data, label)
            
            # Skip careers below minimum alignment threshold
            if alignment_score < Config.MIN_ALIGNMENT_SCORE:
                logger.info(
                    f"Filtered low-alignment career: {label} "
                    f"(score: {alignment_score:.1f} < {Config.MIN_ALIGNMENT_SCORE})"
                )
                continue
            
            # Get SHAP interpretations for this class
            interpretations = explainer.get_local_interpretations(X, class_idx=idx)
            
            # Format top 3 positive factors
            positive_factors = [
                item for item in interpretations if item['impact'] > 0
            ][:3]
            factors_summary = [
                f"{item['feature']} (+{item['impact']:.1f}%)"
                for item in positive_factors
            ]
            
            # Get full validation report
            validation_report = CareerValidator.generate_validation_report(form_data, label)
            
            # Step 5: Compute combined ranking score
            # Combined = w1 × model_confidence + w2 × alignment_score
            combined_score = (
                Config.RANK_WEIGHT_CONFIDENCE * (confidence * 100) +
                Config.RANK_WEIGHT_ALIGNMENT * alignment_score
            )
            
            scored_candidates.append({
                'career': label,
                'probability': round(confidence * 100, 2),
                'alignment_score': round(alignment_score, 1),
                'combined_score': round(combined_score, 2),
                'factors': factors_summary,
                'interpretations': interpretations,
                'alignment_scores': validation_report.get('alignment_scores', {}),
                'warnings': validation_report.get('warnings', []),
                'suggestions': validation_report.get('suggestions', []),
                'skill_gaps': validation_report.get('skill_gaps', []),
                'is_aligned': validation_report.get('is_aligned', True),
            })
        
        # Step 6: Re-rank by combined score and return top N
        scored_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        results = scored_candidates[:n]
        
        if not results:
            logger.warning("No valid recommendations found after filtering")
        
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
