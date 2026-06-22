"""
Model Comparator — Cross-Model Metrics & Inference.

Loads the trained XGBoost (primary) and Random Forest (secondary) models and
their metadata to produce:
  - Side-by-side accuracy / F1 / precision / recall comparison
  - Dual-model career prediction for a single input
  - Agreement rate between the two models

This module is imported by:
  - views.py (comparison API)
  - comparator_dashboard (management command rendering)

The comparator does NOT retrain — it only reads persisted artifacts.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import joblib

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__, Config.LOG_DIR)


# ─── Metadata Loader ──────────────────────────────────────────────────────────

def load_metadata(path: Path) -> Optional[Dict]:
    """Load a JSON metadata file; returns None if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning(f"Metadata file not found: {path}")
    return None


# ─── Model Comparator ─────────────────────────────────────────────────────────

class ModelComparator:
    """
    Singleton that compares the primary XGBoost model vs the secondary RF model.

    Attributes:
        xgb_meta (dict): Metadata from the XGBoost training run.
        rf_meta  (dict): Metadata from the RF training run.
        xgb_model: Loaded XGBClassifier (lazy-loaded).
        rf_model:  Loaded RandomForestClassifier (lazy-loaded).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.xgb_meta: Optional[Dict] = None
        self.rf_meta: Optional[Dict] = None
        self._xgb_model = None
        self._rf_model = None
        self._xgb_target_enc = None
        self._rf_target_enc = None
        self._xgb_scaler = None
        self._xgb_label_encs = None
        self._xgb_schema = None
        self._rf_schema = None
        self._initialized = True
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from both models on first construction."""
        self.xgb_meta = load_metadata(Config.METADATA_PATH)
        self.rf_meta  = load_metadata(Config.RF_METADATA_PATH)

        if self.xgb_meta:
            logger.info(
                f"XGB meta loaded — accuracy: "
                f"{self.xgb_meta.get('metrics', {}).get('accuracy', 'N/A')}"
            )
        if self.rf_meta:
            logger.info(
                f"RF meta loaded — accuracy: "
                f"{self.rf_meta.get('metrics', {}).get('accuracy', 'N/A')}"
            )

    # ─── Lazy model loading ───────────────────────────────────────────────────

    def _load_xgb_artifacts(self):
        """Lazily load XGBoost model + encoders."""
        if self._xgb_model is not None:
            return
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"XGB model not found: {Config.MODEL_PATH}")
        self._xgb_model = joblib.load(Config.MODEL_PATH)
        if Config.TARGET_ENCODER_PATH.exists():
            self._xgb_target_enc = joblib.load(Config.TARGET_ENCODER_PATH)
        if Config.SCALER_PATH.exists():
            self._xgb_scaler = joblib.load(Config.SCALER_PATH)
        if Config.ENCODERS_PATH.exists():
            self._xgb_label_encs = joblib.load(Config.ENCODERS_PATH)
        if Config.FEATURE_SCHEMA_PATH.exists():
            with open(Config.FEATURE_SCHEMA_PATH) as f:
                self._xgb_schema = json.load(f)
        logger.info("XGB model + artifacts loaded.")

    def _load_rf_artifacts(self):
        """Lazily load RF model + encoders."""
        if self._rf_model is not None:
            return
        if not Config.RF_MODEL_PATH.exists():
            raise FileNotFoundError(f"RF model not found: {Config.RF_MODEL_PATH}")
        self._rf_model = joblib.load(Config.RF_MODEL_PATH)
        if Config.RF_TARGET_ENCODER_PATH.exists():
            self._rf_target_enc = joblib.load(Config.RF_TARGET_ENCODER_PATH)
        if Config.RF_FEATURE_SCHEMA_PATH.exists():
            with open(Config.RF_FEATURE_SCHEMA_PATH) as f:
                self._rf_schema = json.load(f)
        logger.info("RF model + artifacts loaded.")

    # ─── Comparison Dashboard Data ────────────────────────────────────────────

    def get_comparison_data(self) -> Dict:
        """
        Return structured comparison data for the dashboard template.

        Returns:
            {
                'xgb': { 'available': bool, 'metrics': {}, 'params': {}, ... },
                'rf':  { 'available': bool, 'metrics': {}, 'params': {}, ... },
                'winner': 'xgb' | 'rf' | 'tie',
                'delta_accuracy': float,
                'comparison_ready': bool,
            }
        """
        def _format_model(meta, model_type) -> Dict:
            if not meta:
                return {
                    'available': False,
                    'model_type': model_type,
                    'metrics': {},
                    'params': {},
                    'n_features': 0,
                    'n_classes': 0,
                    'class_labels': [],
                    'dataset': 'N/A',
                    'trained_at': 'Not trained',
                }
            metrics = meta.get('metrics', {})
            return {
                'available': True,
                'model_type': meta.get('model_type', model_type),
                'metrics': {
                    'accuracy': round(metrics.get('accuracy', 0) * 100, 2),
                    'f1_weighted': round(metrics.get('f1_weighted', 0) * 100, 2),
                    'precision_weighted': round(metrics.get('precision_weighted', 0) * 100, 2),
                    'recall_weighted': round(metrics.get('recall_weighted', 0) * 100, 2),
                },
                'params': meta.get('best_params', {}),
                'n_features': meta.get('n_features', 0),
                'n_classes': meta.get('n_classes', 0),
                'class_labels': meta.get('class_labels', []),
                'dataset': meta.get('dataset', 'career_dataset_student.csv'),
                'trained_at': meta.get('trained_at', 'Unknown'),
                'version': meta.get('version', '1.0.0'),
            }

        xgb_data = _format_model(self.xgb_meta, 'XGBClassifier')
        rf_data  = _format_model(self.rf_meta,  'RandomForestClassifier')

        # Determine winner
        xgb_acc = xgb_data['metrics'].get('accuracy', 0)
        rf_acc  = rf_data['metrics'].get('accuracy', 0)
        delta = abs(xgb_acc - rf_acc)

        if xgb_acc > rf_acc + 0.5:
            winner = 'xgb'
        elif rf_acc > xgb_acc + 0.5:
            winner = 'rf'
        else:
            winner = 'tie'

        both_ready = xgb_data['available'] and rf_data['available']

        return {
            'xgb': xgb_data,
            'rf': rf_data,
            'winner': winner,
            'delta_accuracy': round(delta, 2),
            'comparison_ready': both_ready,
        }

    # ─── Metric Bars for Chart.js ─────────────────────────────────────────────

    def get_chart_data(self) -> Dict:
        """
        Return metric data formatted for Chart.js bar/radar charts.

        Returns:
            {
                'labels': ['Accuracy', 'F1', 'Precision', 'Recall'],
                'xgb_values': [92.3, 91.8, ...],
                'rf_values': [89.1, 88.5, ...],
            }
        """
        data = self.get_comparison_data()
        labels = ['Accuracy', 'F1 (weighted)', 'Precision', 'Recall']
        metric_keys = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

        xgb_vals = [data['xgb']['metrics'].get(k, 0) for k in metric_keys]
        rf_vals  = [data['rf']['metrics'].get(k, 0) for k in metric_keys]

        return {
            'labels': labels,
            'xgb_values': xgb_vals,
            'rf_values': rf_vals,
            'comparison_ready': data['comparison_ready'],
        }

    # ─── Dual Prediction ──────────────────────────────────────────────────────

    def predict_xgb(
        self, input_data: Dict
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Run inference with the XGBoost model.

        Args:
            input_data: Dict matching the primary feature schema.

        Returns:
            List of (career_label, probability) tuples, top-3.
        """
        try:
            self._load_xgb_artifacts()
        except FileNotFoundError as e:
            logger.error(str(e))
            return None

        # Delegate to CareerPredictor to reuse existing encoding logic
        try:
            from src.predictor import CareerPredictor
            predictor = CareerPredictor()
            return predictor.get_top_predictions(input_data, top_n=3)
        except Exception as e:
            logger.error(f"XGB prediction error: {e}")
            return None

    def predict_rf(
        self, aptitude_data: Dict
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Run inference with the Random Forest model on aptitude features.

        Args:
            aptitude_data: Dict with keys matching RF_FEATURE_SCHEMA (ALL_FEATURES).

        Returns:
            List of (career_label, probability) tuples, top-3, or None if unavailable.
        """
        try:
            self._load_rf_artifacts()
        except FileNotFoundError as e:
            logger.error(str(e))
            return None

        try:
            from src.generate_secondary_data import ALL_FEATURES
            X = np.array([[float(aptitude_data.get(f, 50.0)) for f in ALL_FEATURES]])

            proba = self._rf_model.predict_proba(X)[0]
            classes = self._rf_target_enc.classes_

            # Top-3
            top_idx = np.argsort(proba)[::-1][:3]
            return [(classes[i], round(float(proba[i]) * 100, 2)) for i in top_idx]

        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return None

    # ─── Agreement ────────────────────────────────────────────────────────────

    def compute_agreement(
        self,
        xgb_preds: Optional[List[Tuple[str, float]]],
        rf_preds: Optional[List[Tuple[str, float]]],
    ) -> Dict:
        """
        Check if both models agree on the top-1 career.

        Returns:
            {
                'agree': bool,
                'xgb_top': str,
                'rf_top': str,
                'confidence_delta': float (abs diff in top-1 proba)
            }
        """
        if not xgb_preds or not rf_preds:
            return {'agree': False, 'xgb_top': None, 'rf_top': None, 'confidence_delta': 0.0}

        xgb_top = xgb_preds[0][0] if xgb_preds else None
        rf_top  = rf_preds[0][0]  if rf_preds  else None

        # Normalise for comparison (cluster names may differ slightly)
        agree = (xgb_top or '').strip().lower() == (rf_top or '').strip().lower()

        xgb_conf = xgb_preds[0][1] if xgb_preds else 0.0
        rf_conf  = rf_preds[0][1]  if rf_preds  else 0.0
        delta = round(abs(xgb_conf - rf_conf), 2)

        return {
            'agree': agree,
            'xgb_top': xgb_top,
            'rf_top': rf_top,
            'confidence_delta': delta,
        }

    def reload_metadata(self):
        """Force reload of metadata (useful after retraining)."""
        self._load_metadata()
        logger.info("ModelComparator metadata reloaded.")
