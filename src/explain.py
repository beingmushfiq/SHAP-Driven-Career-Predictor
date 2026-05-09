"""
SHAP Explainability Layer for Career Predictor.

Provides:
- Cached TreeExplainer (singleton, never recomputed per request)
- Global SHAP summary plot
- Local SHAP waterfall plot per prediction
- XGBoost vs SHAP feature importance comparison
"""

import uuid
import threading
from pathlib import Path
from typing import Optional, List

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__, Config.LOG_DIR)


class SHAPExplainer:
    """
    Singleton SHAP explainability engine.
    Caches the TreeExplainer so it is never recomputed per request.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._explainer = None
        self._model = None
        self._background = None
        self._feature_names: List[str] = []
        self._loaded = False
        self._last_shap_values = None
        self._last_X = None

    @classmethod
    def get_instance(cls) -> 'SHAPExplainer':
        """Get or create the singleton SHAP explainer."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Load model, background data, and create TreeExplainer."""
        logger.info("Initializing SHAP explainer...")

        import json
        if not Config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}")
        if not Config.SHAP_BACKGROUND_PATH.exists():
            raise FileNotFoundError(f"Background data not found: {Config.SHAP_BACKGROUND_PATH}")

        self._model = joblib.load(Config.MODEL_PATH)
        self._background = np.load(Config.SHAP_BACKGROUND_PATH)

        with open(Config.FEATURE_SCHEMA_PATH, 'r') as f:
            schema = json.load(f)
        self._feature_names = schema['feature_names']

        self._explainer = shap.TreeExplainer(self._model, self._background)
        self._loaded = True
        logger.info(f"SHAP explainer initialized (background shape: {self._background.shape})")

    def compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for input X with caching."""
        if not self._loaded:
            raise RuntimeError("SHAP explainer not initialized.")
        
        # Simple cache to avoid re-computing for same input in one request
        if self._last_X is not None and np.array_equal(X, self._last_X):
            return self._last_shap_values
            
        shap_values = self._explainer(X)
        self._last_X = X.copy()
        self._last_shap_values = shap_values
        return shap_values

    def global_summary_plot(self, X_sample: Optional[np.ndarray] = None) -> Path:
        """
        Generate and save global SHAP summary bar plot.
        Uses background data if no sample provided.
        """
        if X_sample is None:
            X_sample = self._background

        logger.info("Generating global SHAP summary plot...")
        shap_values = self._explainer(X_sample)

        # Handle multiclass: take the mean absolute SHAP value across classes
        if len(shap_values.shape) == 3:
            # shap_values.values is (samples, features, classes)
            # We want (samples, features) where each value is mean(|SHAP|) across classes
            import copy
            sv_copy = copy.deepcopy(shap_values)
            sv_copy.values = np.abs(sv_copy.values).mean(axis=2)
            # Also update base_values if needed, but for bar plot it's not strictly necessary
            # if hasattr(sv_copy, 'base_values'):
            #     sv_copy.base_values = sv_copy.base_values.mean(axis=1)
            shap_values = sv_copy

        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = Config.SHAP_PLOTS_DIR / 'global_summary.png'

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.bar(shap_values, max_display=len(self._feature_names), show=False)
        plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        logger.info(f"Global summary plot saved: {save_path}")
        return save_path

    def local_waterfall_plot(self, X_single: np.ndarray, prediction_id: str = None) -> Path:
        """
        Generate and save SHAP waterfall plot for a single prediction.

        Args:
            X_single: Single input vector of shape (1, n_features).
            prediction_id: Unique ID for the plot filename.

        Returns:
            Path to saved waterfall plot.
        """
        if prediction_id is None:
            prediction_id = str(uuid.uuid4())[:8]

        logger.info(f"Generating SHAP waterfall plot (id={prediction_id})...")
        shap_values = self.compute_shap_values(X_single)

        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = Config.SHAP_PLOTS_DIR / f'shap_waterfall_{prediction_id}.png'

        fig, ax = plt.subplots(figsize=(12, 8))
        # For multiclass, use the predicted class
        if len(shap_values.shape) == 3:
            predicted_class = np.argmax(shap_values.values[0].sum(axis=0))
            sv = shap_values[0, :, predicted_class]
        else:
            sv = shap_values[0]

        shap.plots.waterfall(sv, max_display=len(self._feature_names), show=False)
        plt.title('How This Prediction Was Made (SHAP)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        logger.info(f"Waterfall plot saved: {save_path}")
        return save_path

    def get_local_interpretations(self, X_single: np.ndarray) -> List[dict]:
        """
        Get the top influential features and their SHAP impacts for a single prediction.
        Returns a list of dicts: [{'feature': 'Math', 'impact': 12.5}, ...]
        """
        shap_values = self.compute_shap_values(X_single)
        
        # For multiclass, use the predicted class
        if len(shap_values.shape) == 3:
            predicted_class = np.argmax(shap_values.values[0].sum(axis=0))
            sv_values = shap_values.values[0, :, predicted_class]
        else:
            sv_values = shap_values.values[0]

        # Calculate percentages based on sum of absolute SHAP values
        total_impact = np.sum(np.abs(sv_values)) + 1e-10
        
        interpretations = []
        for i, val in enumerate(sv_values):
            if val != 0:
                impact_pct = (val / total_impact) * 100
                interpretations.append({
                    'feature': self._feature_names[i].replace('_', ' ').title(),
                    'impact': impact_pct
                })
        
        # Sort by absolute impact descending
        interpretations.sort(key=lambda x: abs(x['impact']), reverse=True)
        return interpretations

    def feature_importance_comparison(self) -> Path:
        """
        Generate comparison plot: XGBoost built-in importance vs SHAP.
        """
        logger.info("Generating feature importance comparison plot...")

        # XGBoost built-in importance
        xgb_importance = self._model.feature_importances_

        # SHAP-based importance (mean |SHAP value|)
        shap_values = self._explainer(self._background)
        if len(shap_values.shape) == 3:
            shap_importance = np.abs(shap_values.values).mean(axis=(0, 2))
        else:
            shap_importance = np.abs(shap_values.values).mean(axis=0)

        # Normalize both to [0, 1]
        xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)
        shap_norm = shap_importance / (shap_importance.max() + 1e-10)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Sort by SHAP importance
        sort_idx = np.argsort(shap_norm)
        names_sorted = [self._feature_names[i] for i in sort_idx]

        axes[0].barh(names_sorted, xgb_norm[sort_idx], color='#4a90d9', alpha=0.85)
        axes[0].set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Normalized Importance')

        axes[1].barh(names_sorted, shap_norm[sort_idx], color='#e74c3c', alpha=0.85)
        axes[1].set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Mean |SHAP Value| (Normalized)')

        plt.suptitle(
            'XGBoost vs SHAP Feature Importance Comparison',
            fontsize=15, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        save_path = Config.SHAP_PLOTS_DIR / 'feature_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Comparison plot saved: {save_path}")
        return save_path

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
