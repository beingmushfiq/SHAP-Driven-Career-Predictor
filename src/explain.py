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

        # Pass data=None so SHAP takes the XGBoost-native fast path, which
        # is the only path that correctly handles categorical splits.
        # (Passing background data causes fallthrough to C-ext that raises
        # NotImplementedError for categorical models.)
        self._explainer = shap.TreeExplainer(
            self._model,
            feature_perturbation="tree_path_dependent"
        )
        self._loaded = True
        logger.info(f"SHAP explainer initialized (background shape: {self._background.shape})")

    def compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for input X with caching.
        Returns a raw numpy array: (samples, features) or (samples, features, classes).
        Uses .shap_values() instead of .__call__() to avoid the Explanation
        wrapper path that raises NotImplementedError for categorical XGBoost models.
        """
        if not self._loaded:
            raise RuntimeError("SHAP explainer not initialized.")
        
        # Simple cache to avoid re-computing for same input in one request
        if self._last_X is not None and np.array_equal(X, self._last_X):
            return self._last_shap_values

        raw_sv = self._explainer.shap_values(X)
        # Normalise: if SHAP returns a list (one array per class), stack to 3-D array
        if isinstance(raw_sv, list):
            raw_sv = np.stack(raw_sv, axis=-1)  # (samples, features, classes)
        self._last_X = X.copy()
        self._last_shap_values = raw_sv
        return raw_sv

    def global_summary_plot(self, X_sample: Optional[np.ndarray] = None) -> Path:
        """
        Generate and save global SHAP summary bar plot.
        Uses background data if no sample provided.
        """
        if X_sample is None:
            X_sample = self._background

        logger.info("Generating global SHAP summary plot...")
        # Use shap_values() (returns raw array) instead of __call__() to avoid
        # the Explanation wrapper path which also checks categorical support.
        raw_sv = self._explainer.shap_values(X_sample)

        # Handle multiclass: shap_values() returns (samples, features, classes)
        # Compute mean |SHAP| across classes → (samples, features)
        if isinstance(raw_sv, list):
            # older SHAP versions return a list of arrays, one per class
            mean_abs_sv = np.abs(np.stack(raw_sv, axis=-1)).mean(axis=2)
        elif len(raw_sv.shape) == 3:
            mean_abs_sv = np.abs(raw_sv).mean(axis=2)
        else:
            mean_abs_sv = np.abs(raw_sv)

        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = Config.SHAP_PLOTS_DIR / 'global_summary.png'

        fig, ax = plt.subplots(figsize=(12, 8))
        # Build a simple horizontal bar chart from the mean absolute SHAP values
        feature_importance = mean_abs_sv.mean(axis=0)
        sort_idx = np.argsort(feature_importance)
        sorted_names = [self._feature_names[i] for i in sort_idx]
        sorted_vals = feature_importance[sort_idx]
        ax.barh(sorted_names, sorted_vals, color='#e74c3c', alpha=0.85)
        ax.set_xlabel('Mean |SHAP Value|')
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
        # raw_sv is a numpy array: (1, features) or (1, features, classes)
        raw_sv = self.compute_shap_values(X_single)

        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = Config.SHAP_PLOTS_DIR / f'shap_waterfall_{prediction_id}.png'

        # Extract SHAP values for the predicted class
        if raw_sv.ndim == 3:
            # (1, features, classes) — pick the class with highest summed SHAP
            predicted_class = int(np.argmax(raw_sv[0].sum(axis=0)))
            sv_values = raw_sv[0, :, predicted_class]          # (features,)
        else:
            sv_values = raw_sv[0]                              # (features,)

        # Build a waterfall-style bar chart manually (shap.plots.waterfall
        # requires an Explanation object which needs the __call__ path).
        fig, ax = plt.subplots(figsize=(12, 8))
        sort_idx = np.argsort(np.abs(sv_values))
        sorted_names = [self._feature_names[i] for i in sort_idx]
        sorted_vals = sv_values[sort_idx]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in sorted_vals]
        ax.barh(sorted_names, sorted_vals, color=colors, alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on model output)')
        plt.title('How This Prediction Was Made (SHAP)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        logger.info(f"Waterfall plot saved: {save_path}")
        return save_path

    def get_local_interpretations(self, X_single: np.ndarray, class_idx: Optional[int] = None) -> List[dict]:
        """
        Get the top influential features and their SHAP impacts for a single prediction.
        Returns only features that positively support the predicted career.

        Returns a list of dicts:
            [{'feature': 'Coding Skills', 'impact': 22.5}, ...]
        where impact is always a positive percentage representing how much that
        feature contributed *towards* the predicted career.
        """
        raw_sv = self.compute_shap_values(X_single)

        # For multiclass, use the predicted class if class_idx not provided
        if raw_sv.ndim == 3:
            target_class = class_idx if class_idx is not None else int(np.argmax(raw_sv[0].sum(axis=0)))
            sv_values = raw_sv[0, :, target_class]   # (features,)
        else:
            sv_values = raw_sv[0]                    # (features,)

        # Calculate contribution percentages using only positive SHAP values
        # (features that push the prediction TOWARDS the predicted class)
        positive_sum = np.sum(sv_values[sv_values > 0]) + 1e-10

        interpretations = []
        for i, val in enumerate(sv_values):
            if val > 0:
                impact_pct = (val / positive_sum) * 100
                interpretations.append({
                    'feature': self._feature_names[i].replace('_', ' ').title(),
                    'impact': round(impact_pct, 2),
                })

        # Sort by impact descending (strongest positive contributors first)
        interpretations.sort(key=lambda x: x['impact'], reverse=True)
        return interpretations

    def feature_importance_comparison(self) -> Path:
        """
        Generate comparison plot: XGBoost built-in importance vs SHAP.
        """
        logger.info("Generating feature importance comparison plot...")

        # XGBoost built-in importance
        xgb_importance = self._model.feature_importances_

        # SHAP-based importance (mean |SHAP value|)
        raw_sv = self._explainer.shap_values(self._background)
        if isinstance(raw_sv, list):
            shap_importance = np.abs(np.stack(raw_sv, axis=-1)).mean(axis=(0, 2))
        elif len(raw_sv.shape) == 3:
            shap_importance = np.abs(raw_sv).mean(axis=(0, 2))
        else:
            shap_importance = np.abs(raw_sv).mean(axis=0)

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


def select_features_by_shap(
    model,
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.005
) -> List[str]:
    """
    Ranks features by mean |SHAP| values and returns feature names above a threshold.
    """
    logger.info("Computing SHAP feature importance for feature selection...")
    # Use a subset of background data for speed in TreeExplainer
    bg_sample = X[:min(200, X.shape[0])]
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    raw_sv = explainer.shap_values(X[:min(500, X.shape[0])])
    if isinstance(raw_sv, list):
        raw_sv = np.stack(raw_sv, axis=-1)  # (samples, features, classes)

    if raw_sv.ndim == 3:
        shap_importance = np.abs(raw_sv).mean(axis=(0, 2))
    else:
        shap_importance = np.abs(raw_sv).mean(axis=0)
        
    # Normalize to relative contribution
    shap_norm = shap_importance / (shap_importance.sum() + 1e-10)
    
    kept_features = []
    for i, name in enumerate(feature_names):
        importance = shap_norm[i]
        logger.info(f"Feature: {name}, SHAP relative importance: {importance:.4f}")
        if importance >= threshold:
            kept_features.append(name)
            
    logger.info(f"SHAP feature selection kept {len(kept_features)}/{len(feature_names)} features (threshold={threshold}).")
    return kept_features

