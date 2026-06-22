"""
Random Forest Training Pipeline (Secondary Model).

Trains a Random Forest Classifier on the aptitude/personality secondary dataset
(`career_aptitude_dataset.csv`) independently of the primary XGBoost pipeline.

Key differences from trainer.py:
  - Uses aptitude + personality features (no field/GPA)
  - All-numerical features → no LabelEncoder needed for input
  - Target column: 'career_cluster' (21 classes)
  - Bayesian-tuned via tune_random_forest()
  - Artifacts saved to rf_* paths defined in Config

Run:
    python -m src.rf_trainer
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]

from src.config import Config
from src.utils import get_logger, set_seeds, format_metrics
from src.generate_secondary_data import generate_secondary_data, ALL_FEATURES

logger = get_logger(__name__, Config.LOG_DIR)


class RFModelTrainer:
    """
    Complete training pipeline for the Random Forest secondary model.

    Trains on aptitude / personality data (career_aptitude_dataset.csv),
    producing a separate set of model artifacts distinct from the XGBoost model.

    Usage:
        trainer = RFModelTrainer()
        trainer.run_pipeline()
    """

    # Output feature names (all numerical, ordered)
    FEATURE_NAMES: list = ALL_FEATURES

    def __init__(self):
        self.config = Config
        set_seeds(Config.RANDOM_SEED)
        Config.ensure_directories()
        self.target_encoder = LabelEncoder()

    # ─── Data Loading ────────────────────────────────────────────────────────

    def load_or_generate_data(self) -> pd.DataFrame:
        """
        Load the secondary dataset from disk, or generate it if missing.
        """
        csv_path = Config.DATA_RAW_DIR / 'career_aptitude_dataset.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded secondary dataset from {csv_path} — shape: {df.shape}")
        else:
            logger.info("Secondary dataset not found — generating now...")
            df = generate_secondary_data(n_samples=8400)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved generated dataset to {csv_path}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to (X, y) arrays.

        Args:
            df: Raw DataFrame with feature columns + 'career_cluster'.

        Returns:
            Tuple of (X: float array, y: int encoded labels).
        """
        # Drop any rows with null
        df = df.dropna().reset_index(drop=True)
        logger.info(f"After null drop: {df.shape}")

        X = df[self.FEATURE_NAMES].values.astype(float)
        y_raw = df['career_cluster'].values

        # Encode target
        y = self.target_encoder.fit_transform(y_raw)
        logger.info(
            f"Encoded {len(self.target_encoder.classes_)} career clusters. "
            f"X shape: {X.shape}"
        )

        # Save target encoder
        joblib.dump(self.target_encoder, Config.RF_TARGET_ENCODER_PATH)
        logger.info(f"RF target encoder saved: {Config.RF_TARGET_ENCODER_PATH}")

        # Save feature schema
        schema = {
            'feature_names': self.FEATURE_NAMES,
            'n_features': len(self.FEATURE_NAMES),
            'target_classes': list(self.target_encoder.classes_),
        }
        with open(Config.RF_FEATURE_SCHEMA_PATH, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"RF feature schema saved: {Config.RF_FEATURE_SCHEMA_PATH}")

        return X, y

    # ─── Rebalancing ─────────────────────────────────────────────────────────

    def rebalance_data(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply class-by-class IQR cleaning + SMOTE oversampling."""
        logger.info("Applying data cleaning + SMOTE on RF training set...")
        original_count = X_train.shape[0]
        keep_indices = []

        for cls in np.unique(y_train):
            mask = y_train == cls
            X_cls = X_train[mask]

            if X_cls.shape[0] <= 15:
                keep_indices.extend(np.where(mask)[0])
                continue

            cls_keep = np.ones(X_cls.shape[0], dtype=bool)
            for col_idx in range(X_cls.shape[1]):
                col = X_cls[:, col_idx]
                q25, q75 = np.percentile(col, 25), np.percentile(col, 75)
                iqr = q75 - q25
                if iqr <= 0.1:
                    continue
                cut_off = iqr * 3.0
                cls_keep &= (col >= q25 - cut_off) & (col <= q75 + cut_off)

            idx = np.where(mask)[0]
            keep_indices.extend(idx[cls_keep] if cls_keep.sum() >= 10 else idx)

        X_clean = X_train[keep_indices]
        y_clean = y_train[keep_indices]
        logger.info(
            f"IQR cleaning: {original_count} -> {X_clean.shape[0]} rows "
            f"(dropped {original_count - X_clean.shape[0]})"
        )

        smote = SMOTE(random_state=Config.RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X_clean, y_clean)

        unique, counts = np.unique(y_res, return_counts=True)
        logger.info(
            f"After SMOTE: shape={X_res.shape}, "
            f"classes={dict(zip(unique.tolist(), counts.tolist()))}"
        )
        return X_res, y_res

    # ─── Hyperparameter Tuning ───────────────────────────────────────────────

    def tune_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict:
        """Bayesian tuning for Random Forest via Optuna."""
        try:
            from src.tuner import tune_random_forest
            return tune_random_forest(X_train, y_train)
        except Exception as e:
            logger.warning(f"RF Bayesian tuning failed: {e}. Using default params.")
            return {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
            }

    # ─── Training ────────────────────────────────────────────────────────────

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, params: Dict
    ) -> RandomForestClassifier:
        """Train final Random Forest with tuned parameters."""
        logger.info("Training final Random Forest with best params...")

        full_params = {
            'random_state': Config.RANDOM_SEED,
            'n_jobs': -1,
            **params,
        }
        # Remove None class_weight (RF default)
        if full_params.get('class_weight') is None:
            full_params.pop('class_weight', None)

        model = RandomForestClassifier(**full_params)
        model.fit(X_train, y_train)
        logger.info("RF training complete.")
        return model

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Compute accuracy, F1, precision, recall on test set."""
        logger.info("Evaluating RF on test set...")
        y_pred = model.predict(X_test)
        class_names = list(self.target_encoder.classes_)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        logger.info(format_metrics(metrics))
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        logger.info(f"\nRF Classification Report:\n{report}")

        # Cross-val accuracy on full training (after fit) for reporting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
        cv_scores = cross_val_score(
            RandomForestClassifier(**{
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'max_features': model.max_features,
                'class_weight': model.class_weight,
                'random_state': Config.RANDOM_SEED,
                'n_jobs': -1,
            }),
            X_test, y_test, cv=cv, scoring='accuracy', n_jobs=-1
        )
        metrics['cv_accuracy_mean'] = float(cv_scores.mean())
        metrics['cv_accuracy_std'] = float(cv_scores.std())
        logger.info(
            f"CV Accuracy (5-fold on test): {metrics['cv_accuracy_mean']:.4f} "
            f"± {metrics['cv_accuracy_std']:.4f}"
        )

        return metrics

    def generate_confusion_matrix(
        self,
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Path:
        """Save confusion matrix plot for RF model."""
        y_pred = model.predict(X_test)
        class_names = list(self.target_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Greens', xticks_rotation=45)
        ax.set_title('Confusion Matrix — Random Forest Career Predictor', fontsize=14)
        plt.tight_layout()

        save_path = Config.SHAP_PLOTS_DIR / 'rf_confusion_matrix.png'
        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"RF Confusion matrix saved: {save_path}")
        return save_path

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_background_sample(self, X_train: np.ndarray) -> Path:
        """Save SHAP background sample for RF model."""
        n_samples = min(Config.SHAP_BACKGROUND_SIZE, X_train.shape[0])
        idx = np.random.choice(X_train.shape[0], size=n_samples, replace=False)
        background = X_train[idx]
        np.save(Config.RF_SHAP_BACKGROUND_PATH, background)
        logger.info(f"RF SHAP background saved: {Config.RF_SHAP_BACKGROUND_PATH} shape={background.shape}")
        return Config.RF_SHAP_BACKGROUND_PATH

    def save_model(
        self,
        model: RandomForestClassifier,
        metrics: Dict,
        best_params: Dict,
    ) -> None:
        """Persist RF model + metadata."""
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, Config.RF_MODEL_PATH)
        logger.info(f"RF model saved: {Config.RF_MODEL_PATH}")

        import sklearn
        metadata = {
            'version': '1.0.0',
            'model_type': 'RandomForestClassifier',
            'dataset': 'career_aptitude_dataset.csv',
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'sklearn_version': sklearn.__version__,
            'n_features': len(self.FEATURE_NAMES),
            'feature_names': self.FEATURE_NAMES,
            'n_classes': len(self.target_encoder.classes_),
            'class_labels': list(self.target_encoder.classes_),
            'best_params': {
                k: (int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else v)
                for k, v in best_params.items()
            },
            'metrics': {k: round(float(v), 6) for k, v in metrics.items()},
            'random_seed': Config.RANDOM_SEED,
        }

        with open(Config.RF_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"RF metadata saved: {Config.RF_METADATA_PATH}")

    # ─── Full Pipeline ────────────────────────────────────────────────────────

    def run_pipeline(self) -> RandomForestClassifier:
        """
        Execute the complete RF training pipeline:
          1. Load / generate secondary dataset
          2. Encode target
          3. Train/test split (stratified)
          4. IQR cleaning + SMOTE
          5. Bayesian hyperparameter tuning
          6. Train final model
          7. Evaluate
          8. Confusion matrix
          9. SHAP background
          10. Save model + metadata
        """
        logger.info("=" * 60)
        logger.info("STARTING RF TRAINING PIPELINE (SECONDARY MODEL)")
        logger.info("=" * 60)
        start_time = time.time()

        # 1. Data
        df = self.load_or_generate_data()

        # 2. Prepare
        X, y = self.prepare_data(df)

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_SEED,
            stratify=y,
        )
        logger.info(f"RF split: train={X_train.shape[0]}, test={X_test.shape[0]}")

        # 4. Rebalance
        X_train, y_train = self.rebalance_data(X_train, y_train)

        # 5. Tune
        best_params = self.tune_hyperparameters(X_train, y_train)

        # 6. Train
        model = self.train(X_train, y_train, best_params)

        # 7. Evaluate
        metrics = self.evaluate(model, X_test, y_test)

        # 8. Confusion matrix
        self.generate_confusion_matrix(model, X_test, y_test)

        # 9. SHAP background
        self.save_background_sample(X_train)

        # 10. Save
        self.save_model(model, metrics, best_params)

        elapsed = time.time() - start_time
        logger.info(f"RF TRAINING PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info("=" * 60)

        return model


if __name__ == '__main__':
    trainer = RFModelTrainer()
    trainer.run_pipeline()
