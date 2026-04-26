"""
Model Training Pipeline for Career Predictor.

Handles:
- Train/test splitting with stratification
- XGBoost classifier with RandomizedSearchCV
- Evaluation metrics (accuracy, F1, precision, recall, confusion matrix)
- Model persistence (joblib)
- SHAP background sample generation
- Training metadata tracking

This module is run ONCE during training, never during inference.
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
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
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
from xgboost import XGBClassifier

from src.config import Config
from src.utils import get_logger, set_seeds, compute_data_hash, format_metrics
from src.processor import DataProcessor

logger = get_logger(__name__, Config.LOG_DIR)


class ModelTrainer:
    """
    Complete training pipeline for the XGBoost career predictor.

    Usage:
        trainer = ModelTrainer()
        trainer.run_pipeline()
    """

    def __init__(self):
        """Initialize trainer with config and seed."""
        self.config = Config
        set_seeds(Config.RANDOM_SEED)
        Config.ensure_directories()

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified train/test split.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_SEED,
            stratify=y,
        )
        logger.info(
            f"Data split: train={X_train.shape[0]}, test={X_test.shape[0]} "
            f"(test_size={Config.TEST_SIZE})"
        )

        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training class distribution: {dict(zip(unique, counts))}")

        return X_train, X_test, y_train, y_test

    def build_model(self, params: Optional[Dict] = None) -> XGBClassifier:
        """
        Build an XGBoost classifier with base or custom parameters.

        Args:
            params: Optional parameter overrides.

        Returns:
            Configured XGBClassifier (unfitted).
        """
        model_params = {**Config.XGB_BASE_PARAMS}
        if params:
            model_params.update(params)

        model = XGBClassifier(**model_params)
        logger.info(f"Built XGBClassifier with params: {model_params}")
        return model

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Uses StratifiedKFold to maintain class balance across folds.
        Scoring: f1_weighted (robust to class imbalance).

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary of best hyperparameters.
        """
        logger.info("=" * 60)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info(f"  Iterations: {Config.TUNING_N_ITER}")
        logger.info(f"  CV Folds: {Config.TUNING_CV_FOLDS}")
        logger.info("=" * 60)

        base_model = self.build_model()
        cv = StratifiedKFold(
            n_splits=Config.TUNING_CV_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_SEED,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=Config.PARAM_GRID,
            n_iter=Config.TUNING_N_ITER,
            scoring='f1_weighted',
            cv=cv,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,
            verbose=1,
        )

        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        logger.info(f"Tuning completed in {elapsed:.1f}s")
        logger.info(f"Best score (CV): {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")

        return search.best_params_

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Dict,
    ) -> XGBClassifier:
        """
        Train XGBoost model with the given hyperparameters.

        Args:
            X_train: Training features.
            y_train: Training labels.
            params: Tuned hyperparameters.

        Returns:
            Fitted XGBClassifier.
        """
        logger.info("Training final model with best parameters...")
        model = self.build_model(params)
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model

    def evaluate(
        self,
        model: XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None,
    ) -> Dict:
        """
        Evaluate model on test set.

        Computes accuracy, precision, recall, F1, and classification report.

        Args:
            model: Fitted XGBClassifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Optional list of class label names.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model on test set...")
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        logger.info(format_metrics(metrics))

        # Classification report
        target_names = class_names if class_names else None
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            zero_division=0,
        )
        logger.info(f"\nClassification Report:\n{report}")

        return metrics

    def generate_confusion_matrix(
        self,
        model: XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None,
    ) -> Path:
        """
        Generate and save confusion matrix plot.

        Args:
            model: Fitted model.
            X_test: Test features.
            y_test: Test labels.
            class_names: Optional class label names.

        Returns:
            Path to saved confusion matrix image.
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        ax.set_title('Confusion Matrix — XGBoost Career Predictor', fontsize=14)
        plt.tight_layout()

        save_path = Config.SHAP_PLOTS_DIR / 'confusion_matrix.png'
        Config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Confusion matrix saved: {save_path}")

        return save_path

    def save_background_sample(
        self, X_train: np.ndarray, n_samples: int = None
    ) -> Path:
        """
        Save a background sample from training data for SHAP.

        Args:
            X_train: Training feature matrix.
            n_samples: Number of samples (default from config).

        Returns:
            Path to saved background sample.
        """
        n_samples = n_samples or Config.SHAP_BACKGROUND_SIZE
        n_samples = min(n_samples, X_train.shape[0])

        indices = np.random.choice(X_train.shape[0], size=n_samples, replace=False)
        background = X_train[indices]

        np.save(Config.SHAP_BACKGROUND_PATH, background)
        logger.info(
            f"SHAP background sample saved: {Config.SHAP_BACKGROUND_PATH} "
            f"(shape={background.shape})"
        )
        return Config.SHAP_BACKGROUND_PATH

    def save_model(
        self,
        model: XGBClassifier,
        metrics: Dict,
        best_params: Dict,
        feature_names: list,
        class_names: Optional[list] = None,
        data_hash: str = "",
    ) -> None:
        """
        Persist model and training metadata.

        Args:
            model: Fitted XGBClassifier.
            metrics: Evaluation metrics dictionary.
            best_params: Best hyperparameters from tuning.
            feature_names: Ordered feature names.
            class_names: Target class labels.
            data_hash: SHA256 hash of training data.
        """
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, Config.MODEL_PATH)
        logger.info(f"Model saved: {Config.MODEL_PATH}")

        # Save metadata
        import xgboost
        import sklearn
        metadata = {
            'version': '1.0.0',
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'python_version': f"{__import__('sys').version}",
            'xgboost_version': xgboost.__version__,
            'sklearn_version': sklearn.__version__,
            'n_features': len(feature_names),
            'n_classes': len(class_names) if class_names else int(model.n_classes_),
            'class_labels': list(class_names) if class_names else [],
            'feature_names': feature_names,
            'best_params': {k: (int(v) if isinstance(v, np.integer) else
                               float(v) if isinstance(v, np.floating) else v)
                           for k, v in best_params.items()},
            'metrics': {k: round(float(v), 6) for k, v in metrics.items()},
            'training_data_hash': data_hash,
            'random_seed': Config.RANDOM_SEED,
        }

        with open(Config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved: {Config.METADATA_PATH}")

    def run_pipeline(self, data_path: Optional[Path] = None) -> XGBClassifier:
        """
        Execute the complete training pipeline.

        Steps:
            1. Process data (clean, encode, split)
            2. Tune hyperparameters
            3. Train final model
            4. Evaluate on test set
            5. Generate confusion matrix
            6. Save SHAP background sample
            7. Persist model + metadata

        Args:
            data_path: Optional specific data file path.

        Returns:
            Trained XGBClassifier.
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()

        # 1. Process data
        processor = DataProcessor()
        X, y, feature_names = processor.process_pipeline(data_path)

        # Get class names from target encoder
        class_names = None
        if processor.target_encoder is not None:
            class_names = list(processor.target_encoder.classes_)

        # Compute data hash for versioning
        csv_files = list(Config.DATA_RAW_DIR.glob('*.csv'))
        data_hash = compute_data_hash(csv_files[0]) if csv_files else ""

        # 2. Split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 3. Tune
        best_params = self.tune_hyperparameters(X_train, y_train)

        # 4. Train
        model = self.train(X_train, y_train, best_params)

        # 5. Evaluate
        metrics = self.evaluate(model, X_test, y_test, class_names)

        # 6. Confusion matrix
        self.generate_confusion_matrix(model, X_test, y_test, class_names)

        # 7. SHAP background
        self.save_background_sample(X_train)

        # 8. Save model + metadata
        self.save_model(
            model, metrics, best_params, feature_names,
            class_names=class_names, data_hash=data_hash,
        )

        elapsed = time.time() - start_time
        logger.info(f"TRAINING PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return model


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run_pipeline()
