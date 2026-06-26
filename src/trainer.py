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
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE  # type: ignore[import-untyped]

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

        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training class distribution: {dict(zip(unique, counts))}")

        return X_train, X_test, y_train, y_test

    def rebalance_data(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply outlier detection and rebalancing to training data.
        
        Steps:
            1. IQR-based outlier removal (class-by-class, 3.0 × IQR)
            2. Isolation Forest for anomaly detection (if enabled)
            3. SMOTE/ADASYN/Borderline-SMOTE oversampling

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Tuple of (X_train_resampled, y_train_resampled).
        """
        logger.info("Applying data cleaning and rebalancing on training set...")

        original_count = X_train.shape[0]
        
        # 1. IQR-based outlier removal (class-by-class)
        keep_indices = []
        for class_label in np.unique(y_train):
            class_mask = (y_train == class_label)
            X_class = X_train[class_mask]
            
            # If class has few samples, keep all of them
            if X_class.shape[0] <= 15:
                keep_indices.extend(np.where(class_mask)[0])
                continue
                
            class_keep_mask = np.ones(X_class.shape[0], dtype=bool)
            for col_idx in range(1, X_train.shape[1]):
                col_data = X_class[:, col_idx]
                q25, q75 = np.percentile(col_data, 25), np.percentile(col_data, 75)
                iqr = q75 - q25
                if iqr <= 0.1:
                    continue
                cut_off = iqr * 3.0
                lower, upper = q25 - cut_off, q75 + cut_off
                
                outliers = (col_data < lower) | (col_data > upper)
                class_keep_mask = class_keep_mask & (~outliers)
                
            # Ensure we keep at least 10 samples
            if np.sum(class_keep_mask) >= 10:
                indices = np.where(class_mask)[0][class_keep_mask]
            else:
                indices = np.where(class_mask)[0]
            keep_indices.extend(indices)

        X_train_clean = X_train[keep_indices]
        y_train_clean = y_train[keep_indices]
        removed_iqr = original_count - X_train_clean.shape[0]
        logger.info(f"IQR-based outlier removal: {original_count} -> {X_train_clean.shape[0]} rows (removed {removed_iqr})")

        # 2. Isolation Forest outlier detection (if enabled)
        if Config.ENABLE_ISOLATION_FOREST and X_train_clean.shape[0] > 50:
            iso_forest = IsolationForest(
                contamination=0.05,  # Expect ~5% outliers
                random_state=Config.RANDOM_SEED,
                n_jobs=-1
            )
            outlier_predictions = iso_forest.fit_predict(X_train_clean)
            clean_mask = outlier_predictions != -1  # -1 indicates outlier
            
            X_train_clean = X_train_clean[clean_mask]
            y_train_clean = y_train_clean[clean_mask]
            removed_iso = X_train_clean.shape[0] - np.sum(clean_mask)
            logger.info(f"Isolation Forest anomaly removal: removed {removed_iso} additional anomalies -> {X_train_clean.shape[0]} rows")

        # 3. Rebalancing with selected method
        rebalance_method = Config.REBALANCE_METHOD.lower()
        
        if rebalance_method == 'adasyn':
            logger.info("Applying ADASYN oversampling")
            resampler = ADASYN(random_state=Config.RANDOM_SEED)
        elif rebalance_method == 'borderline_smote':
            logger.info("Applying Borderline-SMOTE oversampling")
            resampler = BorderlineSMOTE(random_state=Config.RANDOM_SEED)
        else:  # Default to SMOTE
            logger.info("Applying SMOTE oversampling")
            resampler = SMOTE(random_state=Config.RANDOM_SEED)

        try:
            X_train_res, y_train_res = resampler.fit_resample(X_train_clean, y_train_clean)
        except Exception as e:
            logger.warning(f"Resampling failed with {rebalance_method}: {e}. Falling back to SMOTE.")
            smote = SMOTE(random_state=Config.RANDOM_SEED)
            X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train_clean)
        
        unique, counts = np.unique(y_train_res, return_counts=True)
        logger.info(f"After rebalancing: train shape={X_train_res.shape}, classes={dict(zip(unique, counts))}")
        
        return X_train_res, y_train_res

    def build_model(self, params: Optional[Dict] = None, feature_names: Optional[list] = None) -> XGBClassifier:
        """
        Build an XGBoost classifier with base or custom parameters.

        Args:
            params: Optional parameter overrides.
            feature_names: Feature names for monotone constraints mapping

        Returns:
            Configured XGBClassifier (unfitted).
        """
        model_params = {**Config.XGB_BASE_PARAMS}
        if params:
            model_params.update(params)

        # Add monotone constraints if available and feature names provided
        if feature_names and hasattr(Config, 'MONOTONE_CONSTRAINTS') and Config.MONOTONE_CONSTRAINTS:
            constraints = []
            for fname in feature_names:
                constraints.append(Config.MONOTONE_CONSTRAINTS.get(fname, 0))
            if any(c != 0 for c in constraints):
                model_params['monotone_constraints'] = tuple(constraints)
                logger.info(f"Applied monotone constraints to {sum(1 for c in constraints if c != 0)} features")

        model = XGBClassifier(**model_params)
        logger.info(f"Built XGBClassifier with params: {model_params}")
        return model

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict:
        """
        Perform hyperparameter tuning. Supports Bayesian optimization (Optuna)
        and falls back to RandomizedSearchCV.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary of best hyperparameters.
        """
        if Config.TUNING_METHOD == 'bayesian':
            try:
                from src.tuner import tune_xgboost
                return tune_xgboost(X_train, y_train)
            except Exception as e:
                logger.warning(f"Bayesian tuning failed with error: {e}. Falling back to RandomizedSearchCV.")

        logger.info("=" * 60)
        logger.info("STARTING RANDOMIZED HYPERPARAMETER TUNING")
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
        feature_names: Optional[list] = None,
    ) -> XGBClassifier:
        """
        Train XGBoost model with the given hyperparameters.

        Args:
            X_train: Training features.
            y_train: Training labels.
            params: Tuned hyperparameters.
            feature_names: Optional feature names for monotone constraints.

        Returns:
            Fitted XGBClassifier.
        """
        from sklearn.model_selection import train_test_split

        logger.info("Training final model with best parameters...")
        # Add early stopping for final training only
        merged = {**(params or {})}
        merged['early_stopping_rounds'] = Config.EARLY_STOPPING_ROUNDS
        model = self.build_model(merged, feature_names=feature_names)

        # Early stopping: hold out 10% for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train,
            random_state=Config.RANDOM_SEED,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("Model training complete (with early stopping).")
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

    def compute_cv_score_distribution(
        self,
        model: XGBClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict:
        """
        Compute cross-validation score distribution for robustness analysis.
        
        Uses 10-fold stratified CV to assess model generalization.
        
        Args:
            model: Fitted XGBClassifier.
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            Dictionary with CV score statistics.
        """
        from sklearn.base import clone as sklearn_clone

        cv = StratifiedKFold(
            n_splits=Config.TUNING_CV_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_SEED,
        )

        # Clone model without early stopping (cross_val_score has no eval_set)
        cv_model = sklearn_clone(model)
        cv_model.set_params(early_stopping_rounds=None)

        # Compute CV scores using multiple metrics
        metrics_dict = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
        }
        
        cv_results = {}
        logger.info("Computing cross-validation score distribution...")
        
        for metric_name, scoring in metrics_dict.items():
            scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'scores': scores.tolist(),
            }
            logger.info(
                f"  {metric_name}: {scores.mean():.4f} ± {scores.std():.4f} "
                f"(min={scores.min():.4f}, max={scores.max():.4f})"
            )
        
        return cv_results

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
        cv_results: Optional[Dict] = None,
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
            cv_results: Optional cross-validation score distribution.
        """
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, Config.MODEL_PATH)
        logger.info(f"Model saved: {Config.MODEL_PATH}")

        # Save metadata
        import xgboost
        import sklearn

        # Use best_params from tuning, or fall back to model's actual parameters
        if best_params:
            params_to_save = best_params
        else:
            # Only record the user-facing hyperparameters (skip internal/null ones)
            _XGB_KEY_PARAMS = [
                'max_depth', 'learning_rate', 'n_estimators', 'subsample',
                'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha',
                'reg_lambda', 'objective', 'eval_metric', 'tree_method',
            ]
            model_params = model.get_params()
            params_to_save = {k: model_params[k] for k in _XGB_KEY_PARAMS if k in model_params}

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
                           for k, v in params_to_save.items()},
            'metrics': {k: round(float(v), 6) for k, v in metrics.items()},
            'training_data_hash': data_hash,
            'random_seed': Config.RANDOM_SEED,
        }
        
        # Add CV results if available
        if cv_results:
            metadata['cv_results'] = {
                k: {
                    'mean': round(float(v['mean']), 6),
                    'std': round(float(v['std']), 6),
                    'min': round(float(v['min']), 6),
                    'max': round(float(v['max']), 6),
                }
                for k, v in cv_results.items()
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

        # 3. Skip tuning, use base params
        best_params = {}

        # 4. Train
        model = self.train(X_train, y_train, best_params, feature_names=feature_names)

        # 5. Evaluate
        metrics = self.evaluate(model, X_test, y_test, class_names)
        
        # 5.5 Compute CV score distribution
        cv_results = self.compute_cv_score_distribution(model, X_train, y_train)

        # 6. Confusion matrix
        self.generate_confusion_matrix(model, X_test, y_test, class_names)

        # 7. SHAP background
        self.save_background_sample(X_train)

        # 8. Save model + metadata
        self.save_model(
            model, metrics, best_params, feature_names,
            class_names=class_names, data_hash=data_hash, cv_results=cv_results,
        )

        elapsed = time.time() - start_time
        logger.info(f"TRAINING PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return model


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run_pipeline()
