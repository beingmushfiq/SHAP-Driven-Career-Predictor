"""
Bayesian Hyperparameter Tuning using Optuna.

Optimizes hyperparameters for XGBoost and Random Forest classifiers.
"""

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__, Config.LOG_DIR)

# Suppress optuna logs to avoid clutter, showing only warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using Optuna.
    """
    logger.info("=" * 60)
    logger.info("STARTING BAYESIAN HYPERPARAMETER TUNING FOR XGBOOST")
    logger.info(f"  Optuna Trials: {Config.OPTUNA_N_TRIALS}")
    logger.info(f"  CV Folds: {Config.TUNING_CV_FOLDS}")
    logger.info("=" * 60)

    cv = StratifiedKFold(
        n_splits=Config.TUNING_CV_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_SEED,
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500, 700]),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            **Config.XGB_BASE_PARAMS
        }
        
        model = XGBClassifier(**params)
        
        # Use cross_val_score with f1_weighted
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS)

    logger.info(f"Bayesian Tuning Completed. Best CV F1 Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    return study.best_params


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters using Optuna.
    """
    logger.info("=" * 60)
    logger.info("STARTING BAYESIAN HYPERPARAMETER TUNING FOR RANDOM FOREST")
    logger.info(f"  Optuna Trials: {Config.OPTUNA_N_TRIALS}")
    logger.info(f"  CV Folds: {Config.TUNING_CV_FOLDS}")
    logger.info("=" * 60)

    cv = StratifiedKFold(
        n_splits=Config.TUNING_CV_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_SEED,
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, 20, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': Config.RANDOM_SEED,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS)

    logger.info(f"Bayesian Tuning Completed. Best CV F1 Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    return study.best_params
