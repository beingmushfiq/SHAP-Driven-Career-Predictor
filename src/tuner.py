"""
Bayesian Hyperparameter Tuning using Optuna.

Optimizes hyperparameters for XGBoost and Random Forest classifiers.
"""

import optuna  # type: ignore[import-untyped]
from optuna.pruners import HyperbandPruner
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
    Tune XGBoost hyperparameters using Optuna with expanded search space.
    
    Includes:
    - 500 trials (up from 100) for better optimization
    - MedianPruner for early stopping of unpromising trials
    - Expanded parameter space: colsample_bylevel, colsample_bynode, max_delta_step
    - 10-fold stratified CV (up from 5) for robustness
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
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200]),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 3),
            **Config.XGB_BASE_PARAMS
        }
        
        model = XGBClassifier(**params)
        
        # Use cross_val_score with f1_weighted and 10-fold CV
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        mean_score = scores.mean()
        
        # Report intermediate score for pruning
        trial.report(mean_score, step=0)
        
        return mean_score

    # Create study with HyperbandPruner for early stopping
    study = optuna.create_study(
        direction='maximize',
        pruner=HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3)
    )
    study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, show_progress_bar=False)

    logger.info(f"Bayesian Tuning Completed. Best CV F1 Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info(f"Number of trials completed: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters using Optuna with expanded search space.
    
    Includes:
    - 500 trials (up from 100) for parity with XGBoost
    - MedianPruner for early stopping
    - 10-fold stratified CV for robustness
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
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500, 1000]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, 20, 30, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': Config.RANDOM_SEED,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        mean_score = scores.mean()
        
        # Report intermediate score for pruning
        trial.report(mean_score, step=0)
        
        return mean_score

    # Create study with HyperbandPruner for early stopping
    study = optuna.create_study(
        direction='maximize',
        pruner=HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3)
    )
    study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, show_progress_bar=False)

    logger.info(f"Bayesian Tuning Completed. Best CV F1 Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info(f"Number of trials completed: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params
