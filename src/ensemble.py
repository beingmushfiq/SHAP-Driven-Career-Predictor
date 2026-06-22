"""
Ensemble Predictor Module for Career Predictor.

Combines multiple base classifiers (XGBoost, Random Forest) using various ensemble strategies:
- Soft Voting
- Hard Voting
- Weighted Voting
- Stacking
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__, Config.LOG_DIR)


class EnsemblePredictor:
    """
    Orchestrates creation and wrapping of sklearn VotingClassifier/StackingClassifier.
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        method: str = 'soft_voting',
        weights: Optional[List[float]] = None
    ):
        self.xgb_params = xgb_params or {}
        self.rf_params = rf_params or {}
        self.method = method or Config.ENSEMBLE_METHOD
        self.weights = weights
        self.model = None

    def build(self) -> Any:
        """
        Builds the configured ensemble classifier.
        """
        # Instantiate base estimators
        xgb_base = {**Config.XGB_BASE_PARAMS, **self.xgb_params}
        xgb = XGBClassifier(**xgb_base)

        rf = RandomForestClassifier(
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,
            **self.rf_params
        )

        estimators = [
            ('xgb', xgb),
            ('rf', rf)
        ]

        logger.info(f"Building Ensemble Model using method: {self.method}")

        if self.method == 'soft_voting':
            self.model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=self.weights,
                n_jobs=-1
            )
        elif self.method == 'hard_voting':
            self.model = VotingClassifier(
                estimators=estimators,
                voting='hard',
                weights=self.weights,
                n_jobs=-1
            )
        elif self.method == 'stacking':
            # Use XGBoost as the final estimator
            final_est = XGBClassifier(**Config.XGB_BASE_PARAMS)
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=final_est,
                cv=Config.TUNING_CV_FOLDS,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        return self.model
