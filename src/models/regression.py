"""
Regression models: predict forward log return.

Two models per horizon:
- RidgeRegressor   : L2-regularized linear baseline. Fast, interpretable.
                     Sets the performance floor — LightGBM must beat this.
- LGBMRegressor    : LightGBM gradient boosting. Main model. Uses early stopping
                     on the validation set to prevent overfitting.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

from src.config import LGBM_EARLY_STOPPING_ROUNDS, LGBM_PARAMS, RIDGE_ALPHA
from src.models.base import BaseModel


class RidgeRegressor(BaseModel):
    """
    Ridge regression baseline.

    Why Ridge over OLS? Financial features are often correlated (momentum at
    different horizons, volatility at different windows). L2 regularization
    handles multicollinearity gracefully and prevents the model from
    over-committing to any single feature.

    Requires pre-scaled features (StandardScaler from preprocessor.py).
    """

    model_name = "ridge"

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "RidgeRegressor":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coef_(self):
        return self._model.coef_

    @property
    def feature_names_in_(self):
        return getattr(self._model, "feature_names_in_", None)


class LGBMRegressor(BaseModel):
    """
    LightGBM gradient-boosted tree regressor. Main model.

    Key configuration choices (all in config.py):
    - num_leaves=31   : Default; kept conservative to avoid overfitting on
                        ~1000-sample training sets.
    - min_child_samples=20 : Each leaf must cover at least 20 samples — the
                              strongest single guard against overfitting.
    - early stopping  : Training stops when val RMSE doesn't improve for
                        LGBM_EARLY_STOPPING_ROUNDS rounds, then best iteration
                        is restored automatically.
    - subsample + colsample_bytree=0.8 : Row and feature subsampling add
                                          regularization similar to random forests.

    Does NOT require feature scaling (trees are invariant to monotonic transforms).
    """

    model_name = "lgbm"

    def __init__(self, params: dict | None = None):
        self.params = {**LGBM_PARAMS, "objective": "regression", "metric": "rmse"}
        if params:
            self.params.update(params)
        self._model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LGBMRegressor":
        callbacks = [
            lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),   # suppress per-iteration output
        ]
        eval_set = [(X_val, y_val)] if X_val is not None else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks if eval_set else None,
            )
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def best_iteration_(self):
        return getattr(self._model, "best_iteration_", None)
