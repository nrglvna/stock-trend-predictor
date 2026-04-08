"""
Classification models: predict direction (Up=1, Down=0).

Two models per horizon:
- LogisticClassifier : L2-regularized logistic regression. Main model.
                       Outputs calibrated probabilities for threshold tuning.
- LGBMClassifier     : LightGBM gradient boosting. Comparison model.
                       Included to show why complex models overfit on small data.

Threshold tuning:
  Use find_optimal_threshold(y_val, probas_val) to find the P(Up) cutoff that
  maximises macro F1 on the validation set, then call predict_with_threshold()
  to apply it to the test set. This balances Up and Down recall without
  degrading AUC (which is threshold-independent).
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import LGBM_EARLY_STOPPING_ROUNDS, LGBM_PARAMS, RIDGE_ALPHA
from src.models.base import BaseModel


def find_optimal_threshold(y_true: np.ndarray, probas: np.ndarray,
                            metric: str = "f1_macro") -> float:
    """
    Scan thresholds [0.30, 0.70] and return the one that maximises the
    chosen metric on a held-out validation set.

    Why tune the threshold?
    The default threshold of 0.5 assumes equal class priors and equal error costs.
    In practice the model's predicted probabilities are often shifted, so a different
    cutoff gives better balance between Up and Down predictions.

    Parameters
    ----------
    y_true : array of true binary labels (0=Down, 1=Up)
    probas : array of P(Up=1) predicted probabilities
    metric : "f1_macro" — macro-average F1 (default; balances Up and Down equally)
             "f1_down"  — F1 for the Down class only (use if Down detection is priority)

    Usage
    -----
    threshold = find_optimal_threshold(y_val, model.predict_proba(X_val)[:, 1])
    preds     = model.predict_with_threshold(X_test, threshold)

    IMPORTANT: always find the threshold on val, apply to test — never tune on test.
    """
    best_t, best_score = 0.5, -np.inf
    for t in np.arange(0.30, 0.71, 0.02):
        preds = (probas >= t).astype(int)
        if metric == "f1_macro":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        else:
            score = f1_score(y_true, preds, pos_label=0, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


class LogisticClassifier(BaseModel):
    """
    Logistic regression classification baseline.

    Why logistic regression instead of a dummy classifier? It gives calibrated
    probabilities and a meaningful decision boundary — we can observe which
    features drive linear separation between Up and Down days.

    C = 1/alpha (sklearn convention). Requires pre-scaled features.
    """

    model_name = "logistic"

    def __init__(self, C: float = 1.0 / RIDGE_ALPHA):
        self.C = C
        self._model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LogisticClassifier":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Returns [P(Down), P(Up)] for each sample."""
        return self._model.predict_proba(X)

    def predict_with_threshold(self, X, threshold: float) -> np.ndarray:
        """Classify using a custom P(Up) threshold instead of 0.5."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def coef_(self):
        return self._model.coef_


class LGBMClassifier(BaseModel):
    """
    LightGBM binary classifier. Main model.

    Uses binary cross-entropy (log-loss) as the objective, which produces
    well-calibrated probability estimates — important for the confidence bar
    in the Streamlit app.

    Early stopping monitors validation AUC (more stable than log-loss for
    imbalanced evaluation during training).

    Same regularization philosophy as LGBMRegressor — see regression.py for notes.
    """

    model_name = "lgbm"

    def __init__(self, params: dict | None = None):
        self.params = {
            **LGBM_PARAMS,
            "objective": "binary",
            "metric":    "auc",
        }
        if params:
            self.params.update(params)
        self._model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LGBMClassifier":
        callbacks = [
            lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
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

    def predict_proba(self, X) -> np.ndarray:
        """Returns [P(Down), P(Up)] for each sample."""
        return self._model.predict_proba(X)

    def predict_with_threshold(self, X, threshold: float) -> np.ndarray:
        """Classify using a custom P(Up) threshold instead of 0.5."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def best_iteration_(self):
        return getattr(self._model, "best_iteration_", None)
