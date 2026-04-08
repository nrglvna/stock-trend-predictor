"""
Abstract base class for all models in this project.

Every model (regression or classification, baseline or main) implements this
interface. This keeps the evaluation and Streamlit code model-agnostic — they
call .fit(), .predict(), .predict_proba() without caring which model is underneath.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import joblib
import numpy as np

from src.config import MODELS_DIR


class BaseModel(ABC):
    """Common interface for all regression and classification models."""

    # Subclasses set this to identify themselves in saved filenames and reports
    model_name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "BaseModel":
        """Train the model. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions (continuous for regression, class labels for classification)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities. Only meaningful for classifiers.
        Default raises — classifiers must override.
        """
        raise NotImplementedError(f"{self.model_name} does not support predict_proba.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, task: str, horizon: int) -> str:
        """
        Persist the fitted model to disk.

        Filename convention: {task}_{model_name}_{horizon}d.joblib
        e.g.  regression_lgbm_1d.joblib
              classification_ridge_5d.joblib

        Returns the full path where the model was saved.
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{task}_{self.model_name}_{horizon}d.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, task: str, model_name: str, horizon: int) -> "BaseModel":
        """Load a previously saved model from disk."""
        path = os.path.join(MODELS_DIR, f"{task}_{model_name}_{horizon}d.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved model found at '{path}'. "
                "Run training in the notebook first."
            )
        return joblib.load(path)
