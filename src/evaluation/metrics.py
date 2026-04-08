"""
Metric computation for regression and classification tasks.

All functions return plain dicts or DataFrames — no plotting here.
The key design goal is a single evaluate_all() call that produces a
structured results table covering train, val, and test splits so that
overfitting analysis is automatic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for one split.

    Returns
    -------
    dict with keys: rmse, mae, r2, directional_accuracy
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # R² — correlation-based to match definition exactly
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Directional accuracy: did the sign of the prediction match the sign of the return?
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

    return {
        "rmse":                round(rmse,    6),
        "mae":                 round(mae,     6),
        "r2":                  round(r2,      4),
        "directional_accuracy": round(dir_acc, 4),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray | None = None) -> dict:
    """
    Compute the five standard classification metrics.

    Parameters
    ----------
    y_true : array of 0/1 labels
    y_pred : array of 0/1 predictions
    y_prob : array of P(Up) probabilities — required for ROC-AUC.
             If None, ROC-AUC is reported as NaN.

    Metric explanations
    -------------------
    Accuracy  : Fraction of days correctly classified. Useful baseline when
                classes are roughly balanced (~50/50 Up/Down over 5 years).
    Precision : Of all days we predicted as "Up", what fraction actually went up.
                High precision → fewer false alarms (important if you act on every signal).
    Recall    : Of all days that actually went up, what fraction we caught.
                High recall → fewer missed opportunities.
    F1        : Harmonic mean of precision and recall. Balances the two; more
                informative than accuracy when you care about both types of error.
    ROC-AUC   : Measures ranking quality independent of the decision threshold.
                A random classifier scores 0.5; a perfect one scores 1.0.
                This is the most robust single metric for imbalanced or near-balanced
                binary problems.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(auc,  4),
    }


# ---------------------------------------------------------------------------
# Unified evaluator — produces the overfitting analysis table
# ---------------------------------------------------------------------------

def evaluate_regression(
    model,
    splits: dict,          # {"train": (X, y), "val": (X, y), "test": (X, y)}
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Evaluate a regression model on all three splits and return a tidy DataFrame.

    Having train / val / test metrics side-by-side immediately reveals overfitting:
    a large train-vs-val gap in RMSE means the model has memorised training patterns
    that don't generalise.

    Returns
    -------
    pd.DataFrame indexed by (model, horizon, split) with metric columns.
    """
    rows = []
    for split_name, (X, y) in splits.items():
        preds = model.predict(X)
        m = regression_metrics(y, preds)
        rows.append({
            "model":   model_name,
            "horizon": f"{horizon}d",
            "split":   split_name,
            **m,
        })
    return pd.DataFrame(rows)


def evaluate_classification(
    model,
    splits: dict,          # {"train": (X, y), "val": (X, y), "test": (X, y)}
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Evaluate a classification model on all three splits.

    Same train/val/test side-by-side structure for overfitting analysis.
    ROC-AUC is computed from predict_proba if available.
    """
    rows = []
    has_proba = hasattr(model, "predict_proba")

    for split_name, (X, y) in splits.items():
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if has_proba else None
        m = classification_metrics(y, preds, probs)
        rows.append({
            "model":   model_name,
            "horizon": f"{horizon}d",
            "split":   split_name,
            **m,
        })
    return pd.DataFrame(rows)


def summarise_results(results: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of per-model/horizon result DataFrames into one summary table.
    Sorts by horizon then split for easy reading.
    """
    df = pd.concat(results, ignore_index=True)
    split_order = {"train": 0, "val": 1, "test": 2}
    df["_split_order"] = df["split"].map(split_order)
    df.sort_values(["horizon", "model", "_split_order"], inplace=True)
    df.drop(columns=["_split_order"], inplace=True)
    return df.reset_index(drop=True)
