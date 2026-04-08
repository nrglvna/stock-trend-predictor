"""
Visualization functions for the notebook.

Each function is self-contained: takes data, produces and shows a figure.
All plots use a consistent style (seaborn whitegrid) and return the
matplotlib Figure so the notebook can save or display it.

Functions
---------
plot_price_history          : OHLCV close with train/val/test boundaries
plot_outlier_boxplots        : Boxplots for outlier analysis
plot_correlation_heatmap     : Feature–target correlation matrix
plot_regression_metrics      : Bar charts of RMSE and MAE (models × horizons)
plot_train_val_gap           : Train vs val metric gap (overfitting analysis)
plot_confusion_matrix        : Heatmap confusion matrix for one horizon/model
plot_classification_metrics  : Bar chart of 5 metrics across horizons
plot_roc_curves              : ROC curves for LightGBM classifier across horizons
plot_feature_importance      : LightGBM split-based feature importance
plot_shap_summary            : SHAP beeswarm summary plot
plot_horizon_comparison      : Line/scatter of test metric across horizons
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

# Suppress SHAP UserWarnings that appear in some notebook environments
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

STYLE      = "whitegrid"
PALETTE    = ["#4C72B0", "#DD8452"]   # blue = baseline, orange = LightGBM
HORIZONS   = ["1d", "3d", "5d"]
FIG_WIDTH  = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style():
    sns.set_style(STYLE)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ---------------------------------------------------------------------------
# Data Overview
# ---------------------------------------------------------------------------

def plot_price_history(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end:   pd.Timestamp,
    ticker:    str = "Stock",
) -> plt.Figure:
    """
    Plot Close price with vertical lines marking the train/val/test boundaries.
    Provides an immediate visual check that the time split looks right.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))
    ax.plot(df.index, df["Close"], linewidth=1, color="#2c7bb6", label="Close")
    ax.axvline(train_end, color="green",  linestyle="--", linewidth=1.2, label="Train end")
    ax.axvline(val_end,   color="orange", linestyle="--", linewidth=1.2, label="Val end")
    ax.set_title(f"{ticker} — Price History with Split Boundaries")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_outlier_boxplots(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    """
    Boxplots for the given columns to visualise distribution and outliers.
    Helps decide whether winsorization bounds are sensible.
    """
    _apply_style()
    n = len(columns)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#AED6F1"))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xticks([])

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — Outlier Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature Analysis
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols:  list[str],
) -> plt.Figure:
    """
    Pearson correlation heatmap between features and targets across horizons.
    Helps identify which features carry signal at which prediction horizon.
    """
    _apply_style()
    corr = df[feature_cols + target_cols].corr().loc[feature_cols, target_cols]
    fig, ax = plt.subplots(figsize=(len(target_cols) * 2 + 3, len(feature_cols) * 0.6 + 2))
    sns.heatmap(
        corr, annot=True, fmt=".2f", center=0,
        cmap="RdBu_r", linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature–Target Correlation (Pearson)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    horizon: int,
    top_n: int = 12,
) -> plt.Figure:
    """
    Horizontal bar chart of LightGBM split-based feature importance.
    Top-N features shown, sorted descending.
    """
    _apply_style()
    idx   = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(names, vals, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance (split count)")
    ax.set_title(f"LightGBM Feature Importance — {horizon}d horizon")
    fig.tight_layout()
    return fig


def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list[str],
                      horizon: int) -> None:
    """
    SHAP beeswarm summary plot.

    Each dot is one sample. The x-axis shows the SHAP value (impact on log-return
    prediction); colour shows the feature value (red=high, blue=low).
    This reveals both the direction and magnitude of each feature's influence.

    Note: shap.summary_plot renders directly via matplotlib — no Figure is returned.
    """
    import shap
    _apply_style()
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {horizon}d horizon", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Regression Evaluation
# ---------------------------------------------------------------------------

def plot_regression_metrics(
    results_df: pd.DataFrame,
    split: str = "test",
) -> plt.Figure:
    """
    Side-by-side bar charts of RMSE and MAE for each model across horizons.
    Only shows the requested split (default: test).
    """
    _apply_style()
    data = results_df[results_df["split"] == split].copy()
    models = data["model"].unique()
    x     = np.arange(len(HORIZONS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 4))
    for metric, ax in zip(["rmse", "mae"], axes):
        for i, (model, color) in enumerate(zip(models, PALETTE)):
            vals = [
                data[(data["model"] == model) & (data["horizon"] == h)][metric].values
                for h in HORIZONS
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=model, color=color, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} — {split} set")
        ax.legend()

    fig.suptitle("Regression Metrics by Model and Horizon", fontsize=13)
    fig.tight_layout()
    return fig


def plot_train_val_gap_regression(results_df: pd.DataFrame) -> plt.Figure:
    """
    For each model × horizon, show train and val RMSE side by side.
    A large gap signals overfitting.
    """
    _apply_style()
    data    = results_df[results_df["split"].isin(["train", "val"])].copy()
    models  = data["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(FIG_WIDTH, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = data[data["model"] == model]
        x   = np.arange(len(HORIZONS))
        width = 0.35
        for i, (split, color) in enumerate(zip(["train", "val"], ["#5dade2", "#e67e22"])):
            vals = [
                sub[(sub["horizon"] == h) & (sub["split"] == split)]["rmse"].values
                for h in HORIZONS
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * width, vals, width, label=split,
                   color=color, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        ax.set_title(f"{model} — Train vs Val RMSE")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("RMSE")
        ax.legend()

    fig.suptitle("Overfitting Analysis — Regression (Train vs Validation RMSE)", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classification Evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    model_name: str,
) -> plt.Figure:
    """
    Confusion matrix heatmap.

    Rows = actual class, columns = predicted class.
    Numbers show raw counts; colour intensity shows proportion.
    Useful to see whether a model is biased toward predicting one direction.
    """
    _apply_style()
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Down (0)", "Up (1)"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{model_name} — {horizon}d", fontsize=11)
    fig.tight_layout()
    return fig


def plot_classification_metrics(
    results_df: pd.DataFrame,
    split: str = "test",
) -> plt.Figure:
    """
    Grouped bar chart of all 5 classification metrics across horizons for each model.
    One subplot per metric, models side by side.
    """
    _apply_style()
    data    = results_df[results_df["split"] == split].copy()
    models  = data["model"].unique()
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x       = np.arange(len(HORIZONS))
    width   = 0.35

    fig, axes = plt.subplots(1, len(metrics), figsize=(FIG_WIDTH * 1.5, 4))
    for ax, metric in zip(axes, metrics):
        for i, (model, color) in enumerate(zip(models, PALETTE)):
            vals = [
                data[(data["model"] == model) & (data["horizon"] == h)][metric].values
                for h in HORIZONS
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * width, vals, width, label=model,
                   color=color, edgecolor="white")

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Horizon")
        if ax == axes[0]:
            ax.set_ylabel("Score")
        ax.legend(fontsize=7)

    fig.suptitle("Classification Metrics by Model and Horizon — Test Set", fontsize=13)
    fig.tight_layout()
    return fig


def plot_train_val_gap_classification(results_df: pd.DataFrame) -> plt.Figure:
    """
    Train vs val F1 score per model and horizon — overfitting check for classifiers.
    """
    _apply_style()
    data    = results_df[results_df["split"].isin(["train", "val"])].copy()
    models  = data["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(FIG_WIDTH, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub   = data[data["model"] == model]
        x     = np.arange(len(HORIZONS))
        width = 0.35
        for i, (split, color) in enumerate(zip(["train", "val"], ["#5dade2", "#e67e22"])):
            vals = [
                sub[(sub["horizon"] == h) & (sub["split"] == split)]["f1"].values
                for h in HORIZONS
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * width, vals, width, label=split,
                   color=color, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        ax.set_title(f"{model} — Train vs Val F1")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1)
        ax.legend()

    fig.suptitle("Overfitting Analysis — Classification (Train vs Validation F1)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    roc_data: list[dict],   # [{"fpr": ..., "tpr": ..., "auc": ..., "label": ...}]
) -> plt.Figure:
    """
    ROC curves for multiple models/horizons on one axes.

    roc_data entries should be pre-computed with:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.6, len(roc_data)))

    for entry, color in zip(roc_data, colors):
        ax.plot(entry["fpr"], entry["tpr"], color=color,
                label=f'{entry["label"]} (AUC={entry["auc"]:.3f})')

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — LightGBM Classifier")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Horizon Comparison
# ---------------------------------------------------------------------------

def plot_horizon_comparison(
    results_df: pd.DataFrame,
    metric:     str   = "directional_accuracy",
    split:      str   = "test",
    title:      str   = "",
) -> plt.Figure:
    """
    Line plot showing how a metric degrades (or improves) across prediction horizons.
    One line per model. Illustrates the signal decay from 1d → 5d.
    """
    _apply_style()
    data   = results_df[results_df["split"] == split].copy()
    models = data["model"].unique()

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, color in zip(models, PALETTE):
        sub  = data[data["model"] == model]
        vals = [sub[sub["horizon"] == h][metric].values for h in HORIZONS]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.plot(HORIZONS, vals, marker="o", label=model, color=color)

    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs Prediction Horizon")
    ax.legend()
    fig.tight_layout()
    return fig
