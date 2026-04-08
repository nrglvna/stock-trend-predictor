"""
Feature pipeline: assembles all features from raw OHLCV into a single DataFrame.

Two entry points:
- build_features(df)     → adds 9 feature columns to the OHLCV DataFrame
- get_feature_columns()  → returns the feature column names

Features (9 total):
  Momentum : ret_1d_lag, ret_5d_lag, ret_21d_lag  — past returns at 1/5/21 days
  Oscillators: rsi_14, macd_hist                   — momentum speed and direction
  Mean reversion: bb_pct_b                         — price position within Bollinger Bands
  Volatility: rvol_21d                             — 21-day realized volatility
  Volume: vol_ratio_20d                            — unusual volume activity
  Range: atr_ratio                                 — normalized true range (High/Low)

Design note: winsorization is computed on the full dataset before splitting. This is
acceptable because it only clips extreme values — it does not use future return info.
StandardScaler (in preprocessor.py) is the only transform fitted on train only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import VOL_LONG_WINDOW, VOL_SHORT_WINDOW, WINSOR_LOWER, WINSOR_UPPER
from src.features.technical import (
    atr_ratio,
    bollinger_pct_b,
    log_returns,
    macd_histogram,
    realized_volatility,
    rsi,
    volume_ratio,
)


# Columns to winsorize (cap extreme spikes before they distort training)
_WINSOR_COLS = [
    "ret_1d_lag", "ret_5d_lag", "ret_21d_lag",
    "macd_hist",
    "vol_ratio_20d",
    "rvol_21d",
    "atr_ratio",
]

# 9 features used by all models
_BASE_NUMERIC_FEATURES = [
    "ret_1d_lag",      # 1-day past return  — short momentum
    "ret_5d_lag",      # 5-day past return  — weekly momentum
    "ret_21d_lag",     # 21-day past return — monthly momentum
    "rsi_14",          # RSI(14)            — overbought / oversold
    "macd_hist",       # MACD histogram     — momentum acceleration
    "rvol_21d",        # Realized vol 21d   — volatility regime
    "bb_pct_b",        # Bollinger %B       — mean reversion signal
    "vol_ratio_20d",   # Volume ratio       — unusual trading activity
    "atr_ratio",       # ATR / Close        — intraday range (uses High/Low)
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 9 features and append them to *df* (OHLCV columns are preserved).

    Steps:
    1. Compute each indicator from raw OHLCV.
    2. Winsorize spiking columns (clip at 1st/99th percentile).
    3. Drop rows with NaN (rolling warm-up period, ~26 rows).

    Parameters
    ----------
    df : pd.DataFrame
        Validated OHLCV DataFrame from preprocessor.validate_ohlcv().

    Returns
    -------
    pd.DataFrame
        Original OHLCV + 9 feature columns. NaN rows from warm-up dropped.
    """
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # --- Momentum / Returns ---
    df["ret_1d_lag"]  = log_returns(close, 1)
    df["ret_5d_lag"]  = log_returns(close, 5)
    df["ret_21d_lag"] = log_returns(close, 21)

    # --- Momentum Oscillators ---
    df["rsi_14"]    = rsi(close)
    df["macd_hist"] = macd_histogram(close)

    # --- Volatility & Mean Reversion ---
    df["rvol_21d"] = realized_volatility(close, VOL_LONG_WINDOW)
    df["bb_pct_b"] = bollinger_pct_b(close)

    # --- Volume ---
    df["vol_ratio_20d"] = volume_ratio(volume)

    # --- Range (uses High/Low — genuinely new info vs. close-only features) ---
    df["atr_ratio"] = atr_ratio(df["High"], df["Low"], close)

    # --- Winsorize return-based features ---
    for col in _WINSOR_COLS:
        if col in df.columns:
            lo = df[col].quantile(WINSOR_LOWER)
            hi = df[col].quantile(WINSOR_UPPER)
            df[col] = df[col].clip(lo, hi)

    # Drop rows where any feature is NaN (rolling warm-up period, ~26 rows)
    df.dropna(subset=_BASE_NUMERIC_FEATURES, inplace=True)

    return df


def get_feature_columns(linear: bool = False) -> list[str]:
    """
    Return the list of feature column names to use for a given model type.

    The `linear` flag is kept for API compatibility but both paths now return
    the same list — day_of_week was removed (confirmed weak signal) and
    all remaining features are numeric, so no one-hot encoding is needed.
    """
    return _BASE_NUMERIC_FEATURES.copy()


def add_one_hot_dow(df: pd.DataFrame) -> pd.DataFrame:
    """
    No-op — kept for notebook compatibility.
    day_of_week was removed from the feature set (confirmed weak signal);
    get_feature_columns no longer requests any dow_* columns.
    """
    return df
