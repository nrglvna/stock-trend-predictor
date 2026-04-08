"""
Technical feature computation.

All functions accept a raw OHLCV DataFrame and return a Series (or DataFrame
for multi-output indicators). They are pure — no side effects, no global state.

Indicators are implemented directly with pandas/numpy so the logic is
transparent and there are no hidden library version surprises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BB_STD,
    BB_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_WINDOW,
    SMA_LONG,
    SMA_SHORT,
    VOL_LONG_WINDOW,
    VOL_SHORT_WINDOW,
    VOLUME_AVG_WINDOW,
)


# ---------------------------------------------------------------------------
# Returns & Momentum
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling log return over *window* days.
    ret_t = log(Close_t / Close_{t-window})

    Why: Captures momentum at the given speed. Log returns are additive over
    time and approximately normally distributed — cleaner than price differences.
    """
    return np.log(close / close.shift(window)).rename(f"ret_{window}d")


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EWM with alpha=1/window).

    RSI < 30 → oversold (potential bounce); RSI > 70 → overbought (potential pullback).
    Tree models exploit the nonlinear relationship between RSI levels and future returns
    better than linear models can.
    """
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    # Wilder's exponential smoothing (equivalent to EWM with com=window-1)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{window}")


def macd_histogram(close: pd.Series,
                   fast: int   = MACD_FAST,
                   slow: int   = MACD_SLOW,
                   signal: int = MACD_SIGNAL) -> pd.Series:
    """
    MACD histogram = MACD line − Signal line.

    Why histogram, not the MACD line itself? The histogram captures the
    *rate of change* of momentum — whether momentum is accelerating or
    decelerating. A shrinking histogram before a zero-cross is an early
    reversal signal.
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line).rename("macd_hist")


def sma_crossover_ratio(close: pd.Series,
                        short: int = SMA_SHORT,
                        long:  int = SMA_LONG) -> pd.Series:
    """
    Ratio of short-term SMA to long-term SMA.

    ratio > 1 → short-term average above long-term → uptrend.
    Using a ratio (not difference) makes the signal scale-invariant across
    different price levels and tickers.
    """
    sma_s = close.rolling(short).mean()
    sma_l = close.rolling(long).mean()
    return (sma_s / sma_l).rename(f"sma_{short}_{long}_ratio")


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """
    Annualized realized volatility = rolling std of daily log returns × √252.

    Why annualized? Comparable across window lengths and to implied vol benchmarks.
    Short vs long window comparison reveals whether recent volatility is elevated
    relative to the norm — a regime signal.
    """
    log_ret = np.log(close / close.shift(1))
    return (log_ret.rolling(window).std() * np.sqrt(252)).rename(f"rvol_{window}d")


def bollinger_pct_b(close: pd.Series,
                    window: int = BB_WINDOW,
                    n_std:  float = BB_STD) -> pd.Series:
    """
    Bollinger Band %B = (Close − Lower Band) / (Upper Band − Lower Band).

    %B = 0 → price at lower band (oversold zone).
    %B = 1 → price at upper band (overbought zone).
    %B outside [0, 1] → price outside bands (strong move or breakout).

    This is a bounded mean-reversion signal — more informative than raw
    Bollinger Band values which are price-scale dependent.
    """
    sma   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (close - lower) / (upper - lower)
    return pct_b.rename("bb_pct_b")


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def volume_ratio(volume: pd.Series, window: int = VOLUME_AVG_WINDOW) -> pd.Series:
    """
    Volume ratio = today's volume / rolling mean volume.

    A ratio > 2 signals unusual activity, which often accompanies significant
    price moves. Using a ratio (not raw volume) normalises across stocks with
    different float sizes.
    """
    avg_vol = volume.rolling(window).mean()
    return (volume / avg_vol).rename(f"vol_ratio_{window}d")


# ---------------------------------------------------------------------------
# Range / True Range
# ---------------------------------------------------------------------------

def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series,
              window: int = 14) -> pd.Series:
    """
    Normalized Average True Range = ATR(window) / Close.

    True Range = max(High−Low, |High−prev_Close|, |Low−prev_Close|) captures
    intraday range AND gap risk — information not present in close-to-close
    realized volatility.  Dividing by Close makes it scale-free (comparable
    across price levels and time).

    ATR is smoothed with Wilder's EWM (span=window) to reduce day-to-day noise.
    High ATR ratio → elevated uncertainty, often precedes reversals or
    continuation moves after consolidation.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return (atr / close).rename("atr_ratio")


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def day_of_week(index: pd.DatetimeIndex) -> pd.Series:
    """
    Day of week as integer: 0 = Monday, 4 = Friday.

    Monday and Friday have well-documented return asymmetries (weekend effect,
    window-dressing). Tree models learn these thresholds naturally from ordinal
    encoding; linear models will receive one-hot encoded versions in the pipeline.
    """
    return pd.Series(index.dayofweek, index=index, name="day_of_week")
