"""
Central configuration for the stock trend predictor.
All constants and hyperparameters live here — edit this file to change behaviour
without touching model or feature code.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "MSFT"
DATA_PERIOD_YEARS = 5          # years of history to fetch
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------------
# Prediction horizons (trading days)
# ---------------------------------------------------------------------------
HORIZONS = [1, 3, 5]           # 1-day, 3-day, 5-day forward returns

# ---------------------------------------------------------------------------
# Train / Validation / Test split ratios (time-ordered, no shuffling)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# TEST_RATIO is the remainder (0.20)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RSI_WINDOW         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
BB_WINDOW          = 20
BB_STD             = 2
VOL_SHORT_WINDOW   = 5         # realized volatility — short lookback
VOL_LONG_WINDOW    = 21        # realized volatility — long lookback
VOLUME_AVG_WINDOW  = 20        # for volume ratio
SMA_SHORT          = 5         # for SMA crossover ratio
SMA_LONG           = 21

# Winsorization bounds for return-based features
WINSOR_LOWER = 0.01            # 1st percentile
WINSOR_UPPER = 0.99            # 99th percentile

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RIDGE_ALPHA = 1.0

LGBM_PARAMS = {
    "n_estimators":    300,        # early stopping limits this in practice anyway
    "learning_rate":   0.05,
    "num_leaves":      15,         # was 31 — halved; with ~860 train rows, 31 leaves overfits badly
    "min_child_samples": 50,       # was 20 — 50/860 = 5.8% minimum per leaf; strong guard
    "subsample":       0.7,        # was 0.8 — more aggressive row subsampling
    "colsample_bytree": 0.7,       # was 0.8 — more aggressive feature subsampling
    "reg_alpha":       0.1,        # new — L1 regularization (encourages feature sparsity)
    "reg_lambda":      1.0,        # new — L2 regularization (weight shrinkage)
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}

LGBM_EARLY_STOPPING_ROUNDS = 50   # stop if val metric doesn't improve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
