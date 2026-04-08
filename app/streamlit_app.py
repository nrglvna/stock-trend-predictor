"""
Streamlit app: Stock Trend Predictor

Layout
------
Sidebar  : quick-select tickers + text input + Run button
Main area: header → price+volume chart (with forecast overlays) →
           indicator snapshot → 3 tabs (1-Day | 3-Day | 5-Day)

Each tab shows
--------------
- Confidence-based direction signal (Strong Up / Weak Up / Neutral / Down)
- Predicted price with % change vs. current
- Confidence probability bar
- "Why this prediction" — top-4 feature importances with plain-English notes

Run from the project root:
    streamlit run app/streamlit_app.py

Prerequisites: train models by running notebooks/stock_prediction.ipynb first.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Make src/ importable when running from project root or app/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import HORIZONS, MODELS_DIR
from src.data.loader import get_ticker_info, load_ohlcv
from src.data.preprocessor import validate_ohlcv
from src.features.pipeline import build_features, get_feature_columns
from src.models.base import BaseModel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
)

TREE_COLS = get_feature_columns(linear=False)

POPULAR_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY", "AMZN"]

# Human-readable labels for each feature
FEATURE_LABELS = {
    "ret_1d_lag":    "1-day momentum",
    "ret_5d_lag":    "5-day (weekly) momentum",
    "ret_21d_lag":   "21-day (monthly) momentum",
    "rsi_14":        "RSI — overbought/oversold level",
    "macd_hist":     "MACD — momentum acceleration",
    "bb_pct_b":      "Bollinger %B — mean reversion",
    "rvol_21d":      "Realized volatility (21d)",
    "vol_ratio_20d": "Volume vs. 20-day average",
    "atr_ratio":     "ATR — intraday price range",
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> dict:
    """Load all saved LightGBM models. Returns None for missing horizons."""
    models = {}
    for h in HORIZONS:
        for task in ("regression", "classification"):
            key = f"{task}_{h}"
            try:
                models[key] = BaseModel.load(task, "lgbm", h)
            except FileNotFoundError:
                models[key] = None
    return models


@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def fetch_and_prepare(ticker: str) -> tuple[dict, object]:
    """Download, validate, and feature-engineer data for *ticker*."""
    info = get_ticker_info(ticker)
    df   = load_ohlcv(ticker)
    df   = validate_ohlcv(df)
    df   = build_features(df)
    return info, df


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_horizon(models: dict, X: np.ndarray, close: float, h: int) -> dict | None:
    """Run regression + classification for one horizon. Returns None if models missing."""
    reg = models.get(f"regression_{h}")
    cls = models.get(f"classification_{h}")

    if reg is None or cls is None:
        return None

    log_ret    = float(reg.predict(X)[0])
    pred_price = close * np.exp(log_ret)
    direction  = int(cls.predict(X)[0])
    confidence = float(cls.predict_proba(X)[0, 1])   # P(Up)

    importances = getattr(cls, "feature_importances_", None)

    return {
        "log_ret":     log_ret,
        "pred_price":  pred_price,
        "direction":   direction,
        "confidence":  confidence,
        "importances": importances,
    }


def direction_signal(confidence: float) -> dict:
    """
    Map P(Up) to a 4-level signal with label, color, and icon.

    Strong Up  > 70%  — green
    Weak Up   55–70%  — light green
    Neutral   45–55%  — gray
    Down       < 45%  — red
    """
    if confidence > 0.70:
        return {"label": "Strong Up", "icon": "↑↑", "color": "#1a9641", "bg": "#eafaf1"}
    if confidence > 0.55:
        return {"label": "Weak Up",   "icon": "↑",  "color": "#2ecc71", "bg": "#f0faf4"}
    if confidence > 0.45:
        return {"label": "Neutral",   "icon": "→",  "color": "#7f8c8d", "bg": "#f5f5f5"}
    return     {"label": "Down",      "icon": "↓",  "color": "#e74c3c", "bg": "#fdecea"}


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def price_chart(df, ticker: str, days: int = 90, results: dict | None = None) -> go.Figure:
    """
    Two-panel chart: price (+ MAs + forecast overlays) above, volume below.

    Prediction points are shown as diamonds at approximate future dates,
    connected to the last real close by a dashed line of matching color.
    """
    plot_df    = df.tail(days).copy()
    close      = plot_df["Close"]
    volume     = plot_df["Volume"]
    last_date  = plot_df.index[-1]
    last_close = float(close.iloc[-1])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # ── Price line ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=close,
        name="Close", line=dict(color="#2c7bb6", width=1.8),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=close.rolling(20).mean(),
        name="20d MA", line=dict(color="#e67e22", width=1.2, dash="dot"),
        hovertemplate="%{y:.2f}<extra>20d MA</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=close.rolling(50).mean(),
        name="50d MA", line=dict(color="#27ae60", width=1.2, dash="dot"),
        hovertemplate="%{y:.2f}<extra>50d MA</extra>",
    ), row=1, col=1)

    # ── Forecast overlays ───────────────────────────────────────────────
    if results:
        FORECAST_COLORS = {1: "#e74c3c", 3: "#8e44ad", 5: "#e67e22"}
        for h, result in results.items():
            if result is None:
                continue
            future_date = last_date + pd.tseries.offsets.BDay(h)
            pred_price  = result["pred_price"]
            color       = FORECAST_COLORS.get(h, "#e74c3c")

            # Dashed connector: last close → predicted price
            fig.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[last_close, pred_price],
                name=f"{h}d Forecast",
                mode="lines+markers",
                line=dict(color=color, width=1.4, dash="dash"),
                marker=dict(size=[0, 10], symbol="diamond", color=color),
                hovertemplate=(
                    f"<b>%{{x|%b %d, %Y}}</b><br>{h}d Forecast: $%{{y:.2f}}"
                    f"<extra></extra>"
                ),
            ), row=1, col=1)

    # ── Volume bars (green = up day, red = down day) ─────────────────────
    vol_colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=volume,
        name="Volume", marker_color=vol_colors, opacity=0.55,
        hovertemplate="<b>%{x|%b %d}</b><br>Volume: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Last {days} Trading Days", font=dict(size=14)),
        legend=dict(orientation="h", y=1.04, x=0),
        height=460,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0", title="Price (USD)"),
        xaxis2=dict(gridcolor="#f0f0f0"),
        yaxis2=dict(gridcolor="#f0f0f0", title="Volume", tickformat=".2s"),
    )
    return fig


def direction_badge(confidence: float) -> str:
    """HTML badge reflecting the 4-level direction signal."""
    sig = direction_signal(confidence)
    return (
        f'<div style="display:inline-block;background:{sig["color"]};color:white;'
        f'padding:10px 28px;border-radius:8px;font-size:1.35rem;'
        f'font-weight:bold;letter-spacing:1px;">'
        f'{sig["icon"]} &nbsp;{sig["label"]}</div>'
    )


def confidence_bar(confidence: float) -> str:
    """Two-colour Up/Down probability bar with signal-aware coloring."""
    sig  = direction_signal(confidence)
    up   = int(round(confidence * 100))
    down = 100 - up
    return (
        f'<div style="background:#e0e0e0;border-radius:6px;height:20px;'
        f'overflow:hidden;margin-bottom:4px;">'
        f'<div style="background:{sig["color"]};width:{up}%;height:100%;float:left;"></div>'
        f'<div style="background:#e74c3c;width:{down}%;height:100%;float:left;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;">'
        f'<span style="color:{sig["color"]};font-weight:600;">'
        f'{sig["icon"]} {sig["label"]} &nbsp;{up}%</span>'
        f'<span style="color:#e74c3c;font-weight:600;">{down}% Down ↓</span>'
        f'</div>'
    )


def _feature_note(feat: str, val: float) -> str:
    """One-line plain-English reading of a single feature value."""
    if np.isnan(val):
        return "—"
    if feat == "rsi_14":
        zone = "overbought" if val > 70 else ("oversold" if val < 30 else "neutral")
        return f"RSI {val:.1f} — {zone}"
    if feat == "bb_pct_b":
        zone = "near upper band" if val > 0.8 else ("near lower band" if val < 0.2 else "mid-range")
        return f"%B {val:.2f} — {zone}"
    if feat in ("ret_1d_lag", "ret_5d_lag", "ret_21d_lag"):
        window = feat.split("_")[1]  # "1d", "5d", "21d"
        direction = "up" if val > 0 else "down"
        return f"{val * 100:+.2f}% over last {window} ({direction})"
    if feat == "macd_hist":
        state = "accelerating" if val > 0 else "decelerating"
        return f"MACD hist {val:.4f} — momentum {state}"
    if feat == "rvol_21d":
        return f"Annualized vol {val * 100:.1f}%"
    if feat == "vol_ratio_20d":
        level = "high" if val > 1.5 else ("low" if val < 0.5 else "normal")
        return f"Volume {val:.2f}× 20d avg — {level} activity"
    if feat == "atr_ratio":
        return f"ATR/Price {val * 100:.2f}% — intraday range"
    return f"{val:.4f}"


def render_why_prediction(result: dict, df) -> None:
    """
    Show top-4 features by LightGBM importance with current values and notes.
    Falls back gracefully if importances are unavailable.
    """
    importances = result.get("importances")
    if importances is None or len(importances) == 0:
        st.caption("Feature importance not available for this model.")
        return

    row      = df.iloc[-1]
    max_imp  = max(importances) or 1
    top4     = sorted(zip(TREE_COLS, importances), key=lambda x: x[1], reverse=True)[:4]

    for feat, imp in top4:
        label = FEATURE_LABELS.get(feat, feat)
        val   = float(row.get(feat, float("nan")))
        note  = _feature_note(feat, val)
        pct   = int(round(imp / max_imp * 100))

        col_a, col_b = st.columns([3, 1])
        col_a.markdown(
            f"<div style='font-size:0.88rem;margin-bottom:2px;'>"
            f"<b>{label}</b><br>"
            f"<span style='color:#666;font-size:0.82rem;'>{note}</span></div>",
            unsafe_allow_html=True,
        )
        col_b.markdown(
            f"<div style='font-size:0.80rem;color:#999;text-align:right;padding-top:4px;'>"
            f"weight: {pct}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#eee;border-radius:4px;height:5px;margin-bottom:10px;">'
            f'<div style="background:#2c7bb6;width:{pct}%;height:100%;border-radius:4px;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_tab(h: int, result: dict | None, current_close: float, df) -> None:
    """Render all prediction content for one horizon tab."""

    if result is None:
        st.warning(
            f"No trained model found for the **{h}-day** horizon.  \n"
            "Run `notebooks/stock_prediction.ipynb` from top to bottom, then refresh."
        )
        return

    pred_price = result["pred_price"]
    log_ret    = result["log_ret"]
    confidence = result["confidence"]
    pct_change = (pred_price / current_close - 1) * 100

    # Direction badge (confidence-driven, 4-level)
    st.markdown(direction_badge(confidence), unsafe_allow_html=True)
    st.markdown("")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label=f"Predicted Price ({h}d)",
        value=f"${pred_price:.2f}",
        delta=f"{pct_change:+.2f}%",
    )
    col2.metric(label="Current Price", value=f"${current_close:.2f}")
    col3.metric(
        label="Predicted Log Return",
        value=f"{log_ret * 100:+.3f}%",
        help="Log return predicted by the LightGBM regressor",
    )

    st.markdown("---")

    # Confidence bar
    st.markdown("**Direction Confidence**")
    st.markdown(confidence_bar(confidence), unsafe_allow_html=True)

    st.markdown("")

    # Feature-level explanation
    with st.expander("Why this prediction?", expanded=True):
        render_why_prediction(result, df)


def render_indicators(df) -> None:
    """Show last-row snapshot of key technical indicators."""
    row = df.iloc[-1]
    st.markdown("#### Current Indicators")
    c1, c2, c3 = st.columns(3)

    rsi_val = row.get("rsi_14", float("nan"))
    rsi_lbl = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    c1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_lbl)

    bb_val = row.get("bb_pct_b", float("nan"))
    bb_lbl = "Near upper band" if bb_val > 0.8 else ("Near lower band" if bb_val < 0.2 else "Mid-range")
    c2.metric("BB %B", f"{bb_val:.2f}", bb_lbl)

    vr_val = row.get("vol_ratio_20d", float("nan"))
    vr_lbl = "High activity" if vr_val > 1.5 else ("Low activity" if vr_val < 0.5 else "Normal")
    c3.metric("Volume Ratio (20d)", f"{vr_val:.2f}x", vr_lbl)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Session state: persist ticker across quick-select clicks ──────────
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL"

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 Stock Trend Predictor")
        st.markdown(
            "Enter a ticker symbol and click **Run Prediction** to forecast "
            "the stock's price and direction over 1-, 3-, and 5-day horizons."
        )
        st.markdown("---")

        # Quick-select grid (2 columns × 3 rows)
        st.markdown("**Quick-select**")
        qs_cols = st.columns(2)
        for i, t in enumerate(POPULAR_TICKERS):
            if qs_cols[i % 2].button(t, key=f"qs_{t}", use_container_width=True):
                st.session_state["ticker"] = t

        st.markdown("")

        ticker_input = st.text_input(
            "Or type a ticker",
            value=st.session_state["ticker"],
            max_chars=10,
            help="e.g. AAPL, MSFT, TSLA, SPY",
        ).strip().upper()

        # Basic validation: letters, digits, dot only (covers BRK.B, etc.)
        if ticker_input and not all(c.isalnum() or c == "." for c in ticker_input):
            st.warning("Ticker should only contain letters, digits, or '.'")
            ticker_input = ""
        elif ticker_input:
            st.session_state["ticker"] = ticker_input

        run_button = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

        st.markdown("---")
        st.caption(
            "**Models:** LightGBM (gradient-boosted trees)  \n"
            "**Features:** 9 technical indicators  \n"
            "**Horizons:** 1-day, 3-day, 5-day  \n\n"
            "_Run `notebooks/stock_prediction.ipynb` to train models._"
        )

    # ── Guard: require explicit run ───────────────────────────────────────
    if not run_button:
        st.markdown(
            "<div style='text-align:center;margin-top:80px;color:#888;font-size:1.1rem;'>"
            "👈 &nbsp;Select a ticker in the sidebar and click <b>Run Prediction</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if not ticker_input:
        st.error("Please enter a valid ticker symbol.")
        return

    # ── Load models ───────────────────────────────────────────────────────
    models = load_all_models()
    if all(models.get(f"regression_{h}") is None for h in HORIZONS):
        st.error(
            "No trained models found in `models/`.  \n"
            "Run `notebooks/stock_prediction.ipynb` end-to-end first."
        )
        return

    # ── Fetch & prepare data ──────────────────────────────────────────────
    try:
        with st.spinner(f"Loading data for **{ticker_input}**…"):
            info, df = fetch_and_prepare(ticker_input)
    except Exception as exc:
        st.error(f"Could not load data for **{ticker_input}**: {exc}")
        return

    current_close = float(df["Close"].iloc[-1])
    last_date     = df.index[-1].date()

    # ── Run all predictions (needed before rendering the chart) ───────────
    X_current = df[TREE_COLS].iloc[[-1]].values
    results   = {h: predict_horizon(models, X_current, current_close, h) for h in HORIZONS}

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"## {info['name']}  &nbsp; `{ticker_input}`")
    st.markdown(
        f"**Sector:** {info['sector']} &nbsp;|&nbsp; "
        f"**Currency:** {info['currency']} &nbsp;|&nbsp; "
        f"**Last close:** ${current_close:.2f} on {last_date}"
    )

    # ── Price chart with forecast overlays ────────────────────────────────
    st.plotly_chart(
        price_chart(df, ticker_input, days=90, results=results),
        use_container_width=True,
    )

    # ── Indicator snapshot ────────────────────────────────────────────────
    render_indicators(df)
    st.markdown("---")

    # ── Prediction tabs ───────────────────────────────────────────────────
    st.markdown("### Predictions")
    tabs = st.tabs(["1-Day", "3-Day", "5-Day"])
    for tab, h in zip(tabs, HORIZONS):
        with tab:
            render_tab(h, results[h], current_close, df)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Predictions are generated by ML models trained on historical data. "
        "This tool is for educational purposes only and does not constitute financial advice."
    )


if __name__ == "__main__":
    main()
