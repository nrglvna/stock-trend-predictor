 Plan: Stock Trend Predictor — Final Plan                                                                                    
                                                        
 Context

 Building an end-to-end stock prediction system from an empty repo. Goal: clean, modular, academically strong ML project
 inspired by real-world applications. Predict both price (regression) and direction (classification) across 1d, 3d, 5d
 horizons.

 ---
 Project Structure

 stock-trend-predictor/
 ├── requirements.txt
 ├── .gitignore
 ├── notebooks/
 │   └── stock_prediction.ipynb
 ├── src/
 │   ├── config.py
 │   ├── data/
 │   │   ├── loader.py
 │   │   └── preprocessor.py
 │   ├── features/
 │   │   ├── technical.py
 │   │   └── pipeline.py
 │   ├── models/
 │   │   ├── base.py
 │   │   ├── regression.py
 │   │   └── classification.py
 │   └── evaluation/
 │       ├── metrics.py
 │       └── visualization.py
 └── app/
     └── streamlit_app.py

 ---
 Data Strategy

 - Ticker: Configurable (default AAPL); 5 years of daily OHLCV via yfinance
 - Targets: Forward log returns for h ∈ {1, 3, 5} days
   - Regression: log(close_{t+h} / close_t)
   - Classification: 1 if forward_log_return > 0 else 0
 - Split: Time-based — first 70% train, next 15% validation, last 15% test. No shuffling.

 ---
 Data Preprocessing

 Outlier Analysis

 - Boxplots for OHLCV and key features to visualize distribution and outliers
 - Handle outliers conservatively: winsorize at 1st/99th percentile for return-based features
 - Do NOT drop rows — financial extremes (crashes, squeezes) are real signals, not noise
 - Flag extreme volume spikes separately as a feature rather than removing them

 Encoding

 - Day of week: ordinal encoding (0–4 Mon–Fri) — already naturally ordered, tree models handle it well
 - For linear models: one-hot encode day of week to avoid spurious ordinality

 Feature Scaling

 - Linear models (Ridge, Logistic): StandardScaler on all numeric features — required for L2 regularization to work fairly
 - Tree models (LightGBM): No scaling — trees are invariant to monotonic transformations
 - Scaling fitted on train set only, applied to val/test

 ---
 Feature Engineering (Lean & Justified)

 Why log returns, not raw prices?
 Prices are non-stationary — models memorize levels, not patterns. Log returns are stationary and scale-independent.

 Features

 ┌──────────────────────────────┬────────────┬────────────────────────────────────────────────────────────────┐
 │           Feature            │  Category  │                          Why it helps                          │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ Log return (1d, 5d, 21d)     │ Momentum   │ Captures short/medium/long momentum                            │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ RSI(14)                      │ Momentum   │ Overbought/oversold signal; nonlinear — trees handle this well │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ MACD histogram               │ Momentum   │ Momentum acceleration/deceleration, not just direction         │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ BB %B                        │ Volatility │ Position within Bollinger Bands — mean reversion signal        │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ Realized vol (5d, 21d)       │ Volatility │ Recent vs longer-term volatility comparison                    │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ Volume ratio (vol / 20d avg) │ Volume     │ Unusual activity often precedes moves                          │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ SMA crossover (5d/21d ratio) │ Trend      │ Simple trend direction signal                                  │
 ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────┤
 │ Day of week (encoded)        │ Temporal   │ Day-of-week drift is well-documented                           │
 └──────────────────────────────┴────────────┴────────────────────────────────────────────────────────────────┘

 ~10 clean features. Each justified, no redundancy.

 Feature Analysis in Notebook

 - Correlation heatmap with each target horizon
 - LightGBM feature importance bar chart + written explanation of top features
 - SHAP summary plot — shows direction and magnitude of each feature's impact

 ---
 Model Strategy

 Baseline: Ridge regression / Logistic regression (L2)
 - Interpretable, fast, no overfitting
 - Sets a performance floor

 Main model: LightGBM
 - Best-in-class for tabular data at this scale
 - Captures nonlinear feature interactions
 - SHAP-compatible for interpretability
 - No neural nets: ~1500 training samples too small

 One model per horizon per task. Models saved to models/ directory via joblib.

 ---
 Evaluation

 Regression

 ┌──────────────────────┬───────────────────────────────────────┐
 │        Metric        │                Purpose                │
 ├──────────────────────┼───────────────────────────────────────┤
 │ RMSE                 │ Penalizes large errors                │
 ├──────────────────────┼───────────────────────────────────────┤
 │ MAE                  │ Robust magnitude error                │
 ├──────────────────────┼───────────────────────────────────────┤
 │ R²                   │ Explained variance (expect 0.01–0.05) │
 ├──────────────────────┼───────────────────────────────────────┤
 │ Directional accuracy │ Economically meaningful correctness   │
 └──────────────────────┴───────────────────────────────────────┘

 Visualization: Bar chart comparing RMSE and MAE across horizons (1d/3d/5d) for both models side-by-side.

 Classification

 ┌───────────┬──────────────────────────────────────────────────────┐
 │  Metric   │                       Purpose                        │
 ├───────────┼──────────────────────────────────────────────────────┤
 │ Accuracy  │ Overall correctness (balanced classes ~50% baseline) │
 ├───────────┼──────────────────────────────────────────────────────┤
 │ Precision │ Of predicted Up days, how many were actually Up      │
 ├───────────┼──────────────────────────────────────────────────────┤
 │ Recall    │ Of actual Up days, how many we caught                │
 ├───────────┼──────────────────────────────────────────────────────┤
 │ F1-score  │ Balances precision and recall                        │
 ├───────────┼──────────────────────────────────────────────────────┤
 │ ROC-AUC   │ Threshold-independent ranking ability (0.5 = random) │
 └───────────┴──────────────────────────────────────────────────────┘

 Visualization:
 - Confusion matrix heatmap per horizon per model
 - Bar chart of all 5 classification metrics per horizon, models side-by-side
 - ROC curve for LightGBM classifier

 Performance vs Horizon

 - Summary table: all metrics × horizons × models
 - Explanation of expected degradation: signal autocorrelation decays; features become stale at 5d

 ---
 Streamlit App

 Layout:
 - Sidebar: stock ticker input + "Run Prediction" button
 - Main area: 3 tabs — 1-Day | 3-Day | 5-Day

 Each tab displays:
 - Historical price chart (last 90 days) with 20d and 50d moving averages overlaid
 - Current indicator values (RSI, volume ratio, BB %B)
 - Predicted price (regression output converted back from log return)
 - Predicted direction: colored badge — green "↑ Up" or red "↓ Down"
 - Confidence: probability bar (from LightGBM classifier)

 ---
 Build Order (Step-by-Step, Confirm Each)

 1. requirements.txt, .gitignore, src/config.py
 2. src/data/loader.py + src/data/preprocessor.py (includes outlier handling + scaling)
 3. src/features/technical.py + src/features/pipeline.py
 4. src/models/base.py + regression.py + classification.py
 5. src/evaluation/metrics.py + visualization.py (confusion matrix, bar charts, SHAP)
 6. notebooks/stock_prediction.ipynb
 7. app/streamlit_app.py

 ---
 Verification

 - Notebook runs end-to-end with no errors
 - No data leakage (assert train max date < val min date)
 - Confusion matrices render for all horizons
 - RMSE/MAE bar charts compare both models across horizons
 - Streamlit app loads, tabs work, predictions render with color