You are a senior machine learning engineer.

I am building an end-to-end stock price prediction system with a clean, production-like structure.

IMPORTANT:
If anything is unclear, ASK QUESTIONS before generating code.

PROJECT GOALS:
Build a system that:
1. Predicts stock price (regression)
2. Predicts direction (classification: up/down)
3. Supports multiple forecasting horizons:
   - 1 day ahead
   - 3 days ahead
   - 5 days ahead (used as weekend approximation)

ARCHITECTURE REQUIREMENTS:
The project must be split into TWO parts:

PART 1 — JUPYTER NOTEBOOK (.ipynb):
Used for analysis and experimentation.

Must include:
- Problem statement
- Data loading (yfinance)
- Data cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering
- Target creation:
  - Regression targets: t+1, t+3, t+5
  - Classification targets: direction for t+1, t+3, t+5
- Time-series split (NO shuffle)
- Model training:
  - Regression: Linear Regression, Random Forest, XGBoost
  - Classification: Logistic Regression, Random Forest, XGBoost
- Evaluation:
  - Regression: RMSE, MAE
  - Classification: Accuracy, F1-score, Confusion Matrix
- Model comparison across different horizons
- Interpretation:
  - Feature importance
  - SHAP values
- Clear markdown explanations for each step

PART 2 — PYTHON MODULES (.py):
Refactor the final logic into clean, reusable modules:

- data_loader.py → fetch data
- features.py → feature engineering
- train.py → train models
- predict.py → inference logic

PART 3 — STREAMLIT APP:
Build a simple UI where user can:
- Input stock ticker
- Select prediction horizon (1d / 3d / 5d)
- See:
  - price chart
  - predicted price
  - predicted direction

FEATURE ENGINEERING REQUIREMENTS:

You must implement strong feature engineering suitable for financial time series.

Include at minimum:
- Trend indicators: moving averages (MA), exponential moving averages (EMA)
- Momentum indicators: RSI, MACD
- Volatility indicators: rolling standard deviation, Bollinger Bands
- Lag features: previous values of price and volume
- Return-based features: daily returns and log returns
- Time-based features: day of week, month

IMPORTANT:
- After generating these features, analyze their usefulness.
- Check correlations with targets.
- Suggest additional features based on the dataset.

CRITICAL THINKING:
Do not just implement standard indicators.
Based on the data, propose additional meaningful features that could improve prediction.

Explain WHY each feature is useful for predicting stock price movement.

Evaluate feature importance and remove weak or redundant features.

TECHNICAL REQUIREMENTS:
- Use Python
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap, yfinance, streamlit
- Handle missing values properly
- Drop NaN after shifting targets
- Avoid data leakage
- Use time-based split (not random)
- Keep code modular and readable

MODEL STRATEGY:
- Train separate models for each horizon (1d, 3d, 5d)
- Compare performance and explain why accuracy changes with horizon

OUTPUT STRATEGY:
1. First, propose the full project structure
2. Then generate the notebook step-by-step
3. Wait for confirmation after each step before continuing
4. After notebook is complete, generate Python modules
5. Finally, build the Streamlit app

DO NOT:
- Generate everything at once
- Skip explanations
- Mix notebook and production code

Start by asking clarifying questions if needed, then propose the project structure.