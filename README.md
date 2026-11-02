# 📈 QuantPulse: Share Market Prediction & Technical Analysis Suite

This package includes **two complete, ready-to-run versions** of the Share Market Forecasting application so you can either run a full Python Machine Learning app with live Yahoo Finance data OR open an instant standalone web dashboard directly in your browser.

---

## Folder Contents

1. **`app.py`** — Full-featured **Python + Streamlit Machine Learning Web App**:
   - Pulls live historical market data via `yfinance` (supports NSE India stocks like `RELIANCE.NS`, `TCS.NS`, `^NSEI`, and global tickers like `AAPL`, `NVDA`, `^GSPC`).
   - Computes **SMA 20/50, EMA 20, Bollinger Bands, 14-Day RSI, and MACD**.
   - Trains a **Random Forest Regressor** or **Ridge Regression** model using chronological splits (preventing data leakage) and projects future prices with confidence intervals.
   - Includes an interactive Plotly candlestick chart, out-of-sample backtest visualizer, and automated signal checklist.

2. **`requirements.txt`** — All required Python packages (`streamlit`, `yfinance`, `pandas`, `numpy`, `scikit-learn`, `plotly`).

3. **`standalone_web_dashboard.html`** — **Zero-Install Browser Dashboard**:
   - Double-click this HTML file to open it directly in Chrome, Edge, or Safari—no Python installation required.
   - Includes built-in market simulation series, interactive Chart.js visualization, 20/50 SMA, 14-day RSI, regression forecast bands, and custom **CSV upload** support.

---

## How to Run the Python Machine Learning App (`app.py`)

### 1. Install Python Requirements
Open your terminal or command prompt inside this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run app.py
```
Your default browser will automatically open `http://localhost:8501` with the live dashboard.

---

## How to Run the Instant Web App (`standalone_web_dashboard.html`)

- Simply double-click **`standalone_web_dashboard.html`** to launch it immediately in any modern web browser.
- You can also upload any historical `.csv` file (with `Date` in the first column and `Close` in the last column) to run forecasts on your own custom dataset.
