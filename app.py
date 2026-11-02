import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

st.set_page_config(
    page_title="QuantPulse | Share Market Forecasting & Technical Suite",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 QuantPulse: Share Market Technical Analysis & Forecasting App")
st.caption("⚠️ Educational & Quantitative Analysis Tool — Financial markets carry inherent risk. Past performance and statistical projections do not guarantee future returns.")

# Sidebar Configuration
st.sidebar.header("⚙️ Data & Model Settings")

data_source = st.sidebar.radio(
    "Select Data Source",
    [
        "📦 Built-in Market Dataset (Instant / Offline)",
        "📂 Upload Custom CSV File",
        "🌐 Live Yahoo Finance API (Auto-Fallback)"
    ],
    index=0
)

popular_tickers = {
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "TCS (NSE)": "TCS.NS",
    "HDFC Bank (NSE)": "HDFCBANK.NS",
    "Infosys (NSE)": "INFY.NS",
    "State Bank of India (NSE)": "SBIN.NS",
    "Nifty 50 Index": "^NSEI",
    "Apple Inc. (NASDAQ)": "AAPL",
    "NVIDIA Corp. (NASDAQ)": "NVDA",
    "Microsoft Corp. (NASDAQ)": "MSFT",
    "Tesla Inc. (NASDAQ)": "TSLA",
    "S&P 500 Index": "^GSPC",
    "Custom Ticker": "CUSTOM"
}

uploaded_file = None
if data_source == "📂 Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV (Date, Open, High, Low, Close, Volume)", type=["csv"])
    ticker = uploaded_file.name.replace(".csv", "") if uploaded_file else "CUSTOM_CSV"
else:
    selected_label = st.sidebar.selectbox("Select Market / Stock", list(popular_tickers.keys()))
    if popular_tickers[selected_label] == "CUSTOM":
        ticker = st.sidebar.text_input("Enter Yahoo Finance Ticker (e.g., SBIN.NS, GOOGL)", value="SBIN.NS").upper().strip()
    else:
        ticker = popular_tickers[selected_label]

history_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y"], index=1)
forecast_days = st.sidebar.slider("Forecast Horizon (Trading Days)", min_value=5, max_value=30, value=10)
model_choice = st.sidebar.selectbox("Forecasting Model", ["Random Forest Regressor", "Ridge Linear Regression"])


def generate_fallback_series(symbol, days=520):
    """Generates a deterministic, realistic OHLCV dataset if no CSV or internet is available."""
    configs = {
        "RELIANCE.NS": (1150.0, 0.0005, 0.014, 6500000),
        "TCS.NS":      (3650.0, 0.0004, 0.012, 2800000),
        "HDFCBANK.NS": (1520.0, 0.00045, 0.011, 8200000),
        "INFY.NS":     (1480.0, 0.0005, 0.013, 5400000),
        "SBIN.NS":     (620.0,  0.0006, 0.015, 12000000),
        "^NSEI":       (21500.0, 0.0004, 0.009, 250000),
        "AAPL":        (185.0,  0.0006, 0.014, 48000000),
        "NVDA":        (95.0,   0.0012, 0.022, 62000000),
        "MSFT":        (390.0,  0.0005, 0.012, 22000000),
        "TSLA":        (210.0,  0.0007, 0.025, 75000000),
        "^GSPC":       (4950.0, 0.00045, 0.008, 3500000000)
    }
    start_p, drift, vol, v_base = configs.get(symbol, (500.0, 0.0005, 0.015, 5000000))
    seed_val = sum(ord(c) for c in symbol) + 2026
    rng = np.random.default_rng(seed_val)

    end_dt = datetime.today()
    bdates = pd.bdate_range(end=end_dt, periods=days)
    n = len(bdates)
    t = np.linspace(0, 8 * np.pi, n)
    cycle = 0.0015 * np.sin(t) + 0.001 * np.cos(2.3 * t)
    shocks = drift + cycle + rng.normal(0, vol, n)

    closes = [start_p]
    for s in shocks[1:]:
        closes.append(max(1.0, closes[-1] * (1.0 + s)))
    closes = np.round(closes, 2)
    opens = np.round(closes * (1.0 + rng.normal(0, vol * 0.35, n)), 2)
    highs = np.round(np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, vol * 0.5, n))), 2)
    lows = np.round(np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, vol * 0.5, n))), 2)
    volumes = (v_base * rng.uniform(0.65, 1.45, n)).astype(int)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=bdates)
    df.index.name = "Date"
    return df


def load_builtin_csv(symbol):
    """Loads pre-bundled market_data.csv or generates it on the fly if missing."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "market_data.csv")
    if os.path.exists(csv_path):
        full_df = pd.read_csv(csv_path)
        sub = full_df[full_df["Ticker"] == symbol].copy()
        if not sub.empty:
            sub["Date"] = pd.to_datetime(sub["Date"])
            sub = sub.sort_values("Date").set_index("Date")
            return sub[["Open", "High", "Low", "Close", "Volume"]]
    return generate_fallback_series(symbol)


def parse_uploaded_csv(file_obj):
    df = pd.read_csv(file_obj)
    col_map = {c.lower().strip(): c for c in df.columns}
    date_col = col_map.get("date") or col_map.get("timestamp") or df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df.index.name = "Date"

    close_col = col_map.get("close") or col_map.get("adj close") or df.columns[-1]
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["Close"])

    for col_name, factor in [("Open", 1.0), ("High", 1.004), ("Low", 0.996)]:
        c_match = col_map.get(col_name.lower())
        if c_match and c_match in df.columns:
            df[col_name] = pd.to_numeric(df[c_match], errors="coerce").fillna(df["Close"] * factor)
        else:
            df[col_name] = np.round(df["Close"] * factor, 2)

    vol_match = col_map.get("volume")
    if vol_match and vol_match in df.columns:
        df["Volume"] = pd.to_numeric(df[vol_match], errors="coerce").fillna(1000000)
    else:
        df["Volume"] = 1000000

    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=900)
def fetch_live_or_fallback(symbol, period):
    try:
        df = yf.download(symbol, period=period, progress=False, timeout=8)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) >= 60:
            return df, "Live Yahoo Finance API"
    except Exception:
        pass
    return load_builtin_csv(symbol), "Built-in Market Dataset (Offline Fallback)"


def compute_indicators(df):
    data = df.copy()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    rolling_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["SMA_20"] + (rolling_std * 2)
    data["BB_Lower"] = data["SMA_20"] - (rolling_std * 2)

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Volatility_10d"] = data["Return_1d"].rolling(window=10).std()

    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)

    return data


def train_and_forecast(df, days_ahead, model_name):
    data = compute_indicators(df).dropna().copy()
    feature_cols = [
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
        "SMA_20", "EMA_20", "RSI", "MACD", "Volatility_10d"
    ]

    X = data[feature_cols]
    y = data["Close"]
    split_idx = int(len(data) * 0.85)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))

    model.fit(X, y)

    future_dates = []
    future_preds = []
    working_df = df.copy()
    last_date = working_df.index[-1]

    for _ in range(1, days_ahead + 1):
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        last_date = next_date

        temp_ind = compute_indicators(working_df).iloc[-1:]
        X_next = temp_ind[feature_cols].ffill().bfill()
        next_close = float(model.predict(X_next)[0])

        future_dates.append(next_date)
        future_preds.append(next_close)

        new_row = pd.DataFrame({
            "Open": [next_close],
            "High": [next_close * 1.005],
            "Low": [next_close * 0.995],
            "Close": [next_close],
            "Volume": [working_df["Volume"].iloc[-1]]
        }, index=[next_date])
        working_df = pd.concat([working_df, new_row])

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": future_preds,
        "Upper_Bound": [p + 1.5 * rmse for p in future_preds],
        "Lower_Bound": [p - 1.5 * rmse for p in future_preds]
    }).set_index("Date")

    test_eval_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": preds_test
    }, index=y_test.index)

    return data, forecast_df, test_eval_df, mae, rmse


# Load Data Based on Selected Source
raw_df = pd.DataFrame()
active_source_label = ""

if data_source == "📦 Built-in Market Dataset (Instant / Offline)":
    raw_df = load_builtin_csv(ticker)
    active_source_label = f"Built-in Market Dataset ({len(raw_df)} trading days loaded)"
elif data_source == "📂 Upload Custom CSV File":
    if uploaded_file is not None:
        raw_df = parse_uploaded_csv(uploaded_file)
        active_source_label = f"Uploaded CSV: {uploaded_file.name} ({len(raw_df)} rows)"
    else:
        st.info("👆 Please upload a CSV file in the left sidebar, or switch to **Built-in Market Dataset** to explore preloaded stocks immediately.")
        st.stop()
else:
    with st.spinner(f"Fetching market data for {ticker}..."):
        raw_df, active_source_label = fetch_live_or_fallback(ticker, history_period)

if raw_df.empty or len(raw_df) < 60:
    st.error("Dataset needs at least 60 trading days of prices to calculate 50-day Moving Averages and train the model.")
else:
    st.success(f"✅ **Data Loaded Successfully:** `{ticker}` via **{active_source_label}** (Date Range: `{raw_df.index[0].strftime('%Y-%m-%d')}` to `{raw_df.index[-1].strftime('%Y-%m-%d')}`)")

    data, forecast_df, test_eval_df, mae, rmse = train_and_forecast(raw_df, forecast_days, model_choice)

    latest_close = float(data["Close"].iloc[-1])
    prev_close = float(data["Close"].iloc[-2])
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100

    proj_end_price = float(forecast_df["Predicted_Close"].iloc[-1])
    proj_pct = ((proj_end_price - latest_close) / latest_close) * 100
    latest_rsi = float(data["RSI"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"{latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    col2.metric(f"{forecast_days}-Day Projected Price", f"{proj_end_price:,.2f}", f"{proj_pct:+.2f}% Est.")

    rsi_status = "Neutral"
    if latest_rsi > 70:
        rsi_status = "Overbought (>70)"
    elif latest_rsi < 30:
        rsi_status = "Oversold (<30)"
    col3.metric("14-Day RSI", f"{latest_rsi:.1f}", rsi_status)
    col4.metric("Backtest RMSE (Error)", f"±{rmse:.2f}", f"MAE: {mae:.2f}")

    st.subheader(f"📊 {ticker} Price Action, Moving Averages & {forecast_days}-Day Forecast")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=("Candlestick, Moving Averages & ML Forecast", "Relative Strength Index (RSI 14)")
    )

    plot_df = data.tail(180)

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="OHLC Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["SMA_20"],
        mode="lines", name="SMA 20", line=dict(color="#38bdf8", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["SMA_50"],
        mode="lines", name="SMA 50", line=dict(color="#fbbf24", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["Predicted_Close"],
        mode="lines+markers", name="ML Forecast",
        line=dict(color="#a855f7", width=2.5, dash="dot")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["Upper_Bound"]) + list(forecast_df["Lower_Bound"][::-1]),
        fill="toself",
        fillcolor="rgba(168, 85, 247, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Interval"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["RSI"],
        mode="lines", name="RSI (14)", line=dict(color="#10b981", width=1.5)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3b82f6", row=2, col=1)

    fig.update_layout(
        height=680,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Day-by-Day Forecast Table",
        "🎯 Model Backtest Verification",
        "🧭 Technical Signal Summary",
        "🗄️ Raw Historical Data & Download"
    ])

    with tab1:
        st.markdown("### Projected Price Schedule")
        display_forecast = forecast_df.copy()
        display_forecast.index = display_forecast.index.strftime("%Y-%m-%d")
        st.dataframe(display_forecast.style.format("{:,.2f}"), use_container_width=True)

    with tab2:
        st.markdown("### Out-of-Sample Backtest (Actual vs. Predicted on Unseen Test Split)")
        bt_fig = go.Figure()
        bt_fig.add_trace(go.Scatter(x=test_eval_df.index, y=test_eval_df["Actual"], name="Actual Close", line=dict(color="#38bdf8")))
        bt_fig.add_trace(go.Scatter(x=test_eval_df.index, y=test_eval_df["Predicted"], name="Model Predicted", line=dict(color="#f43f5e", dash="dash")))
        bt_fig.update_layout(height=400, template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(bt_fig, use_container_width=True)

    with tab3:
        sma20_val = float(data["SMA_20"].iloc[-1])
        sma50_val = float(data["SMA_50"].iloc[-1])
        macd_val = float(data["MACD"].iloc[-1])
        macd_sig = float(data["MACD_Signal"].iloc[-1])

        st.markdown("### Automated Technical Checklist")
        signals = [
            ("Trend (SMA 20 vs SMA 50)", "Bullish (Golden Cross Zone)" if sma20_val > sma50_val else "Bearish (Death Cross Zone)", f"SMA20: {sma20_val:,.2f} | SMA50: {sma50_val:,.2f}"),
            ("Momentum (RSI 14)", rsi_status, f"Current RSI: {latest_rsi:.2f}"),
            ("MACD Crossover", "Bullish Momentum" if macd_val > macd_sig else "Bearish Momentum", f"MACD: {macd_val:.2f} | Signal: {macd_sig:.2f}"),
            ("ML Forecast Trajectory", "Upward Projection" if proj_pct > 0 else "Downward Projection", f"{proj_pct:+.2f}% over {forecast_days} days")
        ]
        sig_df = pd.DataFrame(signals, columns=["Indicator", "Signal", "Details"])
        st.table(sig_df)

    with tab4:
        st.markdown("### Historical OHLCV & Calculated Indicators (Latest 100 Trading Days)")
        st.dataframe(data.tail(100).sort_index(ascending=False).style.format("{:,.2f}"), use_container_width=True)
        csv_bytes = raw_df.to_csv().encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {ticker} Historical CSV",
            data=csv_bytes,
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv"
        )
