import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Pro Market Dashboard", layout="wide")

# Organized Asset Database with short display names
categories = {
    "Tech": {"Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA"},
    "Vehicles": {"Ferrari": "RACE", "Ford": "F", "GM": "GM", "Toyota": "TM", "VW": "VWAGY"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"},
    "Currencies": {"EUR/USD": "EURUSD=X", "USD/ILS": "USDILS=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"}
}

st.title("📈 Market Dashboard")

# 1. Sidebar Navigation
selected_category = st.sidebar.selectbox("Category", list(categories.keys()))
selected_asset_name = st.sidebar.selectbox("Asset", list(categories[selected_category].keys()))
ticker = categories[selected_category][selected_asset_name]
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

# 2. Data Fetching
@st.cache_data
def load_data(ticker, start):
    return yf.download(ticker, start=start, auto_adjust=True)

data = load_data(ticker, start_date)

# 3. Dashboard Display
if data.empty or len(data) < 2:
    st.error("Insufficient data. Please check the asset or date range.")
else:
    # Extract closing price safely
    close_data = data['Close']
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]
        
    current_val = float(close_data.iloc[-1])
    previous_val = float(close_data.iloc[-2])
    change = current_val - previous_val

    st.metric(
        label=f"Current {selected_asset_name}",
        value=f"{current_val:,.4f}",
        delta=f"{change:,.4f}"
    )

    tab1, tab2 = st.tabs(["📊 Price Chart", "🔮 AI Prediction"])
    
    with tab1:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'].iloc[:, 0] if isinstance(data['Open'], pd.DataFrame) else data['Open'],
            high=data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High'],
            low=data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low'],
            close=data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_train = pd.DataFrame({'ds': data.index, 'y': close_data.values})
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=data.index, y=close_data.values, name='Actual Price', line=dict(color='blue')))
        st.plotly_chart(fig2, use_container_width=True)
