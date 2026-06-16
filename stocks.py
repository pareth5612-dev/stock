import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Pro Market Dashboard", layout="wide")

st.title("📈 Pro Market & Crypto Dashboard")

# Define your asset list here
# You can add as many as you want in the format "Name": "Ticker"
assets = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Ferrari": "RACE",
    "Lamborghini (VW Group)": "VWAGY",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "Netflix": "NFLX"
}

# 1. Sidebar Selection
selected_name = st.sidebar.selectbox("Search/Select Asset", list(assets.keys()))
ticker = assets[selected_name]
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

# 2. Fetch Data
@st.cache_data  # This makes the app faster by caching the data
def load_data(ticker, start):
    return yf.download(ticker, start=start)

data = load_data(ticker, start_date)

if data.empty:
    st.error("No data found. Check your symbol or date range.")
else:
    # 3. Tabs
    tab1, tab2 = st.tabs(["📊 Price Chart", "🔮 AI Prediction"])

    with tab1:
        st.subheader(f"{selected_name} ({ticker})")
        fig = go.Figure(data=[go.Candlestick(x=data.index, 
                                            open=data['Open'], high=data['High'], 
                                            low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Forecast (using Prophet)")
        # Prepare data for Prophet
        df_train = data.reset_index()[['Date', 'Close']]
        df_train.columns = ['ds', 'y']
        
        # Prophet model
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        
        # Visualize forecast
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price', line=dict(color='blue')))
        st.plotly_chart(fig2, use_container_width=True)
