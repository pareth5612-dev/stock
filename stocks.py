import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Pro Market Dashboard", layout="wide")

# Organized asset database
categories = {
    "Tech Giants": {"Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA"},
    "Automotive": {"Ferrari": "RACE", "Lamborghini (VW)": "VWAGY", "Ford": "F", "Toyota": "TM"},
    "Cryptocurrency": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "Dogecoin": "DOGE-USD"},
    "Consumer Goods": {"Coca-Cola": "KO", "McDonald's": "MCD", "Walmart": "WMT", "Nike": "NKE"}
}

st.title("📈 Pro Market & Crypto Dashboard")

# 1. Subject/Category Selection
selected_category = st.sidebar.selectbox("Select Industry", list(categories.keys()))

# 2. Asset Selection based on Category
selected_asset_name = st.sidebar.selectbox(f"Select from {selected_category}", list(categories[selected_category].keys()))
ticker = categories[selected_category][selected_asset_name]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

# 3. Data Fetching
@st.cache_data
def load_data(ticker, start):
    return yf.download(ticker, start=start)

data = load_data(ticker, start_date)

# Displaying Content
if not data.empty:
    tab1, tab2 = st.tabs(["📊 Price Chart", "🔮 AI Prediction"])
    
    with tab1:
        st.subheader(f"{selected_asset_name} ({ticker})")
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Forecast (using Prophet)")
        df_train = data.reset_index()[['Date', 'Close']]
        df_train.columns = ['ds', 'y']
        
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price', line=dict(color='blue')))
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.error("No data found for this selection.")
