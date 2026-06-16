import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Stock Analysis App", layout="wide")

st.title("📈 Professional Stock Dashboard")

# 1. Sidebar Inputs
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

# 2. Fetch Data
# We use a try-except block to handle potential errors with invalid tickers
try:
    data = yf.download(ticker, start=start_date)
    
    if data.empty:
        st.error("No data found for this ticker. Please check the symbol.")
    else:
        # 3. Tabs for Dashboard
        tab1, tab2 = st.tabs(["Interactive Chart", "AI Prediction"])

        with tab1:
            st.subheader(f"{ticker} Historical Data")
            fig = go.Figure(data=[go.Candlestick(x=data.index, 
                                                open=data['Open'], 
                                                high=data['High'], 
                                                low=data['Low'], 
                                                close=data['Close'])])
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Forecast (using Prophet)")
            # Prepare data for Prophet
            df_train = data.reset_index()[['Date', 'Close']]
            df_train.columns = ['ds', 'y']
            
            # Initialize and fit the Prophet model
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # Plotting the forecast
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction'))
            fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
            st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
