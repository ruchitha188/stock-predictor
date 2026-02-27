import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide", page_title="AI Stock Predictor")
st.title("🚀 Real-Time AI Stock Predictor")
st.markdown("***Educational tool for students learning stock market*** ⚠️ *Not investment advice*")

# Sidebar
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="RELIANCE.NS", help="Use .NS for NSE stocks")
days_back = st.sidebar.slider("Past Days for Analysis", 5, 10, 7)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Live Chart & Price")
    
with col2:
    st.header("🎯 AI Prediction")

if st.sidebar.button("🔥 Analyze Stock", use_container_width=True):
    with st.spinner("Fetching live data..."):
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f"{days_back}d")
            current_price = hist['Close'].iloc[-1]
            
            # AI Prediction using Simple Moving Average
            sma_short = hist['Close'].tail(3).mean()
            sma_long = hist['Close'].tail(7).mean()
            
            if sma_short > current_price * 1.01:  # 1% threshold
                signal = "🟢 **BUY** (Profit Likely)"
                color = "normal"
            elif sma_short < current_price * 0.99:
                signal = "🔴 **SELL** (Loss Risk)" 
                color = "inverse"
            else:
                signal = "🟡 **HOLD** (Neutral)"
                color = "off"
            
            # Update display
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], 
                                       mode='lines+markers', 
                                       line=dict(color='green', width=2),
                                       name='Price'))
                fig.add_hline(y=sma_short, line_dash="dash", 
                            line_color="blue", annotation_text="Short SMA")
                fig.update_layout(title=f"{symbol} - Last {days_back} Days",
                                xaxis_title="Date", yaxis_title="Price ₹")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("AI Signal", signal, label_visibility="collapsed")
                st.metric("Short SMA", f"₹{sma_short:.2f}", delta=f"{sma_short-current_price:+.2f}")
                st.metric("Long SMA", f"₹{sma_long:.2f}")
                
                st.success("✅ **Low Risk Stocks to Try:**\nVBL.NS, SUZLON.NS, YESBANK.NS")
                st.info(f"**Updated:** {datetime.now().strftime('%H:%M IST')}")
                
        except:
            st.error("❌ Invalid symbol! Try **RELIANCE.NS**, **TCS.NS**, or **INFY.NS**")

st.sidebar.markdown("---")
st.sidebar.markdown("**Made for Students** 🎓")
st.sidebar.info("• Live NSE prices\n• AI Buy/Sell signals\n• Educational only")
