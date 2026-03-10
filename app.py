import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

st.set_page_config(layout="wide", page_title="AI Stock Predictor - Random Forest")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4;}
.metric-card {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 AI Stock Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Random Forest ML Model | Up/Down Trend Prediction | Live NSE Data**")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")
period = st.sidebar.selectbox("Training Data", ["1y", "2y", "5y"], index=1)

@st.cache_data
def fetch_and_process_data(symbol, period):
    """Data Collection & Preprocessing"""
    st.info("📥 Collecting data...")
    stock = yf.download(symbol, period=period)
    
    if stock.empty:
        st.error("No data found!")
        return None
    
    # Features: Open, High, Low, Close, Volume
    df = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Feature Engineering
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Close'] / df['Open']
    df['Volume_SMA'] = df['Volume'].rolling(5).mean()
    
    # Target: 1 = Up, 0 = Down (next day)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    return df

@st.cache_data
def train_model(df):
    """Train Random Forest Model"""
    st.info("🎯 Training Random Forest...")
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 
                'High_Low_Ratio', 'Open_Close_Ratio', 'Volume_SMA']
    X = df[features]
    y = df['Target']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred

# Main App
if symbol:
    df = fetch_and_process_data(symbol, period)
    if df is not None:
        model, scaler, accuracy, X_test, y_test, y_pred = train_model(df)
        
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        with col2:
            latest_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:].values
            features = np.column_stack([latest_data.flatten(), 
                                      df['Price_Change'].iloc[-1],
                                      df['High_Low_Ratio'].iloc[-1],
                                      df['Open_Close_Ratio'].iloc[-1],
                                      df['Volume_SMA'].iloc[-1]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            signal = "🟢 **UP (BUY)**" if prediction == 1 else "🔴 **DOWN (SELL)**"
            st.markdown(f'<div class="metric-card"><h2>{signal}</h2><p>Confidence: {max(prob):.1%}</p></div>', unsafe_allow_html=True)
        with col3:
            st.metric("Last Close", f"₹{df['Close'].iloc[-1]:.2f}")
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                       low=df['Low'], close=df['Close'],
                                       name='Price'))
            fig.update_layout(title=f"{symbol} - OHLC Chart")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature Importance
            importances = pd.DataFrame({
                'Feature': ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 
                           'High_Low_Ratio', 'Open_Close_Ratio', 'Volume_SMA'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importances.set_index('Feature'))
        
        # Model Report
        st.subheader("📊 Model Evaluation")
        st.text(f"Dataset Size: {len(df):,} rows")
        st.text(f"Features Used: 9 (OHLCV + Engineered)")
        st.success("✅ Random Forest reduces overfitting with 100 Decision Trees!")

# Sidebar Info
with st.sidebar.expander("📋 Project Methodology"):
    st.markdown("""
    1. **Data Collection**: Yahoo Finance API
    2. **Preprocessing**: Clean + Normalize
    3. **Feature Engineering**: Ratios + SMA
    4. **Train/Test Split**: 80/20
    5. **Random Forest**: 100 Trees
    6. **Evaluation**: Accuracy + Feature Importance
    7. **Deployment**: Streamlit Cloud
    """)

st.sidebar.markdown("---")
st.markdown("**Future Scope**: Real-time + News Sentiment + Mobile App")
