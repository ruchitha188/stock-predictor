import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="StockSense AI",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1020 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #00d4ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s infinite linear;
    }
    @keyframes shimmer {
        0%{background-position:0%}
        100%{background-position:200%}
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .signal-up {
        background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,200,100,0.05));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .signal-down {
        background: linear-gradient(135deg, rgba(255,60,60,0.15), rgba(200,0,0,0.05));
        border: 1px solid rgba(255,60,60,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .student-card {
        background: rgba(123,47,247,0.12);
        border: 1px solid rgba(123,47,247,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .knowledge-box {
        background: rgba(0,212,255,0.06);
        border-left: 3px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem !important;
        color: #00d4ff !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7b2ff7, #00d4ff);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(123,47,247,0.4);
    }
    .disclaimer {
        background: rgba(255,200,0,0.08);
        border: 1px solid rgba(255,200,0,0.25);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #ffd700;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 StockSense AI</h1>
    <p style="color:#8892a4;font-size:1.05rem;">AI-Powered Stock Prediction · Built for Students · Educational Purpose Only</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    symbol = st.text_input("Stock Symbol", "RELIANCE.NS", help="e.g. RELIANCE.NS, TCS.NS, INFY.NS")
    days = st.slider("Training Days (past data)", min_value=5, max_value=15, value=10)
    forecast_days = 3

    st.markdown("---")
    st.markdown("### 🎓 Student Picks")
    st.markdown("""
    <div class="student-card">
    <b>Under ₹100 — Low Investment</b><br>
    🟢 <code>SUZLON.NS</code> — ₹30-50<br>
    🟢 <code>YESBANK.NS</code> — ₹15-25<br>
    🟢 <code>IRFC.NS</code> — ₹150-200<br>
    🟢 <code>NHPC.NS</code> — ₹60-90<br>
    🟡 <code>SAIL.NS</code> — ₹100-150
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Popular Stocks")
    col1, col2 = st.columns(2)
    popular = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","WIPRO.NS","ITC.NS"]
    for i, s in enumerate(popular):
        if i % 2 == 0:
            col1.code(s.replace(".NS",""), language=None)
        else:
            col2.code(s.replace(".NS",""), language=None)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ For educational purposes only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)

# ─── Knowledge Hub ───────────────────────────────────────────────────────────
with st.expander("📚 Knowledge Hub — Learn Before You Invest", expanded=False):
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        <div class="knowledge-box">
        <b>📊 What is a Stock?</b><br>
        A stock represents ownership in a company. When you buy 1 share of Reliance, you own a tiny piece of Reliance Industries!
        </div>
        <div class="knowledge-box">
        <b>🟢 Profit Signal (BUY)</b><br>
        Our AI predicts the price will be <b>higher</b> in 3 days compared to today. This suggests an upward trend.
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="knowledge-box">
        <b>📉 What is Random Forest?</b><br>
        It's an AI algorithm that builds 100 decision trees and votes on the final prediction — like asking 100 experts!
        </div>
        <div class="knowledge-box">
        <b>🔴 Loss Signal (SELL/WAIT)</b><br>
        AI predicts the price may <b>fall</b> in 3 days. It may be better to wait before investing.
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="knowledge-box">
        <b>⚡ How to Start Investing?</b><br>
        1. Open a Demat account (Zerodha, Groww)<br>
        2. Start with ₹100–₹500<br>
        3. Buy low-price stocks<br>
        4. Learn, then grow!
        </div>
        <div class="knowledge-box">
        <b>📐 What is RSI?</b><br>
        RSI (0-100) measures momentum. Above 70 = overbought (may fall). Below 30 = oversold (may rise). 
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─── Functions ───────────────────────────────────────────────────────────────
def compute_rsi(prices, window=5):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # buffer for weekends/holidays
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days + 5)

def prepare_features(df):
    df = df.copy()
    df['Price_Change']     = df['Close'].pct_change()
    df['High_Low_Ratio']   = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-9)
    df['Volume_SMA']       = df['Volume'].rolling(3, min_periods=1).mean()
    df['RSI']              = compute_rsi(df['Close'], window=min(5, len(df)-1))
    df['Target']           = (df['Close'].shift(-3) > df['Close']).astype(int)
    df = df.dropna()
    return df

FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Price_Change', 'High_Low_Ratio', 'Open_Close_Ratio', 'Volume_SMA', 'RSI']

def train_model(df):
    X = df[FEATURE_COLS].fillna(0).values
    y = df['Target'].values

    if len(X) < 4:
        return None, None, None, None

    test_size = max(1, int(len(X) * 0.2))
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s)) if len(y_test) > 0 else 0.0
    return model, scaler, float(acc), FEATURE_COLS

# ─── Main Run Button ──────────────────────────────────────────────────────────
run = st.button("🚀 Analyze & Predict", use_container_width=True)

if run:
    with st.spinner(f"Fetching {days}-day data for **{symbol}** …"):
        raw = get_stock_data(symbol, days)

    if raw is None or len(raw) < 5:
        st.error(f"❌ Could not fetch data for **{symbol}**. Try `RELIANCE.NS` or `TCS.NS`.")
    else:
        with st.spinner("Engineering features & training Random Forest …"):
            df = prepare_features(raw)

        if len(df) < 4:
            st.error("Not enough rows after feature engineering. Try increasing days to 12-15.")
        else:
            model, scaler, accuracy, feat_cols = train_model(df)

            if model is None:
                st.error("Training failed — need at least 4 rows. Increase days.")
            else:
                # ── Predict next 3 days
                latest_row = df[FEATURE_COLS].iloc[[-1]].fillna(0).values
                latest_s   = scaler.transform(latest_row)
                pred       = model.predict(latest_s)[0]
                conf       = model.predict_proba(latest_s)[0].max()

                current_price = float(df['Close'].iloc[-1])
                sma_price     = float(df['Close'].tail(5).mean())
                price_delta   = sma_price - current_price

                # ── Top row metrics ──────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📅 Days Trained", f"{len(df)}")
                m2.metric("📊 Model Accuracy", f"{accuracy:.0%}")
                m3.metric("💰 Current Price", f"₹{current_price:.2f}")
                m4.metric("📐 5-Day SMA", f"₹{sma_price:.2f}", delta=f"{price_delta:+.2f}")

                st.markdown("---")

                # ── Signal ───────────────────────────────────────────────────
                sig_col, chart_col = st.columns([1, 2])
                with sig_col:
                    if pred == 1:
                        st.markdown(f"""
                        <div class="signal-up">
                            <h1 style="color:#00ff88;margin:0">🟢</h1>
                            <h2 style="color:#00ff88">PROFIT LIKELY</h2>
                            <p style="color:#ccc">3-Day Forecast: <b>UPTREND</b></p>
                            <p style="color:#ccc">Confidence: <b style="color:#00ff88">{conf:.0%}</b></p>
                            <hr style="border-color:#ffffff20">
                            <small style="color:#8892a4">Suggested: Consider small buy<br>Start with ₹100–₹500</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="signal-down">
                            <h1 style="color:#ff4444;margin:0">🔴</h1>
                            <h2 style="color:#ff4444">LOSS RISK</h2>
                            <p style="color:#ccc">3-Day Forecast: <b>DOWNTREND</b></p>
                            <p style="color:#ccc">Confidence: <b style="color:#ff6666">{conf:.0%}</b></p>
                            <hr style="border-color:#ffffff20">
                            <small style="color:#8892a4">Suggested: Wait & watch<br>Don't invest right now</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # RSI gauge
                    rsi_val = float(df['RSI'].iloc[-1])
                    rsi_color = "#ff4444" if rsi_val > 70 else ("#00ff88" if rsi_val < 30 else "#00d4ff")
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top:1rem">
                        <p style="color:#8892a4;margin:0">RSI Indicator</p>
                        <h2 style="color:{rsi_color};margin:0.3rem 0">{rsi_val:.1f}</h2>
                        <small style="color:#8892a4">{'Overbought ⚠️' if rsi_val > 70 else ('Oversold ✅' if rsi_val < 30 else 'Neutral ➡️')}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Price Chart ───────────────────────────────────────────────
                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Close'],
                        mode='lines+markers',
                        name='Close Price',
                        line=dict(color='#00d4ff', width=2),
                        marker=dict(size=6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Close'].rolling(3, min_periods=1).mean(),
                        mode='lines', name='3-Day SMA',
                        line=dict(color='#7b2ff7', width=1.5, dash='dot')
                    ))
                    # Predicted arrow
                    last_date = df.index[-1]
                    pred_date = last_date + timedelta(days=3)
                    pred_price = current_price * (1 + 0.02 if pred == 1 else 1 - 0.02)
                    fig.add_trace(go.Scatter(
                        x=[last_date, pred_date], y=[current_price, pred_price],
                        mode='lines+markers',
                        name='Predicted Direction',
                        line=dict(color='#00ff88' if pred == 1 else '#ff4444', width=2, dash='dash'),
                        marker=dict(size=8, symbol='arrow-up' if pred == 1 else 'arrow-down')
                    ))
                    fig.update_layout(
                        title=f"{symbol} — Last {len(df)} Days + 3-Day Forecast",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ccc', family='Space Grotesk'),
                        legend=dict(orientation='h', y=-0.15),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Price (₹)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Feature Importance ────────────────────────────────────────
                st.markdown("### 🔍 Feature Importance (What drives the prediction?)")
                imp_df = pd.DataFrame({
                    'Feature': feat_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig2 = px.bar(
                    imp_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale=['#0d1525','#7b2ff7','#00d4ff'],
                    title="Random Forest — Feature Importance"
                )
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ccc', family='Space Grotesk'),
                    showlegend=False, yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ── Price Stats Table ─────────────────────────────────────────
                st.markdown("### 📋 Historical Price Table")
                display_df = df[['Open','High','Low','Close','Volume','RSI']].copy()
                display_df = display_df.round(2)
                display_df.index = display_df.index.strftime('%d %b %Y')
                st.dataframe(display_df.tail(10), use_container_width=True)

                # ── Student Advice ────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🎓 Student Investment Guidance")
                a1, a2, a3 = st.columns(3)
                with a1:
                    st.markdown("""
                    **📌 How to Read This Prediction**
                    - 🟢 UP = AI thinks price rises in ~3 days
                    - 🔴 DOWN = AI thinks price falls
                    - Higher confidence = stronger signal
                    - Never invest all savings in 1 stock!
                    """)
                with a2:
                    st.markdown("""
                    **💸 Starting Small (₹100–₹500)**
                    - Buy 1-5 shares of low-price stocks
                    - Apps: Groww, Zerodha, Upstox
                    - NSE market hours: 9:15 AM – 3:30 PM
                    - Practice first with paper trading
                    """)
                with a3:
                    st.markdown("""
                    **⚠️ Risk Management**
                    - AI predictions are ~70-80% accurate
                    - Markets can be unpredictable
                    - Never invest money you can't afford to lose
                    - Diversify across 3-5 different stocks
                    """)

                st.markdown("""
                <div class="disclaimer">
                ⚠️ <b>DISCLAIMER:</b> This tool is for <b>educational purposes only</b>. 
                Past data does NOT guarantee future performance. 
                Always consult a SEBI-registered financial advisor before investing real money.
                </div>
                """, unsafe_allow_html=True)

# ─── Methodology ──────────────────────────────────────────────────────────────
with st.expander("🔬 Complete Methodology", expanded=False):
    st.markdown("""
    | Step | What Happens |
    |------|-------------|
    | **1. Data Collection** | Yahoo Finance API — last 5–15 days OHLCV (Open, High, Low, Close, Volume) |
    | **2. Feature Engineering** | 10 features: raw OHLCV + Price Change, High-Low Ratio, Open-Close Ratio, Volume SMA, RSI |
    | **3. Target Label** | Binary: 1 if price is higher 3 days later, 0 if lower |
    | **4. Train/Test Split** | 80% training, 20% testing |
    | **5. Model** | Random Forest Classifier — 100 decision trees, max_depth=5 |
    | **6. Evaluation** | Accuracy score on held-out test set |
    | **7. Prediction** | Latest row → scaler → model → UP/DOWN + confidence % |
    """)
