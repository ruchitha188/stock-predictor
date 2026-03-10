import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="StockSense AI", page_icon="📈",
                   initial_sidebar_state="expanded")

# ─── ALL STOCK LISTS ──────────────────────────────────────────────────────────
NIFTY50 = {
    "Reliance Industries":    "RELIANCE.NS",
    "TCS":                    "TCS.NS",
    "HDFC Bank":              "HDFCBANK.NS",
    "Infosys":                "INFY.NS",
    "ICICI Bank":             "ICICIBANK.NS",
    "Hindustan Unilever":     "HINDUNILVR.NS",
    "ITC":                    "ITC.NS",
    "State Bank of India":    "SBIN.NS",
    "Bharti Airtel":          "BHARTIARTL.NS",
    "Kotak Mahindra Bank":    "KOTAKBANK.NS",
    "Larsen & Toubro":        "LT.NS",
    "Axis Bank":              "AXISBANK.NS",
    "Asian Paints":           "ASIANPAINT.NS",
    "Maruti Suzuki":          "MARUTI.NS",
    "Sun Pharma":             "SUNPHARMA.NS",
    "Bajaj Finance":          "BAJFINANCE.NS",
    "Titan Company":          "TITAN.NS",
    "Wipro":                  "WIPRO.NS",
    "HCL Technologies":       "HCLTECH.NS",
    "UltraTech Cement":       "ULTRACEMCO.NS",
    "Power Grid":             "POWERGRID.NS",
    "NTPC":                   "NTPC.NS",
    "Tech Mahindra":          "TECHM.NS",
    "Nestle India":           "NESTLEIND.NS",
    "Mahindra & Mahindra":    "M&M.NS",
    "Tata Motors":            "TATAMOTORS.NS",
    "Tata Steel":             "TATASTEEL.NS",
    "JSW Steel":              "JSWSTEEL.NS",
    "Bajaj Auto":             "BAJAJ-AUTO.NS",
    "Hindalco":               "HINDALCO.NS",
    "Grasim Industries":      "GRASIM.NS",
    "IndusInd Bank":          "INDUSINDBK.NS",
    "Dr. Reddys Labs":        "DRREDDY.NS",
    "Cipla":                  "CIPLA.NS",
    "Eicher Motors":          "EICHERMOT.NS",
    "Hero MotoCorp":          "HEROMOTOCO.NS",
    "Divis Laboratories":     "DIVISLAB.NS",
    "BPCL":                   "BPCL.NS",
    "ONGC":                   "ONGC.NS",
    "Coal India":             "COALINDIA.NS",
    "Adani Ports":            "ADANIPORTS.NS",
    "Adani Enterprises":      "ADANIENT.NS",
    "SBI Life Insurance":     "SBILIFE.NS",
    "HDFC Life":              "HDFCLIFE.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "UPL":                    "UPL.NS",
    "Apollo Hospitals":       "APOLLOHOSP.NS",
    "Britannia":              "BRITANNIA.NS",
    "Shriram Finance":        "SHRIRAMFIN.NS",
    "BEL":                    "BEL.NS",
}

STUDENT_STOCKS = {
    # Under 50
    "Suzlon Energy":       "SUZLON.NS",
    "Yes Bank":            "YESBANK.NS",
    "Vodafone Idea":       "IDEA.NS",
    "IFCI":                "IFCI.NS",
    "Trident Ltd":         "TRIDENT.NS",
    "Bank of Maharashtra": "MAHABANK.NS",
    "Central Bank":        "CENTRALBK.NS",
    "Jaiprakash Power":    "JPPOWER.NS",
    # 50 to 200
    "NHPC":                "NHPC.NS",
    "IRFC":                "IRFC.NS",
    "Punjab National Bank":"PNB.NS",
    "Canara Bank":         "CANBK.NS",
    "Bank of Baroda":      "BANKBARODA.NS",
    "Indian Oil":          "IOC.NS",
    "HPCL":                "HPCL.NS",
    "Tata Power":          "TATAPOWER.NS",
    "SAIL":                "SAIL.NS",
    "REC Ltd":             "RECLTD.NS",
    "PFC":                 "PFC.NS",
    "IREDA":               "IREDA.NS",
    "SJVN":                "SJVN.NS",
    "Hindustan Copper":    "HINDCOPPER.NS",
    # 200 to 500
    "Vedanta":             "VEDL.NS",
    "NMDC":                "NMDC.NS",
    "Ashok Leyland":       "ASHOKLEY.NS",
    "IDBI Bank":           "IDBI.NS",
    "Union Bank":          "UNIONBANK.NS",
    "Indian Bank":         "INDIANB.NS",
    "NALCO":               "NATIONALUM.NS",
}

SECTOR_MAP = {
    "Banking & Finance": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS",
                          "AXISBANK.NS","BAJFINANCE.NS","INDUSINDBK.NS","YESBANK.NS",
                          "PNB.NS","CANBK.NS","BANKBARODA.NS","IDBI.NS","UNIONBANK.NS",
                          "INDIANB.NS","MAHABANK.NS","CENTRALBK.NS","SHRIRAMFIN.NS",
                          "IRFC.NS","RECLTD.NS","PFC.NS","SBILIFE.NS","HDFCLIFE.NS"],
    "IT & Technology":   ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS"],
    "Energy & Oil":      ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HPCL.NS",
                          "TATAPOWER.NS","NTPC.NS","POWERGRID.NS","NHPC.NS",
                          "SJVN.NS","IREDA.NS","SUZLON.NS","JPPOWER.NS"],
    "Metals & Mining":   ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","SAIL.NS",
                          "VEDL.NS","NMDC.NS","HINDCOPPER.NS","NATIONALUM.NS","COALINDIA.NS"],
    "Pharma & Health":   ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS"],
    "Auto":              ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
                          "EICHERMOT.NS","M&M.NS","ASHOKLEY.NS"],
    "Infrastructure":    ["LT.NS","ADANIPORTS.NS","ADANIENT.NS","ULTRACEMCO.NS","GRASIM.NS"],
    "FMCG & Consumer":   ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","TITAN.NS",
                          "ASIANPAINT.NS","BRITANNIA.NS","TATACONSUM.NS"],
    "Telecom":           ["BHARTIARTL.NS","IDEA.NS"],
}

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:linear-gradient(160deg,#060b18 0%,#0c1428 60%,#060d1f 100%);}
.hero{text-align:center;padding:2.5rem 0 1.5rem;}
.hero h1{font-size:2.8rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8,#f472b6,#38bdf8);background-size:300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:flow 4s linear infinite;}
@keyframes flow{0%{background-position:0%}100%{background-position:300%}}
.hero p{color:#64748b;font-size:1rem;margin-top:0.4rem;}
.stock-chip{display:inline-block;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);border-radius:20px;padding:4px 12px;margin:3px;font-size:0.78rem;color:#38bdf8;}
.student-chip{background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);color:#4ade80;}
.card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:1.2rem;margin:0.4rem 0;}
.up-card{background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.25);border-radius:16px;padding:1.5rem;text-align:center;}
.dn-card{background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.25);border-radius:16px;padding:1.5rem;text-align:center;}
.disc{background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.2);border-radius:10px;padding:.7rem 1rem;font-size:.8rem;color:#fbbf24;text-align:center;margin-top:1rem;}
div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;color:#38bdf8!important;}
.stButton>button{background:linear-gradient(90deg,#6366f1,#38bdf8);color:white;border:none;border-radius:12px;font-weight:700;font-size:.95rem;padding:.7rem 1.5rem;width:100%;transition:all .3s;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,102,241,.4);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 StockSense AI</h1>
  <p>Nifty 50 · Student Picks · Sector View · AI 3-Day Forecast · Real-Time NSE Data</p>
</div>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def compute_rsi(prices, w=5):
    d = prices.diff()
    g = d.where(d > 0, 0.).rolling(w).mean()
    l = (-d.where(d < 0, 0.)).rolling(w).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def fetch_data(sym, days):
    end = datetime.now()
    start = end - timedelta(days=days + 12)
    d = yf.download(sym, start=start, end=end, progress=False)
    if d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().tail(days + 5)

def engineer_features(df):
    df = df.copy()
    df['PC']  = df['Close'].pct_change()
    df['HLR'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['OCR'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-9)
    df['VS']  = df['Volume'].rolling(3, min_periods=1).mean()
    df['RSI'] = compute_rsi(df['Close'], min(5, max(2, len(df) - 1)))
    df['TGT'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    return df.dropna()

FEATS = ['Open', 'High', 'Low', 'Close', 'Volume', 'PC', 'HLR', 'OCR', 'VS', 'RSI']
FEAT_LABELS = ['Open', 'High', 'Low', 'Close', 'Volume',
               'Price Change', 'H-L Ratio', 'O-C Ratio', 'Vol SMA', 'RSI']

def train_model(df):
    X = df[FEATS].fillna(0).values
    y = df['TGT'].values
    if len(X) < 4:
        return None
    n = max(1, int(len(X) * 0.2))
    Xtr, Xte = X[:-n], X[-n:]
    ytr, yte = y[:-n], y[-n:]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    m = RandomForestClassifier(100, max_depth=5, random_state=42)
    m.fit(Xtr_s, ytr)
    acc = accuracy_score(yte, m.predict(Xte_s)) if len(yte) > 0 else 0.
    return m, sc, float(acc)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    days = st.slider("Training Days", 5, 15, 10)
    st.markdown("---")
    mode = st.radio("🔍 Mode", [
        "🔎 Single Stock Analysis",
        "📊 Market Dashboard",
        "🎓 Student Starter Pack"
    ])
    st.markdown("---")

    if mode == "🔎 Single Stock Analysis":
        category = st.selectbox("Category", ["🏆 Nifty 50", "🎓 Student Picks", "✏️ Custom Symbol"])
        if category == "🏆 Nifty 50":
            name_sel = st.selectbox("Stock", list(NIFTY50.keys()))
            symbol = NIFTY50[name_sel]
        elif category == "🎓 Student Picks":
            name_sel = st.selectbox("Stock", list(STUDENT_STOCKS.keys()))
            symbol = STUDENT_STOCKS[name_sel]
        else:
            symbol = st.text_input("Symbol (e.g. INFY.NS)", "INFY.NS")
            name_sel = symbol

    elif mode == "📊 Market Dashboard":
        scan_cat = st.selectbox("Scan Pool", [
            "Top 20 Nifty 50", "Full Nifty 50", "Student Picks"
        ])

    else:
        budget = st.select_slider("Max Price per Share (₹)", [25, 50, 100, 200, 500], value=200)

    st.markdown("""<div class="disc">⚠️ Educational use only.<br>Not financial advice.</div>""",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔎 Single Stock Analysis":
    st.markdown(f"### 🔎 Analyzing: `{symbol}`")
    run = st.button("🚀 Predict Now", use_container_width=True)

    if run:
        with st.spinner(f"Fetching {days}-day data for {symbol}…"):
            raw = fetch_data(symbol, days)

        if raw is None or len(raw) < 5:
            st.error("❌ No data found. Try RELIANCE.NS or SUZLON.NS")
        else:
            with st.spinner("Training Random Forest AI…"):
                df = engineer_features(raw)

            if len(df) < 4:
                st.error("Not enough data rows. Try 12-15 days.")
            else:
                result = train_model(df)
                if result is None:
                    st.error("Training failed. Try more days.")
                else:
                    m, sc, acc = result
                    row  = df[FEATS].iloc[[-1]].fillna(0).values
                    pred = m.predict(sc.transform(row))[0]
                    conf = m.predict_proba(sc.transform(row))[0].max()
                    price = float(df['Close'].iloc[-1])
                    sma5  = float(df['Close'].tail(5).mean())
                    rsi_v = float(df['RSI'].iloc[-1])
                    chg   = float(df['PC'].iloc[-1]) * 100

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("💰 Price",      f"₹{price:.2f}", f"{chg:+.2f}%")
                    c2.metric("📊 Accuracy",   f"{acc:.0%}")
                    c3.metric("🎯 Confidence", f"{conf:.0%}")
                    c4.metric("📐 5-Day SMA",  f"₹{sma5:.2f}")
                    c5.metric("⚡ RSI",        f"{rsi_v:.1f}")

                    st.markdown("---")
                    sig_col, chart_col = st.columns([1, 2])

                    with sig_col:
                        if pred == 1:
                            st.markdown(f"""
                            <div class="up-card">
                              <div style="font-size:3rem">🟢</div>
                              <h2 style="color:#4ade80;margin:.3rem 0">PROFIT LIKELY</h2>
                              <p style="color:#94a3b8">3-Day Forecast: <b>UPTREND</b></p>
                              <p style="color:#94a3b8">Confidence: <b style="color:#4ade80">{conf:.0%}</b></p>
                              <hr style="border-color:#ffffff15">
                              <small style="color:#64748b">AI suggests a small buy<br>Start with ₹100–₹500</small>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="dn-card">
                              <div style="font-size:3rem">🔴</div>
                              <h2 style="color:#f87171;margin:.3rem 0">LOSS RISK</h2>
                              <p style="color:#94a3b8">3-Day Forecast: <b>DOWNTREND</b></p>
                              <p style="color:#94a3b8">Confidence: <b style="color:#f87171">{conf:.0%}</b></p>
                              <hr style="border-color:#ffffff15">
                              <small style="color:#64748b">Wait for a better entry<br>Don't rush to invest</small>
                            </div>""", unsafe_allow_html=True)

                        rsi_color = "#f87171" if rsi_v > 70 else ("#4ade80" if rsi_v < 30 else "#38bdf8")
                        rsi_label = "Overbought ⚠️" if rsi_v > 70 else ("Oversold ✅" if rsi_v < 30 else "Neutral ➡️")
                        st.markdown(f"""
                        <div class="card" style="margin-top:1rem;text-align:center">
                          <p style="color:#64748b;margin:0;font-size:.8rem">RSI</p>
                          <h2 style="color:{rsi_color};margin:.2rem 0">{rsi_v:.1f}</h2>
                          <small style="color:#64748b">{rsi_label}</small>
                        </div>""", unsafe_allow_html=True)

                    with chart_col:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                            mode='lines+markers', name='Close',
                            line=dict(color='#38bdf8', width=2), marker=dict(size=5)))
                        fig.add_trace(go.Scatter(x=df.index,
                            y=df['Close'].rolling(3, min_periods=1).mean(),
                            mode='lines', name='3D SMA',
                            line=dict(color='#818cf8', width=1.5, dash='dot')))
                        last_dt = df.index[-1]
                        pred_p  = price * (1.02 if pred == 1 else 0.98)
                        fig.add_trace(go.Scatter(
                            x=[last_dt, last_dt + timedelta(days=3)],
                            y=[price, pred_p],
                            mode='lines+markers', name='AI Forecast',
                            line=dict(color='#4ade80' if pred == 1 else '#f87171',
                                      width=2.5, dash='dash'),
                            marker=dict(size=9,
                                        symbol='triangle-up' if pred == 1 else 'triangle-down')))
                        fig.update_layout(
                            title=f"{symbol} — {len(df)}-Day Chart + 3-Day Forecast",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#94a3b8', family='Syne'),
                            legend=dict(orientation='h', y=-0.15),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Price (₹)'))
                        st.plotly_chart(fig, use_container_width=True)

                    imp = pd.DataFrame({'Feature': FEAT_LABELS,
                                        'Importance': m.feature_importances_}).sort_values('Importance')
                    fig2 = px.bar(imp, x='Importance', y='Feature', orientation='h',
                                  color='Importance',
                                  color_continuous_scale=['#0c1428', '#6366f1', '#38bdf8'],
                                  title="Feature Importance")
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                       plot_bgcolor='rgba(0,0,0,0)',
                                       font=dict(color='#94a3b8', family='Syne'),
                                       showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

                    st.markdown("### 📋 Historical Data")
                    disp = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].copy().round(2)
                    disp.index = disp.index.strftime('%d %b %Y')
                    st.dataframe(disp.tail(12), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — MARKET DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📊 Market Dashboard":
    st.markdown("## 📊 Market Dashboard — Live AI Signals")

    if scan_cat == "Top 20 Nifty 50":
        scan_pool = dict(list(NIFTY50.items())[:20])
    elif scan_cat == "Full Nifty 50":
        scan_pool = NIFTY50
    else:
        scan_pool = STUDENT_STOCKS

    scan_btn = st.button(f"🔍 Scan All {len(scan_pool)} Stocks Now", use_container_width=True)

    if scan_btn:
        prog = st.progress(0, text="Scanning stocks…")
        results = []
        items = list(scan_pool.items())
        for i, (name, sym) in enumerate(items):
            prog.progress((i + 1) / len(items), text=f"Analyzing {name}…")
            try:
                raw = fetch_data(sym, days)
                if raw is None or len(raw) < 5:
                    continue
                df = engineer_features(raw)
                if len(df) < 4:
                    continue
                res = train_model(df)
                if res is None:
                    continue
                m, sc, acc = res
                row  = df[FEATS].iloc[[-1]].fillna(0).values
                pred = m.predict(sc.transform(row))[0]
                conf = m.predict_proba(sc.transform(row))[0].max()
                price = float(df['Close'].iloc[-1])
                chg   = float(df['PC'].iloc[-1]) * 100
                rsi_v = float(df['RSI'].iloc[-1])
                results.append(dict(
                    Name=name,
                    Symbol=sym.replace('.NS', ''),
                    Price=round(price, 2),
                    Change=round(chg, 2),
                    RSI=round(rsi_v, 1),
                    Signal="🟢 BUY" if pred == 1 else "🔴 WAIT",
                    Confidence=f"{conf:.0%}",
                    Accuracy=f"{acc:.0%}"
                ))
            except Exception:
                continue
        prog.empty()

        if not results:
            st.error("Could not fetch data. Check internet.")
        else:
            res_df = pd.DataFrame(results)
            buys  = res_df[res_df['Signal'] == "🟢 BUY"]
            waits = res_df[res_df['Signal'] == "🔴 WAIT"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Scanned",     len(res_df))
            c2.metric("🟢 BUY",         len(buys))
            c3.metric("🔴 WAIT",        len(waits))
            c4.metric("📈 Market Mood", "Bullish 🐂" if len(buys) > len(waits) else "Bearish 🐻")

            st.markdown("---")

            pie_fig = px.pie(
                values=[len(buys), len(waits)],
                names=['BUY 🟢', 'WAIT 🔴'],
                color_discrete_sequence=['#4ade80', '#f87171'],
                hole=0.55, title="Signal Distribution"
            )
            pie_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#94a3b8', family='Syne'))

            bar_fig = px.bar(
                res_df.sort_values('Change'), x='Symbol', y='Change',
                color='Signal',
                color_discrete_map={"🟢 BUY": "#4ade80", "🔴 WAIT": "#f87171"},
                title="Day Change % by Stock"
            )
            bar_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#94a3b8', family='Syne'),
                                   xaxis_tickangle=-45,
                                   xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                   yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))

            p1, p2 = st.columns(2)
            p1.plotly_chart(pie_fig, use_container_width=True)
            p2.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("### 🟢 BUY Signals")
            st.dataframe(buys.sort_values('Confidence', ascending=False)
                            .reset_index(drop=True), use_container_width=True)

            st.markdown("### 🔴 WAIT Signals")
            st.dataframe(waits.sort_values('Confidence', ascending=False)
                            .reset_index(drop=True), use_container_width=True)

            st.markdown("### 🏭 Sector Breakdown")
            for sector, syms in SECTOR_MAP.items():
                sector_hits = res_df[res_df['Symbol'].apply(lambda s: s + '.NS' in syms)]
                if sector_hits.empty:
                    continue
                b = len(sector_hits[sector_hits['Signal'] == "🟢 BUY"])
                w = len(sector_hits) - b
                mood = "🐂 Bullish" if b > w else ("🐻 Bearish" if w > b else "➡️ Neutral")
                with st.expander(f"{sector} — {mood} ({b} buy / {w} wait)"):
                    st.dataframe(sector_hits.reset_index(drop=True), use_container_width=True)

    else:
        st.markdown("### 🏆 All Nifty 50 Stocks Available")
        chips = "".join(
            f'<span class="stock-chip">📈 {n} <code style="font-size:.7rem;color:#64748b">{s.replace(".NS","")}</code></span>'
            for n, s in NIFTY50.items()
        )
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown("### 🎓 Student Stocks Available")
        chips2 = "".join(
            f'<span class="stock-chip student-chip">💚 {n} <code style="font-size:.7rem;color:#64748b">{s.replace(".NS","")}</code></span>'
            for n, s in STUDENT_STOCKS.items()
        )
        st.markdown(chips2, unsafe_allow_html=True)

        st.markdown("### 🏭 Stocks by Sector")
        all_stocks = {**NIFTY50, **STUDENT_STOCKS}
        for sector, syms in SECTOR_MAP.items():
            with st.expander(f"{'🏦' if 'Bank' in sector else '💻' if 'IT' in sector else '🏭'} {sector}"):
                names_in = {n: s for n, s in all_stocks.items() if s in syms}
                html = "".join(
                    f'<span class="stock-chip">{n} <code style="font-size:.7rem;color:#64748b">{s.replace(".NS","")}</code></span>'
                    for n, s in names_in.items()
                )
                st.markdown(html or "More stocks being added…", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — STUDENT STARTER PACK
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🎓 Student Starter Pack")
    st.markdown(f"**Budget filter: under ₹{budget} per share** — Perfect for learning with real money")

    with st.expander("📚 Before You Invest — Must Read!", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown("""**🏦 Open Demat Account**
- Groww (easiest)
- Zerodha (popular)
- Upstox (low cost)
- Angel One (beginner)

*Free to open, 10 min*""")
        k2.markdown("""**💡 How to Start**
1. Open Demat account
2. Link bank account
3. Add ₹100–₹500
4. Buy 1–5 shares
5. Watch & learn daily

*NSE: 9:15AM–3:30PM IST*""")
        k3.markdown("""**📊 What to Watch**
- Daily price change %
- RSI momentum
- Volume (trade activity)
- 52-week high/low
- Company news

*Start with 1 sector*""")
        k4.markdown("""**⚠️ Risk Rules**
- Never invest savings
- Max ₹500 to start
- Buy 3-5 stocks
- Don't panic on dips
- Hold 1–3 months

*Losses = paid lessons*""")

    scan_btn = st.button(f"🔍 Scan All Student Stocks Under ₹{budget}", use_container_width=True)

    if scan_btn:
        prog = st.progress(0, text="Scanning student stocks…")
        results = []
        items = list(STUDENT_STOCKS.items())
        for i, (name, sym) in enumerate(items):
            prog.progress((i + 1) / len(items), text=f"Checking {name}…")
            try:
                raw = fetch_data(sym, days)
                if raw is None or len(raw) < 5:
                    continue
                price = float(raw['Close'].iloc[-1])
                if price > budget:
                    continue
                df = engineer_features(raw)
                if len(df) < 4:
                    continue
                res = train_model(df)
                if res is None:
                    continue
                m, sc, acc = res
                row  = df[FEATS].iloc[[-1]].fillna(0).values
                pred = m.predict(sc.transform(row))[0]
                conf = m.predict_proba(sc.transform(row))[0].max()
                chg   = float(df['PC'].iloc[-1]) * 100
                rsi_v = float(df['RSI'].iloc[-1])
                results.append(dict(
                    Name=name,
                    Symbol=sym.replace('.NS', ''),
                    Price=f"₹{price:.2f}",
                    Signal="🟢 BUY" if pred == 1 else "🔴 WAIT",
                    Confidence=f"{conf:.0%}",
                    Accuracy=f"{acc:.0%}",
                    RSI=f"{rsi_v:.1f}",
                    DayChange=f"{chg:+.2f}%",
                    Shares_with_100=int(100 / price) if price > 0 else 0,
                    Shares_with_500=int(500 / price) if price > 0 else 0,
                ))
            except Exception:
                continue
        prog.empty()

        if not results:
            st.warning(f"No stocks found under ₹{budget}. Try increasing the budget slider.")
        else:
            res_df = pd.DataFrame(results)
            buys  = res_df[res_df['Signal'] == "🟢 BUY"]
            waits = res_df[res_df['Signal'] == "🔴 WAIT"]

            c1, c2, c3 = st.columns(3)
            c1.metric("💚 BUY Signals",  len(buys))
            c2.metric("🔴 WAIT Signals", len(waits))
            c3.metric("💸 Budget",       f"₹{budget}")

            st.markdown("### 🟢 Best Student Picks — AI says BUY")
            if not buys.empty:
                st.dataframe(buys.reset_index(drop=True), use_container_width=True)
                st.info("💡 **Shares_with_100** = how many shares you can buy with ₹100. "
                        "**Shares_with_500** = shares with ₹500.")
            else:
                st.info("No BUY signals right now. Market may be bearish. Good time to observe!")

            st.markdown("### 🔴 Watch List — Wait for Better Price")
            if not waits.empty:
                st.dataframe(waits.reset_index(drop=True), use_container_width=True)

    else:
        st.markdown("### 🎯 All Student-Friendly Stocks")
        groups = [
            ("🟤 Under ₹50 — Ultra Low Budget", [(n, s) for n, s in STUDENT_STOCKS.items()
             if n in ["Suzlon Energy","Yes Bank","Vodafone Idea","IFCI","Trident Ltd",
                      "Bank of Maharashtra","Central Bank","Jaiprakash Power"]]),
            ("🟡 ₹50–₹200 — Starter Range", [(n, s) for n, s in STUDENT_STOCKS.items()
             if n in ["NHPC","IRFC","Punjab National Bank","Canara Bank","Bank of Baroda",
                      "Indian Oil","HPCL","Tata Power","SAIL","REC Ltd","PFC",
                      "IREDA","SJVN","Hindustan Copper"]]),
            ("🟠 ₹200–₹500 — Growing Budget", [(n, s) for n, s in STUDENT_STOCKS.items()
             if n in ["Vedanta","NMDC","Ashok Leyland","IDBI Bank","Union Bank",
                      "Indian Bank","NALCO"]]),
        ]
        for label, stocks in groups:
            st.markdown(f"**{label}**")
            html = "".join(
                f'<span class="stock-chip student-chip">💚 {n} <code style="font-size:.7rem;color:#64748b">{s.replace(".NS","")}</code></span>'
                for n, s in stocks
            )
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("")

# ─── KNOWLEDGE HUB ───────────────────────────────────────────────────────────
with st.expander("📚 Knowledge Hub — Learn Stock Market Basics", expanded=False):
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown("""**📈 What is a Stock?**
Owning a stock = owning a small piece of a company. Reliance earns more profit → your stock goes up!

**📉 What is a Loss?**
If company performs badly or market sentiment drops, price falls below your buy price.""")
    h2.markdown("""**🤖 How Random Forest Works**
100 decision trees each vote UP or DOWN. The majority wins — like asking 100 experts!

**📐 What is RSI?**
RSI 0-100. Above 70 = overbought (may fall). Below 30 = oversold (may rise).""")
    h3.markdown("""**📊 What is SMA?**
Average of last N days' prices. Price > SMA = uptrend. Price < SMA = downtrend.

**📦 What is Volume?**
Shares traded per day. High volume = more people interested = stronger signal.""")
    h4.markdown("""**💸 How Much to Invest?**
Start ₹100–₹500. Buy 1-5 shares. Watch daily. Goal = learning, not instant profit.

**🔄 When to Sell?**
Target: sell at +10% profit. Stop loss: exit at -5%. Never hold forever in hope.""")

st.markdown("""
<div class="disc">
⚠️ <b>DISCLAIMER:</b> StockSense AI is for <b>educational purposes only</b>.
AI predictions are not guaranteed. Past data does not guarantee future results.
Never invest money you cannot afford to lose. Consult a SEBI-registered advisor before real investments.
</div>
""", unsafe_allow_html=True)
