# 📈 StockSense AI — Student Stock Predictor

> AI-powered stock market prediction website for students. Predicts next 3-day trend using Random Forest ML on the past 5–15 days of data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🚀 Features

- **Real-time data** from Yahoo Finance (NSE/BSE stocks)
- **Random Forest AI** — 100 decision trees for UP/DOWN prediction
- **3-day price forecast** trained on only 5–15 days of data
- **RSI indicator** and feature importance charts
- **Student-friendly UI** with knowledge hub and investment tips
- **Low-price stock suggestions** (under ₹100) for student investors

---

## 🔄 Methodology

| Step | Details |
|------|---------|
| Data Collection | Yahoo Finance — 5–15 days OHLCV |
| Feature Engineering | 10 features including RSI, SMA, ratios |
| Target | Binary: price UP or DOWN after 3 days |
| Model | RandomForestClassifier (100 trees) |
| Split | 80% train / 20% test |
| Output | UP/DOWN signal + confidence % |

---

## 🛠️ Local Setup

```bash
git clone https://github.com/yourusername/stock-predictor
cd stock-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select repo → branch `main` → file `app.py`
5. Click **Deploy** — live in 60 seconds!

---

## 📱 Try These Stocks

| Symbol | Company | Price Range |
|--------|---------|-------------|
| `RELIANCE.NS` | Reliance Industries | ₹2,800+ |
| `TCS.NS` | Tata Consultancy | ₹3,500+ |
| `SUZLON.NS` | Suzlon Energy | ₹30–60 ⭐ Student Pick |
| `YESBANK.NS` | Yes Bank | ₹15–25 ⭐ Student Pick |
| `NHPC.NS` | NHPC Ltd | ₹60–90 ⭐ Student Pick |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is NOT financial advice. Stock markets are risky — never invest money you cannot afford to lose. Always consult a SEBI-registered advisor.

---

**Built with:** Python · Streamlit · yfinance · scikit-learn · Plotly
