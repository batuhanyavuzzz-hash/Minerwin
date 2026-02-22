# ===============================
# FINNHUB FREE PLAN UYUMLU APP
# ===============================

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Tek Hisse Teknik Analiz (Finnhub)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Finnhub (V1 | Daily Only)")

# API KEY (Streamlit Cloud)
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    st.error("FINNHUB_API_KEY Secrets içinde bulunamadı.")
    st.stop()

# ===============================
# FINNHUB CLIENT
# ===============================
class FinnhubClient:
    def __init__(self, key):
        self.base = "https://finnhub.io/api/v1"
        self.key = key

    def _get(self, path, params):
        params["token"] = self.key
        r = requests.get(f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def quote(self, symbol):
        return self._get("/quote", {"symbol": symbol})

    def daily_candles(self, symbol, _from, _to):
        return self._get(
            "/stock/candle",
            {
                "symbol": symbol,
                "resolution": "D",
                "from": _from,
                "to": _to,
            },
        )

# ===============================
# INDICATORS
# ===============================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    rs = g.ewm(alpha=1/n).mean() / l.ewm(alpha=1/n).mean()
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n).mean()

# ===============================
# ANALYSIS
# ===============================
@dataclass
class Result:
    fit: str
    entry: str
    text: str

def analyze(df):
    c = df.iloc[-1]
    atr_pct = c["atr"] / c["close"] * 100

    fit = "UYGUN" if (
        c["ema50"] > c["ema150"] > c["ema200"]
        and c["close"] > c["ema150"]
        and c["rsi"] >= 55
        and 2 <= atr_pct <= 6
    ) else "SINIRDA / UYGUN DEĞİL"

    entry = "BEKLE (Geri çekilme)" if (c["close"] - c["ema50"]) / c["ema50"] * 100 > 8 else "İZLENEBİLİR"

    text = f"""
**Trend:** EMA dizilimi pozitif  
**RSI:** {c['rsi']:.1f}  
**ATR %:** {atr_pct:.2f}  

**Strateji Uygunluğu:** {fit}  
**Zamanlama:** {entry}
"""

    return Result(fit, entry, text)

# ===============================
# UI
# ===============================
client = FinnhubClient(FINNHUB_API_KEY)

with st.sidebar:
    st.header("Kontroller")
    bars = st.slider("Bar sayısı (Günlük)", 120, 600, 300)

symbol = st.text_input("Ticker", placeholder="NVDA")

if st.button("Getir & Analiz Et") and symbol:
    now = int(time.time())
    _from = now - bars * 2 * 24 * 3600

    try:
        quote = client.quote(symbol.upper())
        candles = client.daily_candles(symbol.upper(), _from, now)

        if candles.get("s") != "ok":
            st.error("Daily candle verisi alınamadı (Finnhub free plan kısıtı).")
        else:
            df = pd.DataFrame({
                "time": pd.to_datetime(candles["t"], unit="s"),
                "open": candles["o"],
                "high": candles["h"],
                "low": candles["l"],
                "close": candles["c"],
                "volume": candles["v"],
            })

            df["ema50"] = ema(df["close"], 50)
            df["ema150"] = ema(df["close"], 150)
            df["ema200"] = ema(df["close"], 200)
            df["rsi"] = rsi(df["close"])
            df["atr"] = atr(df)

            res = analyze(df)

            fig = go.Figure(go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"]
            ))
            fig.add_scatter(x=df["time"], y=df["ema50"], name="EMA50")
            fig.add_scatter(x=df["time"], y=df["ema150"], name="EMA150")
            fig.add_scatter(x=df["time"], y=df["ema200"], name="EMA200")

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(res.text)

    except Exception as e:
        st.error(str(e))
