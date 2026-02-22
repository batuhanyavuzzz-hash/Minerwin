import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Tek Hisse Teknik Analiz (Twelve Data)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Twelve Data (V1)")

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error('TWELVEDATA_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine ekle: TWELVEDATA_API_KEY="..."')
    st.stop()


# =========================================================
# TWELVE DATA CLIENT
# =========================================================
class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://api.twelvedata.com"

    def time_series(self, symbol: str, interval: str, outputsize: int = 300) -> dict:
        # interval examples: "1day", "1h", "15min"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": self.api_key,
            "format": "JSON",
        }
        r = requests.get(f"{self.base}/time_series", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def quote(self, symbol: str) -> dict:
        params = {"symbol": symbol, "apikey": self.api_key, "format": "JSON"}
        r = requests.get(f"{self.base}/quote", params=params, timeout=20)
        r.raise_for_status()
        return r.json()


client = TwelveDataClient(API_KEY)


# =========================================================
# INDICATORS
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.bfill()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def slope(series: pd.Series, lookback: int = 20) -> float:
    s = series.dropna()
    if len(s) < lookback + 2:
        return float("nan")
    y = s.iloc[-lookback:].values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


# =========================================================
# STRATEGY EVALUATION (V1)
# =========================================================
@dataclass
class StrategyFitResult:
    fit_label: str
    entry_label: str
    buy_zone_text: str
    narrative: str
    debug: dict


def evaluate(df: pd.DataFrame) -> StrategyFitResult:
    last = df.iloc[-1]
    close = float(last["close"])
    ema50 = float(last["ema50"])
    ema150 = float(last["ema150"])
    ema200 = float(last["ema200"])
    rsi14 = float(last["rsi14"])
    atr14 = float(last["atr14"])
    atr_pct = (atr14 / close) * 100 if close else float("nan")

    trend_stack = (ema50 > ema150 > ema200)
    dist_ema150_pct = ((close - ema150) / ema150) * 100 if ema150 else float("nan")
    price_above_ema150 = close > ema150
    price_borderline_ema150 = (-2.0 <= dist_ema150_pct <= 0.0)

    ema200_slope = slope(df["ema200"], lookback=20)
    long_trend_ok = (ema200_slope > 0)

    momentum_ok = (rsi14 >= 55)
    momentum_borderline = (50 <= rsi14 < 55)

    vol_ok = (2.0 <= atr_pct <= 6.0)
    vol_borderline = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 9.0)
    vol_fail = (atr_pct > 9.0)

    price_below_ema200 = close < ema200

    core_ok = 0
    core_ok += int(trend_stack)
    core_ok += int(price_above_ema150 or price_borderline_ema150)
    core_ok += int(long_trend_ok)
    core_ok += int(momentum_ok or momentum_borderline)
    core_ok += int(vol_ok or vol_borderline)

    if price_below_ema200 or (vol_fail and rsi14 < 55 and not trend_stack):
        fit_label = "UYGUN DEĞİL"
    elif core_ok >= 5:
        fit_label = "UYGUN"
    elif core_ok >= 3:
        fit_label = "SINIRDA"
    else:
        fit_label = "UYGUN DEĞİL"

    dist_ema50_pct = ((close - ema50) / ema50) * 100 if ema50 else float("nan")
    extended = (dist_ema50_pct > 8.0)

    lookback = 20
    breakout = False
    if len(df) >= lookback + 5:
        hh20 = df["high"].iloc[-lookback:].max()
        vol_sma20 = df["volume"].iloc[-lookback:].mean()
        breakout = (close >= hh20 * 0.995) and (float(last["volume"]) >= 1.5 * vol_sma20)

    if fit_label == "UYGUN DEĞİL":
        entry_label = "ŞU AN UYGUN DEĞİL"
        buy_zone_text = "EMA150/EMA200 üzeri kabul + RSI≥55 + volatilite normalize olunca tekrar bak."
    else:
        if breakout and not extended:
            entry_label = "ALINABİLİR (Kırılım + Hacim)"
            buy_zone_text = f"Kırılım bölgesi: ~{close:.2f} ± %1.5 (hacim onayı sürerse)."
        elif extended:
            entry_label = "BEKLE (Geri çekilme)"
            buy_zone_text = f"Uzamış (EMA50 uzaklık ~%{dist_ema50_pct:.1f}). EMA20–EMA50 bandına geri çekilme beklenebilir."
        else:
            entry_label = "BEKLE / İZLE"
            buy_zone_text = "EMA20–EMA50 bandı veya önceki pivot üzerinde güç gösterimi arayın."

    trend_text = (
        "güçlü" if trend_stack and (price_above_ema150 or price_borderline_ema150)
        else ("zayıf" if price_below_ema200 else "karışık")
    )
    mom_text = "sağlıklı" if 55 <= rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_borderline else "yüksek")

    narrative = (
        f"**Trend:** {trend_text}  \n"
        f"Close={close:.2f}, EMA50={ema50:.2f}, EMA150={ema150:.2f}, EMA200={ema200:.2f}  \n\n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}  \n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}  \n\n"
        f"**Stratejiye Uygunluk:** **{fit_label}**  \n"
        f"**Zamanlama:** **{entry_label}**  \n"
        f"**Alım Bölgesi:** {buy_zone_text}"
    )

    debug = {
        "trend_stack": trend_stack,
        "dist_ema150_pct": dist_ema150_pct,
        "ema200_slope": ema200_slope,
        "rsi14": rsi14,
        "atr_pct": atr_pct,
        "breakout_heuristic": breakout,
        "extended_vs_ema50": dist_ema50_pct,
        "core_ok_count": core_ok,
    }

    return StrategyFitResult(fit_label, entry_label, buy_zone_text, narrative, debug)


# =========================================================
# DATA LOADING / PARSING
# =========================================================
INTERVAL_MAP = {
    "Günlük (1day)": "1day",
    "Saatlik (1h)": "1h",
    "15 Dakika (15min)": "15min",
}

@st.cache_data(ttl=60)  # aynı sembol/interval tekrarında krediyi yememek için
def fetch_ohlcv(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    data = client.time_series(symbol=symbol, interval=interval, outputsize=bars)

    # Twelve Data error format:
    # {"code": 400, "message": "...", "status": "error"}
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')} (code={data.get('code')})")

    values = data.get("values")
    if not values:
        raise RuntimeError("TwelveData: values boş döndü (sembol/interval desteklenmiyor olabilir).")

    df = pd.DataFrame(values)

    # Normalize columns
    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")

    # If volume is missing for some instruments, fill with 0
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)

    return df


@st.cache_data(ttl=60)
def fetch_quote(symbol: str) -> dict:
    q = client.quote(symbol)
    if isinstance(q, dict) and q.get("status") == "error":
        return {}
    return q


# =========================================================
# PLOTTING
# =========================================================
def plot_chart(df: pd.DataFrame, symbol: str, last_price: float | None):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema150"], name="EMA150", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", mode="lines"))

    if last_price is not None and np.isfinite(last_price):
        fig.add_hline(y=float(last_price), line_dash="dot")
        fig.add_annotation(
            x=df["time"].iloc[-1],
            y=float(last_price),
            text=f"Last ~ {float(last_price):.2f}",
            showarrow=True,
            arrowhead=2,
        )

    fig.update_layout(
        title=f"{symbol} — Candlestick + EMA'lar",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =========================================================
# UI
# =========================================================
with st.sidebar:
    st.header("Kontroller")
    interval_label = st.selectbox("Zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=0)
    bars = st.slider("Bar sayısı", min_value=120, max_value=800, value=300, step=10)
    st.caption("Not: Free planda dakika limiti var (8/dk). Aynı sembolde tekrarlar cache ile hafifler.")

left, right = st.columns([0.36, 0.64], vertical_alignment="top")

with left:
    st.subheader("Hisse")
    symbol_in = st.text_input("Ticker", placeholder="Örn: NVDA, TSLA, PLTR").strip()
    run = st.button("Getir & Analiz Et", type="primary", use_container_width=True)

    if run:
        if not symbol_in:
            st.warning("Lütfen ticker gir.")
        else:
            symbol = symbol_in.upper()
            interval = INTERVAL_MAP[interval_label]

            with st.spinner("Twelve Data'dan veri çekiliyor..."):
                try:
                    df = fetch_ohlcv(symbol, interval, bars)
                except Exception as e:
                    st.error(f"Veri alınamadı: {e}")
                    st.stop()

            # indicators
            df["ema50"] = ema(df["close"], 50)
            df["ema150"] = ema(df["close"], 150)
            df["ema200"] = ema(df["close"], 200)
            df["rsi14"] = rsi(df["close"], 14)
            df["atr14"] = atr(df, 14)

            res = evaluate(df)

            # Quote (optional)
            q = {}
            try:
                q = fetch_quote(symbol)
            except Exception:
                q = {}

            last_price = None
            if isinstance(q, dict):
                # TwelveData quote fields can include: close, price, previous_close etc.
                for key in ["close", "price"]:
                    if key in q:
                        try:
                            last_price = float(q[key])
                            break
                        except Exception:
                            pass

            st.divider()
            st.markdown("### Sonuç")
            st.metric("Stratejiye Uygunluk", res.fit_label)
            st.metric("Zamanlama", res.entry_label)
            st.info(res.buy_zone_text)

            if q:
                st.markdown("### Anlık Özet (Quote)")
                # güvenli subset
                show = {}
                for k in ["symbol", "name", "exchange", "currency", "close", "price", "change", "percent_change", "previous_close"]:
                    if k in q:
                        show[k] = q[k]
                st.write(show)

            st.session_state["__df"] = df
            st.session_state["__symbol"] = symbol
            st.session_state["__last"] = last_price
            st.session_state["__res"] = res

with right:
    st.subheader("Grafik & Yorum")

    if "__df" not in st.session_state:
        st.write("Soldan bir ticker girip **Getir & Analiz Et** ile başlayın.")
    else:
        df = st.session_state["__df"]
        symbol = st.session_state["__symbol"]
        last_price = st.session_state["__last"]
        res = st.session_state["__res"]

        fig = plot_chart(df, symbol, last_price)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Otomatik Teknik Yorum")
        st.markdown(res.narrative)

        with st.expander("Detay (debug)"):
            st.json(res.debug)
