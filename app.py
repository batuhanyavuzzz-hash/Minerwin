import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# ENV + APP CONFIG
# =========================================================
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

st.set_page_config(page_title="Tek Hisse Teknik Analiz (Finnhub)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Finnhub (V1)")

if not FINNHUB_API_KEY:
    st.error("FINNHUB_API_KEY bulunamadı. app.py ile aynı klasöre .env koyup FINNHUB_API_KEY=... yazın.")
    st.stop()


# =========================================================
# FINNHUB CLIENT
# =========================================================
class FinnhubClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://finnhub.io/api/v1"

    def _get(self, path: str, params: dict):
        params = dict(params or {})
        params["token"] = self.api_key
        r = requests.get(f"{self.base}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def quote(self, symbol: str) -> dict:
        return self._get("/quote", {"symbol": symbol})

    def candles(self, symbol: str, resolution: str, _from: int, _to: int) -> dict:
        # resolution: "D" (daily), "60" (1h) etc.
        return self._get(
            "/stock/candle",
            {"symbol": symbol, "resolution": resolution, "from": _from, "to": _to},
        )


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
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
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
# EVALUATION (YOUR V1 THRESHOLDS)
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

    # --- Trend checks
    trend_stack = (ema50 > ema150 > ema200)
    dist_ema150_pct = ((close - ema150) / ema150) * 100 if ema150 else float("nan")
    price_above_ema150 = close > ema150
    price_borderline_ema150 = (-2.0 <= dist_ema150_pct <= 0.0)

    ema200_slope = slope(df["ema200"], lookback=20)
    long_trend_ok = (ema200_slope > 0)

    # --- Momentum
    momentum_ok = (rsi14 >= 55)
    momentum_borderline = (50 <= rsi14 < 55)

    # --- Volatility
    vol_ok = (2.0 <= atr_pct <= 6.0)
    vol_borderline = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 9.0)
    vol_fail = (atr_pct > 9.0)

    # --- Hard fail
    price_below_ema200 = close < ema200

    # Core fit score (5 items)
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

    # --- Entry / Buy zone (V1)
    dist_ema50_pct = ((close - ema50) / ema50) * 100 if ema50 else float("nan")
    extended = (dist_ema50_pct > 8.0)

    # Simple breakout heuristic: close at ~20d high + volume >= 1.5x vol_sma20
    lookback = 20
    breakout = False
    if len(df) >= lookback + 5:
        hh20 = df["high"].iloc[-lookback:].max()
        vol_sma20 = df["volume"].iloc[-lookback:].mean()
        breakout = (close >= hh20 * 0.995) and (float(last["volume"]) >= 1.5 * vol_sma20)

    if fit_label == "UYGUN DEĞİL":
        entry_label = "ŞU AN UYGUN DEĞİL"
        buy_zone_text = (
            "Strateji dışı görünüyor. Trend için EMA150/EMA200 üzeri kabul ve RSI≥55 koşullarıyla yeniden değerlendirin."
        )
    else:
        if breakout and not extended:
            entry_label = "ALINABİLİR (Kırılım + Hacim)"
            buy_zone_text = f"Kırılım seviyesi çevresi: ~{close:.2f} ± %1.5 (hacim onayı sürerse)."
        elif extended:
            entry_label = "BEKLE (Geri çekilme)"
            buy_zone_text = f"Uzamış görünüyor (EMA50'ye uzaklık ~%{dist_ema50_pct:.1f}). EMA20–EMA50 bandına geri çekilme izlenebilir."
        else:
            entry_label = "BEKLE (Geri çekilme / Konsolidasyon)"
            buy_zone_text = "EMA20–EMA50 bandı veya önceki pivot bölgesi üzerinde güç gösterimi ile takip."

    # Narrative
    trend_text = (
        "güçlü" if trend_stack and (price_above_ema150 or price_borderline_ema150)
        else ("zayıf" if price_below_ema200 else "karışık")
    )
    mom_text = (
        "sağlıklı" if 55 <= rsi14 <= 75
        else ("ısınmış" if rsi14 > 75 else "zayıf")
    )
    vol_text = "uygun" if vol_ok else ("agresif" if vol_borderline else "yüksek")

    narrative = (
        f"**Trend:** {trend_text}. Close={close:.2f}, EMA50={ema50:.2f}, EMA150={ema150:.2f}, EMA200={ema200:.2f}\n\n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}\n\n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}\n\n"
        f"**Stratejiye Uygunluk:** **{fit_label}**\n\n"
        f"**Zamanlama:** **{entry_label}**\n\n"
        f"**Alım Bölgesi:** {buy_zone_text}"
    )

    debug = {
        "trend_stack": trend_stack,
        "price_above_ema150": price_above_ema150,
        "price_borderline_ema150": price_borderline_ema150,
        "dist_ema150_pct": dist_ema150_pct,
        "ema200_slope": ema200_slope,
        "long_trend_ok": long_trend_ok,
        "rsi14": rsi14,
        "momentum_ok": momentum_ok,
        "momentum_borderline": momentum_borderline,
        "atr_pct": atr_pct,
        "vol_ok": vol_ok,
        "vol_borderline": vol_borderline,
        "vol_fail": vol_fail,
        "price_below_ema200": price_below_ema200,
        "dist_ema50_pct": dist_ema50_pct,
        "breakout_heuristic": breakout,
        "core_ok_count": core_ok,
    }

    return StrategyFitResult(
        fit_label=fit_label,
        entry_label=entry_label,
        buy_zone_text=buy_zone_text,
        narrative=narrative,
        debug=debug,
    )


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
        fig.add_hline(y=last_price, line_dash="dot")
        fig.add_annotation(
            x=df["time"].iloc[-1],
            y=last_price,
            text=f"Last ~ {last_price:.2f}",
            showarrow=True,
            arrowhead=2,
        )

    fig.update_layout(
        title=f"{symbol} — Candlestick + EMA'lar",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    return fig


# =========================================================
# STATE MGMT
# =========================================================
def reset_state():
    st.session_state["symbol"] = ""
    st.session_state["df"] = None
    st.session_state["result"] = None
    st.session_state["quote"] = None


for k, v in {"symbol": "", "df": None, "result": None, "quote": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# UI
# =========================================================
client = FinnhubClient(FINNHUB_API_KEY)

with st.sidebar:
    st.header("Kontroller")
    resolution = st.selectbox("Zaman çözünürlüğü", ["D", "60"], index=0, help="D=günlük (önerilen), 60=saatlik")
    bars = st.slider("Bar sayısı", min_value=120, max_value=600, value=300, step=10)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Temizle", use_container_width=True):
            reset_state()
            st.rerun()
    with c2:
        st.caption("V1: EMA/RSI/ATR + durum bazlı yorum")

left, right = st.columns([0.36, 0.64], vertical_alignment="top")

with left:
    st.subheader("Hisse")
    symbol_in = st.text_input(
        "Ticker",
        value=st.session_state["symbol"],
        placeholder="Örn: NVDA",
        help="Büyük harf önerilir.",
    )
    run = st.button("Getir & Analiz Et", type="primary", use_container_width=True)

    if run:
        symbol = symbol_in.strip().upper()
        st.session_state["symbol"] = symbol

        now = int(time.time())
        _to = now
        if resolution == "D":
            _from = now - int(bars * 1.6 * 24 * 3600)
        else:
            _from = now - int(bars * 2.2 * 3600)

        with st.spinner("Finnhub'tan veri çekiliyor..."):
            try:
                quote = client.quote(symbol)
                candles = client.candles(symbol, resolution, _from, _to)
            except requests.HTTPError as e:
                st.error(f"HTTP hata: {e}")
                st.session_state["df"] = None
                st.session_state["result"] = None
                st.session_state["quote"] = None
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")
                st.session_state["df"] = None
                st.session_state["result"] = None
                st.session_state["quote"] = None
            else:
                st.session_state["quote"] = quote

                if candles.get("s") != "ok":
                    st.error("Candle verisi alınamadı. (Ticker/borsa/plan kaynaklı olabilir.)")
                    st.session_state["df"] = None
                    st.session_state["result"] = None
                else:
                    df = pd.DataFrame(
                        {
                            "time": pd.to_datetime(candles["t"], unit="s"),
                            "open": candles["o"],
                            "high": candles["h"],
                            "low": candles["l"],
                            "close": candles["c"],
                            "volume": candles["v"],
                        }
                    ).sort_values("time")

                    # indicators
                    df["ema50"] = ema(df["close"], 50)
                    df["ema150"] = ema(df["close"], 150)
                    df["ema200"] = ema(df["close"], 200)
                    df["rsi14"] = rsi(df["close"], 14)
                    df["atr14"] = atr(df, 14)

                    res = evaluate(df)
                    st.session_state["df"] = df
                    st.session_state["result"] = res

    st.divider()

    if st.session_state["quote"]:
        q = st.session_state["quote"]
        ts = q.get("t")
        ts_text = (
            datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if ts
            else "—"
        )
        st.markdown("### Anlık Özet (Quote)")
        st.write(
            {
                "Son Fiyat (c)": q.get("c"),
                "Günlük Değişim (d)": q.get("d"),
                "Günlük % (dp)": q.get("dp"),
                "Gün İçi Yüksek (h)": q.get("h"),
                "Gün İçi Düşük (l)": q.get("l"),
                "Açılış (o)": q.get("o"),
                "Önceki Kapanış (pc)": q.get("pc"),
                "Zaman": ts_text,
            }
        )

    if st.session_state["result"]:
        res = st.session_state["result"]
        st.markdown("### Sonuç")
        st.metric("Stratejiye Uygunluk", res.fit_label)
        st.metric("Zamanlama", res.entry_label)
        st.info(res.buy_zone_text)

with right:
    st.subheader("Grafik & Yorum")

    if st.session_state["df"] is None:
        st.write("Soldan bir ticker girip **Getir & Analiz Et** ile başlayın.")
    else:
        df = st.session_state["df"]
        symbol = st.session_state["symbol"]
        quote = st.session_state["quote"] or {}
        last_price = quote.get("c", None)

        fig = plot_chart(df, symbol, last_price)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Otomatik Teknik Yorum")
        st.markdown(st.session_state["result"].narrative)

        with st.expander("Detay Kontroller (debug)"):
            st.json(st.session_state["result"].debug)
