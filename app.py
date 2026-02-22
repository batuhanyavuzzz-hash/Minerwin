import time
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Tek Hisse Teknik Analiz (Finnhub)", layout="wide")

# -----------------------------
# Finnhub client (minimal)
# -----------------------------
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
        return self._get("/stock/candle", {
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": _to
        })


# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def slope(series: pd.Series, lookback: int = 20) -> float:
    s = series.dropna()
    if len(s) < lookback + 2:
        return np.nan
    y = s.iloc[-lookback:].values
    x = np.arange(len(y))
    # simple linear regression slope
    a = np.polyfit(x, y, 1)[0]
    return float(a)

# -----------------------------
# Evaluation logic (your V1 thresholds)
# -----------------------------
@dataclass
class StrategyFitResult:
    fit_label: str               # UYGUN / SINIRDA / UYGUN DEĞİL
    fit_checks: dict             # booleans + values
    entry_label: str             # ALINABİLİR / BEKLE (Geri çekilme) / BEKLE (Kırılım) / ŞU AN UYGUN DEĞİL
    buy_zone_text: str           # Suggested buy zone text
    narrative: str               # Human-readable commentary

def evaluate(df: pd.DataFrame) -> StrategyFitResult:
    # df must include ema50/150/200, rsi14, atr14
    last = df.iloc[-1]
    close = float(last["close"])
    ema50 = float(last["ema50"])
    ema150 = float(last["ema150"])
    ema200 = float(last["ema200"])
    rsi14 = float(last["rsi14"])
    atr14 = float(last["atr14"])
    atr_pct = (atr14 / close) * 100 if close else np.nan

    # Trend stack + price location
    trend_stack = (ema50 > ema150 > ema200)
    # borderline tolerance for EMA150: allow up to -2%
    dist_ema150_pct = ((close - ema150) / ema150) * 100 if ema150 else np.nan
    price_above_ema150 = close > ema150
    price_borderline = (-2.0 <= dist_ema150_pct <= 0.0)

    ema200_slope = slope(df["ema200"], lookback=20)
    long_trend_ok = (ema200_slope > 0)

    # Momentum thresholds
    momentum_ok = (rsi14 >= 55)
    momentum_borderline = (50 <= rsi14 < 55)

    # Volatility thresholds
    vol_ok = (2.0 <= atr_pct <= 6.0)
    vol_borderline = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 9.0)
    vol_fail = (atr_pct > 9.0)

    # Hard fail
    price_below_ema200 = close < ema200

    checks = {
        "trend_stack": trend_stack,
        "price_above_ema150": price_above_ema150,
        "price_borderline_to_ema150": price_borderline,
        "ema200_slope": ema200_slope,
        "long_trend_ok": long_trend_ok,
        "rsi14": rsi14,
        "momentum_ok": momentum_ok,
        "momentum_borderline": momentum_borderline,
        "atr_pct": atr_pct,
        "vol_ok": vol_ok,
        "vol_borderline": vol_borderline,
        "price_below_ema200": price_below_ema200,
        "dist_ema150_pct": dist_ema150_pct,
    }

    # Fit score via 5 core booleans
    core_ok = 0
    core_ok += int(trend_stack)
    core_ok += int(price_above_ema150 or price_borderline)
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

    # Entry / Buy zone logic (simple V1, yorumlayan)
    # distance to EMA50 to detect "extended"
    dist_ema50_pct = ((close - ema50) / ema50) * 100 if ema50 else np.nan

    # A) If fit not good: still propose conditional zones
    # B) If extended > 8% from EMA50 => wait pullback
    extended = dist_ema50_pct > 8.0

    # Simple "breakout" heuristic: close at 20-day high and volume above 20-day avg * 1.5
    # (You can refine later.)
    lookback = 20
    if len(df) >= lookback + 5:
        hh20 = df["high"].iloc[-lookback:].max()
        vol_sma20 = df["volume"].iloc[-lookback:].mean()
        breakout = (close >= hh20 * 0.995) and (float(last["volume"]) >= 1.5 * vol_sma20)
    else:
        breakout = False

    if fit_label == "UYGUN DEĞİL":
        entry_label = "ŞU AN UYGUN DEĞİL"
        buy_zone_text = "Strateji dışı görünüyor. Trend düzeldikten sonra EMA150/EMA200 üzeri kabul ve RSI≥55 koşullarıyla yeniden değerlendirin."
    else:
        if breakout and not extended:
            entry_label = "ALINABİLİR (Kırılım + Hacim)"
            # approximate pivot zone around close
            buy_zone_text = f"Kırılım seviyesi çevresi: ~{close:.2f} ± %1.5 (hacim onayı sürerse)."
        elif extended:
            entry_label = "BEKLE (Geri çekilme)"
            buy_zone_text = f"Uzamış görünüyor (EMA50'ye uzaklık ~%{dist_ema50_pct:.1f}). EMA20–EMA50 bandına geri çekilme izlenebilir."
        else:
            entry_label = "BEKLE (Geri çekilme / Konsolidasyon)"
            buy_zone_text = "EMA20–EMA50 bandı veya önceki pivot bölgesi üzerinde güç gösterimi ile takip."

    # Narrative (conditional text)
    trend_text = "güçlü" if trend_stack and (price_above_ema150 or price_borderline) else ("zayıf" if price_below_ema200 else "karışık")
    mom_text = "sağlıklı" if rsi14 >= 55 and rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_borderline else "yüksek")

    narrative = (
        f"**Trend:** {trend_text}. Close={close:.2f}, EMA50={ema50:.2f}, EMA150={ema150:.2f}, EMA200={ema200:.2f}.\n\n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}.\n\n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}.\n\n"
        f"**Durum:** {fit_label} | **Zamanlama:** {entry_label}\n\n"
        f"**Alım Bölgesi:** {buy_zone_text}"
    )

    return StrategyFitResult(
        fit_label=fit_label,
        fit_checks=checks,
        entry_label=entry_label,
        buy_zone_text=buy_zone_text,
        narrative=narrative
    )

# -----------------------------
# Plot
# -----------------------------
def plot_chart(df: pd.DataFrame, symbol: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema150"], name="EMA150", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", mode="lines"))

    fig.update_layout(
        title=f"{symbol} — Candlestick + EMA'lar",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    return fig

# -----------------------------
# UI state helpers
# -----------------------------
def reset_state():
    st.session_state["symbol"] = ""
    st.session_state["df"] = None
    st.session_state["result"] = None
    st.session_state["quote"] = None

# init state
for k, v in {"symbol": "", "df": None, "result": None, "quote": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# UI
# -----------------------------
st.title("Tek Hisse Teknik Analiz — Finnhub (V1)")

with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("Finnhub API Key", type="password", help="API key'i burada saklıyoruz (local).")
    resolution = st.selectbox("Zaman çözünürlüğü", ["D", "60"], index=0, help="D = günlük, 60 = saatlik")
    bars = st.slider("Bar sayısı", min_value=120, max_value=600, value=300, step=10)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Temizle", use_container_width=True):
            reset_state()
            st.rerun()
    with colB:
        st.caption("V1: EMA/RSI/ATR + durum bazlı yorum")

left, right = st.columns([0.36, 0.64], vertical_alignment="top")

with left:
    st.subheader("Hisse Seç")
    symbol = st.text_input("Ticker", value=st.session_state["symbol"], placeholder="Örn: NVDA", help="Büyük harf önerilir.")
    analyze_btn = st.button("Getir & Analiz Et", type="primary", use_container_width=True, disabled=not (api_key and symbol))

    if analyze_btn:
        symbol = symbol.strip().upper()
        st.session_state["symbol"] = symbol

        client = FinnhubClient(api_key)

        now = int(time.time())
        # +1 day buffer for daily; for intraday we also use buffer
        _to = now
        # approx: daily bars -> bars*1.6 days back; hourly -> bars*2 hours back
        if resolution == "D":
            _from = now - int(bars * 1.6 * 24 * 3600)
        else:
            _from = now - int(bars * 2.2 * 3600)

        with st.spinner("Finnhub'tan veri çekiliyor..."):
            quote = client.quote(symbol)
            candles = client.candles(symbol, resolution, _from, _to)

        if candles.get("s") != "ok":
            st.error("Veri alınamadı. (Finnhub candle yanıtı 'ok' değil.) Ticker/borsa/plan kaynaklı olabilir.")
            st.session_state["df"] = None
            st.session_state["result"] = None
            st.session_state["quote"] = quote
        else:
            df = pd.DataFrame({
                "time": pd.to_datetime(candles["t"], unit="s"),
                "open": candles["o"],
                "high": candles["h"],
                "low": candles["l"],
                "close": candles["c"],
                "volume": candles["v"],
            }).sort_values("time")

            # indicators
            df["ema50"] = ema(df["close"], 50)
            df["ema150"] = ema(df["close"], 150)
            df["ema200"] = ema(df["close"], 200)
            df["rsi14"] = rsi(df["close"], 14)
            df["atr14"] = atr(df, 14)

            # evaluate
            result = evaluate(df)

            st.session_state["df"] = df
            st.session_state["result"] = result
            st.session_state["quote"] = quote

    st.divider()

    if st.session_state["quote"]:
        q = st.session_state["quote"]
        # Finnhub quote fields: c=current, d=change, dp=percent change, h/l/o/pc, t=timestamp
        ts = q.get("t")
        ts_text = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
        st.markdown("### Anlık Özet (Quote)")
        st.write({
            "Son Fiyat (c)": q.get("c"),
            "Günlük Değişim (d)": q.get("d"),
            "Günlük % (dp)": q.get("dp"),
            "Gün İçi Yüksek (h)": q.get("h"),
            "Gün İçi Düşük (l)": q.get("l"),
            "Açılış (o)": q.get("o"),
            "Önceki Kapanış (pc)": q.get("pc"),
            "Zaman": ts_text,
        })

    if st.session_state["result"]:
        res = st.session_state["result"]
        st.markdown("### Sonuç")
        st.metric("Stratejiye Uygunluk", res.fit_label)
        st.metric("Zamanlama", res.entry_label)
        st.info(res.buy_zone_text)

with right:
    st.subheader("Grafik & Analiz")

    if st.session_state["df"] is None:
        st.write("Sol taraftan bir ticker girip **Getir & Analiz Et** ile başlayın.")
    else:
        df = st.session_state["df"]
        symbol = st.session_state["symbol"]
        fig = plot_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Otomatik Teknik Yorum")
        st.markdown(st.session_state["result"].narrative)

        with st.expander("Detay Kontroller (debug)"):
            st.json(st.session_state["result"].fit_checks)
