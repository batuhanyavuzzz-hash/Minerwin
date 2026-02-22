import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dataclasses import dataclass

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Tek Hisse Teknik Analiz (V3)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Twelve Data (V3)")

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error('TWELVEDATA_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine ekle: TWELVEDATA_API_KEY="..."')
    st.stop()

BASE_URL = "https://api.twelvedata.com"

INTERVAL_MAP = {
    "Günlük (1day)": "1day",
    "Saatlik (1h)": "1h",
    "15 Dakika (15min)": "15min",
}


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
# DATA (Twelve Data)
# =========================================================
@st.cache_data(ttl=60)
def td_time_series(symbol: str, interval: str, outputsize: int) -> dict:
    r = requests.get(
        f"{BASE_URL}/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": API_KEY,
            "format": "JSON",
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def td_quote(symbol: str) -> dict:
    r = requests.get(
        f"{BASE_URL}/quote",
        params={"symbol": symbol, "apikey": API_KEY, "format": "JSON"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def parse_ohlcv(payload: dict) -> pd.DataFrame:
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"TwelveData: {payload.get('message')} (code={payload.get('code')})")

    values = payload.get("values")
    if not values:
        raise RuntimeError("TwelveData: 'values' boş döndü (ticker/interval desteklenmiyor olabilir).")

    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        raise RuntimeError("TwelveData: datetime alanı yok (beklenmeyen format).")

    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"TwelveData: {col} alanı yok.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df


# =========================================================
# STRATEGY / SCORING / TRADE PLAN
# =========================================================
@dataclass
class ScoreBreakdown:
    trend_stack: int
    price_vs_ema150: int
    momentum_rsi: int
    volatility_atr: int
    extension_vs_ema50: int


@dataclass
class TradePlan:
    score: int
    label: str
    entry_low: float
    entry_high: float
    entry_mid: float
    stop: float
    tp1: float
    rr: float
    narrative: str
    debug: dict
    breakdown: ScoreBreakdown


def label_from_score(score: int) -> str:
    if score >= 75:
        return "UYGUN"
    if score >= 60:
        return "SINIRDA"
    return "UYGUN DEĞİL"


def build_trade_plan(df: pd.DataFrame) -> TradePlan:
    last = df.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema150 = float(last["ema150"])
    ema200 = float(last["ema200"])
    rsi14 = float(last["rsi14"])
    atr14 = float(last["atr14"])

    atr_pct = (atr14 / close) * 100 if close else float("nan")
    dist_ema50_pct = ((close - ema50) / ema50) * 100 if ema50 else float("nan")
    dist_ema150_pct = ((close - ema150) / ema150) * 100 if ema150 else float("nan")

    trend_stack_ok = (ema50 > ema150 > ema200)
    ema200_slope = slope(df["ema200"], lookback=20)
    long_trend_ok = (ema200_slope > 0)

    momentum_ok = (rsi14 >= 55)
    momentum_border = (50 <= rsi14 < 55)

    vol_ok = (2.0 <= atr_pct <= 6.0)
    vol_border = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 8.0)
    vol_fail = (atr_pct > 8.0)

    price_above_ema150 = close >= ema150
    price_near_ema150 = close >= ema150 * 0.98

    extended = dist_ema50_pct > 8.0

    score = 0

    trend_pts = 30 if (trend_stack_ok and long_trend_ok) else (20 if trend_stack_ok else (10 if long_trend_ok else 0))
    score += trend_pts

    p_pts = 20 if price_above_ema150 else (10 if price_near_ema150 else 0)
    score += p_pts

    m_pts = 20 if momentum_ok else (10 if momentum_border else 0)
    score += m_pts

    v_pts = 15 if vol_ok else (7 if vol_border else 0)
    score += v_pts

    e_pts = 15 if not extended else 0
    score += e_pts

    label = label_from_score(score)

    breakdown = ScoreBreakdown(
        trend_stack=trend_pts,
        price_vs_ema150=p_pts,
        momentum_rsi=m_pts,
        volatility_atr=v_pts,
        extension_vs_ema50=e_pts,
    )

    # ENTRY ZONE
    entry_low = float(min(ema20, ema50))
    entry_high = float(max(ema20, ema50))
    entry_mid = (entry_low + entry_high) / 2.0

    # Breakout heuristic
    lookback = 20
    breakout = False
    if len(df) >= lookback + 5:
        hh20 = float(df["high"].iloc[-lookback:].max())
        vol_sma20 = float(df["volume"].iloc[-lookback:].mean())
        breakout = (close >= hh20 * 0.995) and (float(last["volume"]) >= 1.5 * vol_sma20)

    if breakout and not extended:
        entry_low = close * 0.985
        entry_high = close * 1.015
        entry_mid = (entry_low + entry_high) / 2.0

    # STOP LOGIC (C): tighter = higher stop
    stop_ema = ema50 * 0.995
    stop_atr = entry_mid - 1.2 * atr14
    stop = float(max(stop_ema, stop_atr))

    if stop >= entry_mid:
        stop = float(entry_mid * 0.99)

    # TP1 (2R)
    risk = entry_mid - stop
    tp1 = float(entry_mid + 2.0 * risk) if risk > 0 else float(entry_mid * 1.02)
    rr = (tp1 - entry_mid) / risk if risk > 0 else float("nan")

    trend_text = (
        "güçlü" if (trend_stack_ok and (price_above_ema150 or price_near_ema150))
        else ("zayıf" if close < ema200 else "karışık")
    )
    mom_text = "sağlıklı" if 55 <= rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf/sınır")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_border else "yüksek")

    timing = (
        "ALIM ARANIR" if (label == "UYGUN" and not extended and (momentum_ok or breakout))
        else ("BEKLE / İZLE" if extended or label == "SINIRDA" else "UZAK DUR / ŞARTLAR OLUŞSUN")
    )

    entry_zone_text = f"{entry_low:.2f} – {entry_high:.2f}"
    narrative = (
        f"**Skor:** {score}/100 → **{label}**  \n\n"
        f"**Güncel (Candle) Fiyat:** {close:.2f}  \n"
        f"EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | EMA150: {ema150:.2f} | EMA200: {ema200:.2f}  \n\n"
        f"**Trend:** {trend_text} (EMA200 eğim={ema200_slope:.4f})  \n"
        f"**Fiyat Konumu:** EMA150 uzaklık %{dist_ema150_pct:.2f}  \n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}  \n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}  \n"
        f"**Uzama (EMA50 mesafe):** %{dist_ema50_pct:.2f} → {'uzamış' if extended else 'normal'}  \n\n"
        f"**Zamanlama:** **{timing}**  \n"
        f"**Giriş Bölgesi:** {entry_zone_text}  \n"
        f"**Stop:** {stop:.2f} (EMA50 vs ATR → sıkı olan)  \n"
        f"**TP1:** {tp1:.2f} (2R)  \n"
        f"**R/R:** 1 : {rr:.2f}"
    )

    debug = {
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema150": ema150,
        "ema200": ema200,
        "rsi14": rsi14,
        "atr14": atr14,
        "atr_pct": atr_pct,
        "dist_ema50_pct": dist_ema50_pct,
        "dist_ema150_pct": dist_ema150_pct,
        "trend_stack_ok": trend_stack_ok,
        "ema200_slope": ema200_slope,
        "long_trend_ok": long_trend_ok,
        "momentum_ok": momentum_ok,
        "momentum_border": momentum_border,
        "vol_ok": vol_ok,
        "vol_border": vol_border,
        "vol_fail": vol_fail,
        "price_above_ema150": price_above_ema150,
        "price_near_ema150": price_near_ema150,
        "extended": extended,
        "breakout_heuristic": breakout,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_mid": entry_mid,
        "stop_ema": stop_ema,
        "stop_atr": stop_atr,
        "stop": stop,
        "tp1": tp1,
        "rr": rr,
    }

    return TradePlan(
        score=score,
        label=label,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_mid=entry_mid,
        stop=stop,
        tp1=tp1,
        rr=rr,
        narrative=narrative,
        debug=debug,
        breakdown=breakdown,
    )


# =========================================================
# PDF EXPORT
# =========================================================
def _wrap_lines(text: str, max_chars: int = 92):
    """Simple char-based wrap for ReportLab (keeps it dependency-free)."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        while len(line) > max_chars:
            out.append(line[:max_chars])
            line = line[max_chars:]
        out.append(line)
    return out


def build_pdf_bytes(
    ticker: str,
    interval_label: str,
    bars: int,
    plan: TradePlan,
    quote: dict | None,
):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    x = 2.0 * cm
    y = h - 2.0 * cm
    lh = 14  # line height

    def draw_line(txt, font="Helvetica", size=11, space=lh):
        nonlocal y
        c.setFont(font, size)
        c.drawString(x, y, txt)
        y -= space
        if y < 2.0 * cm:
            c.showPage()
            y = h - 2.0 * cm

    # Header
    draw_line(f"Tek Hisse Teknik Analiz Raporu", font="Helvetica-Bold", size=16, space=18)
    draw_line(f"Ticker: {ticker}    Zaman: {interval_label}    Bar: {bars}", font="Helvetica", size=11)
    draw_line(f"Tarih: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", font="Helvetica", size=10)
    draw_line("", size=10, space=10)

    # Summary
    draw_line("Özet", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"Skor: {plan.score}/100    Durum: {plan.label}", size=11)
    draw_line(f"Giriş Bölgesi: {plan.entry_low:.2f} – {plan.entry_high:.2f}", size=11)
    draw_line(f"Stop: {plan.stop:.2f}", size=11)
    draw_line(f"TP1: {plan.tp1:.2f}", size=11)
    rr_text = f"1 : {plan.rr:.2f}" if np.isfinite(plan.rr) else "—"
    draw_line(f"Risk/Reward: {rr_text}", size=11)
    draw_line("", size=10, space=10)

    # Quote
    if quote and isinstance(quote, dict):
        draw_line("Quote (Anlık Özet)", font="Helvetica-Bold", size=13, space=16)
        for k in ["name", "exchange", "currency", "close", "price", "change", "percent_change", "previous_close"]:
            if k in quote:
                draw_line(f"{k}: {quote[k]}", size=10, space=12)
        draw_line("", size=10, space=10)

    # Narrative (markdown-like -> plain)
    draw_line("Otomatik Teknik Yorum", font="Helvetica-Bold", size=13, space=16)
    plain = (
        plan.narrative
        .replace("**", "")
        .replace("  \n", "\n")
    )
    for ln in _wrap_lines(plain, max_chars=95):
        draw_line(ln, size=10, space=12)

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
# PLOTTING
# =========================================================
def plot_chart(df: pd.DataFrame, symbol: str, plan: TradePlan, last_price: float | None):
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

    fig.add_trace(go.Scatter(x=df["time"], y=df["ema20"], name="EMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema150"], name="EMA150", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", mode="lines"))

    fig.add_hrect(
        y0=plan.entry_low,
        y1=plan.entry_high,
        opacity=0.15,
        line_width=0,
        annotation_text="ENTRY ZONE",
        annotation_position="top left",
    )

    fig.add_hline(y=plan.stop, line_dash="dash", annotation_text="STOP", annotation_position="bottom left")
    fig.add_hline(y=plan.tp1, line_dash="dash", annotation_text="TP1", annotation_position="top left")

    if last_price is not None and np.isfinite(last_price):
        fig.add_hline(y=float(last_price), line_dash="dot", annotation_text="LAST", annotation_position="top right")

    fig.update_layout(
        title=f"{symbol} — Candlestick + EMA'lar + Trade Levels",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=680,
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
    show_quote = st.checkbox("Quote paneli (anlık özet)", value=True)
    st.caption("Free planda 8 istek/dk. Cache (60 sn) aktif.")

left, right = st.columns([0.36, 0.64], vertical_alignment="top")

with left:
    st.subheader("Hisse")
    ticker = st.text_input("Ticker", placeholder="Örn: NVDA, TSLA, PLTR").strip().upper()
    run = st.button("Getir & Analiz Et", type="primary", use_container_width=True)

    if run:
        if not ticker:
            st.warning("Ticker gir.")
        else:
            interval = INTERVAL_MAP[interval_label]
            with st.spinner("Twelve Data'dan veri çekiliyor..."):
                try:
                    payload = td_time_series(ticker, interval, bars)
                    df = parse_ohlcv(payload)
                except Exception as e:
                    st.error(f"Veri alınamadı: {e}")
                    st.stop()

            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["ema150"] = ema(df["close"], 150)
            df["ema200"] = ema(df["close"], 200)
            df["rsi14"] = rsi(df["close"], 14)
            df["atr14"] = atr(df, 14)

            plan = build_trade_plan(df)

            q = {}
            last_price = None
            if show_quote:
                try:
                    q = td_quote(ticker)
                    for key in ["close", "price"]:
                        if key in q:
                            try:
                                last_price = float(q[key])
                                break
                            except Exception:
                                pass
                except Exception:
                    q = {}

            st.session_state["__df"] = df
            st.session_state["__ticker"] = ticker
            st.session_state["__plan"] = plan
            st.session_state["__quote"] = q
            st.session_state["__last_price"] = last_price
            st.session_state["__interval_label"] = interval_label
            st.session_state["__bars"] = bars

            st.divider()
            st.subheader("📊 Strateji Özeti")
            st.metric("Skor", f"{plan.score} / 100")
            st.metric("Durum", plan.label)

            st.subheader("📌 İşlem Planı")
            table = pd.DataFrame(
                {
                    "Parametre": ["Giriş Bölgesi", "Stop", "TP1", "Risk/Reward"],
                    "Değer": [
                        f"{plan.entry_low:.2f} – {plan.entry_high:.2f}",
                        f"{plan.stop:.2f}",
                        f"{plan.tp1:.2f}",
                        f"1 : {plan.rr:.2f}" if np.isfinite(plan.rr) else "—",
                    ],
                }
            )
            st.table(table)

            st.subheader("🧠 Skor Dağılımı")
            b = plan.breakdown
            bdf = pd.DataFrame(
                {
                    "Bileşen": ["Trend", "Fiyat/EMA150", "Momentum (RSI)", "Volatilite (ATR%)", "Uzama (EMA50)"],
                    "Puan": [b.trend_stack, b.price_vs_ema150, b.momentum_rsi, b.volatility_atr, b.extension_vs_ema50],
                    "Maks": [30, 20, 20, 15, 15],
                }
            )
            st.table(bdf)

            st.subheader("📝 Otomatik Teknik Yorum")
            st.markdown(plan.narrative)

            if show_quote and q:
                st.subheader("⚡ Quote (Anlık Özet)")
                keys = ["symbol", "name", "exchange", "currency", "close", "price", "change", "percent_change", "previous_close"]
                compact = {k: q[k] for k in keys if k in q}
                st.write(compact)

            # ✅ PDF BUTTON
            st.subheader("📄 Rapor")
            pdf_bytes = build_pdf_bytes(
                ticker=ticker,
                interval_label=interval_label,
                bars=bars,
                plan=plan,
                quote=(q if show_quote else None),
            )
            st.download_button(
                label="Raporu PDF'e Çevir (İndir)",
                data=pdf_bytes,
                file_name=f"{ticker}_{INTERVAL_MAP[interval_label]}_rapor.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            with st.expander("Detay (debug)"):
                st.json(plan.debug)

with right:
    st.subheader("Grafik")
    if "__df" not in st.session_state:
        st.info("Soldan ticker girip **Getir & Analiz Et** ile başla.")
    else:
        df = st.session_state["__df"]
        ticker = st.session_state["__ticker"]
        plan = st.session_state["__plan"]
        last_price = st.session_state.get("__last_price")

        fig = plot_chart(df, ticker, plan, last_price)
        st.plotly_chart(fig, use_container_width=True)
