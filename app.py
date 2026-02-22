import io
import os
import csv
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime

# -----------------------------
# Optional PDF (ReportLab)
# -----------------------------
PDF_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
except Exception:
    PDF_AVAILABLE = False


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Minervini — Tek Hisse + Portföy (V5 Full)", layout="wide")
st.title("Minervini — Tek Hisse + Portföy Analiz (V5 Full)")

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

# Minervini #5:
MIN_52W_ABOVE_LOW_PCT = 25.0  # Current price >= 52W Low * 1.25

# Files (Streamlit Cloud’da kalıcılık garanti değil; indir/yükle butonları var)
HISTORY_FILE = "history.csv"
PORTFOLIO_FILE = "portfolio.csv"

# Session memory
if "tests" not in st.session_state:
    st.session_state.tests = []

if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ticker", "qty", "avg_cost", "stop", "tp1"])


# =========================================================
# UTILS
# =========================================================
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct_change(a: float, b: float) -> float:
    """(a-b)/b * 100"""
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return np.nan
    return (a - b) / b * 100.0


def rr_from_price(price: float, stop: float, tp: float) -> float:
    if not (np.isfinite(price) and np.isfinite(stop) and np.isfinite(tp)):
        return np.nan
    risk = price - stop
    reward = tp - price
    if risk <= 0:
        return np.nan
    return reward / risk


# =========================================================
# DATA (Twelve Data)
# =========================================================
@st.cache_data(ttl=180)
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
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=180)
def td_quote(symbol: str) -> dict:
    r = requests.get(
        f"{BASE_URL}/quote",
        params={
            "symbol": symbol,
            "apikey": API_KEY,
            "format": "JSON",
        },
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


def parse_ohlcv(payload: dict) -> pd.DataFrame:
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"TwelveData: {payload.get('message')} (code={payload.get('code')})")

    values = payload.get("values")
    if not values:
        raise RuntimeError("TwelveData: values boş döndü (ticker/interval desteklenmiyor olabilir).")

    df = pd.DataFrame(values)

    if "datetime" not in df.columns:
        raise RuntimeError("TwelveData: datetime alanı yok.")

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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema150"] = ema(df["close"], 150)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    return df


# =========================================================
# MINERVINI #5 (52W Low +25%)
# =========================================================
def calc_52w_low_rule(daily_df: pd.DataFrame, current_price: float) -> dict:
    """
    daily_df: daily data containing low column.
    Uses last ~252 trading days.
    """
    tail = daily_df.tail(252).copy()
    low_52w = float(tail["low"].min()) if not tail.empty else float("nan")
    if not np.isfinite(low_52w) or low_52w <= 0 or not np.isfinite(current_price):
        return {"low_52w": low_52w, "pct_above": float("nan"), "ok": False}

    pct_above = (current_price / low_52w - 1.0) * 100.0
    ok = pct_above >= MIN_52W_ABOVE_LOW_PCT
    return {"low_52w": low_52w, "pct_above": float(pct_above), "ok": bool(ok)}


# =========================================================
# SCORING (Split: Position Health vs Add Timing)
# =========================================================
@dataclass
class ScorePack:
    position_health: int  # 0..100
    add_timing: int       # 0..100
    total_legacy: int     # 0..100 (eski style)
    status_tag: str       # emoji label
    action: str           # HOLD / NO-ADD / REDUCE / EXIT
    entry_low: float
    entry_high: float
    stop_suggest: float
    tp1_suggest: float
    rr_suggest: float
    dist_to_entry_pct: float
    watch_level: float
    narrative: str
    scenario: str
    minervini_low_52w: float
    minervini_pct_above_low: float
    minervini_ok: bool


def label_from_total(score: int) -> str:
    if score >= 75:
        return "UYGUN"
    if score >= 60:
        return "SINIRDA"
    return "UYGUN DEĞİL"


def _dist_to_entry_pct(price: float, entry_low: float, entry_high: float) -> float:
    if price > entry_high:
        return ((price - entry_high) / entry_high) * 100
    if price < entry_low:
        return -((entry_low - price) / entry_low) * 100
    return 0.0


def _proximity_points(dist_pct: float) -> int:
    # 0..60
    if dist_pct == 0:
        return 60
    d = abs(dist_pct)
    if d <= 2:
        return 45
    if d <= 5:
        return 30
    if d <= 10:
        return 15
    return 0


def _extension_points(is_extended: bool) -> int:
    return 0 if is_extended else 40


def _detect_consolidation(atr_pct: float, rsi14: float) -> bool:
    # basit sıkışma
    return (atr_pct < 2.0) and (45 <= rsi14 <= 55)


def status_tag_for_add(timing_score: int, setup_ok: bool, trend_broken: bool, extended: bool, in_entry: bool, consolidation: bool) -> str:
    if trend_broken or not setup_ok:
        return "🔴 TREND BOZULDU"
    if consolidation:
        return "🔵 KONSOLİDASYON"
    if in_entry and timing_score >= 70:
        return "🟢 ALIM BÖLGESİNDE"
    if extended and timing_score < 50:
        return "⚫ UZAMIŞ — KOVALAMA"
    return "🟡 PULLBACK BEKLENİYOR"


def decide_action(position_health: int, status_tag: str, minervini_ok: bool) -> str:
    """
    Portföy yönetimi aksiyonu:
    - minervini_ok False: genelde weak base -> NO-ADD/REDUCE
    - trend bozuk: REDUCE/EXIT (stop planına bağlı)
    - uzamış: HOLD / NO-ADD
    - pullback/konsolidasyon: HOLD, add only pullback
    """
    if not minervini_ok:
        # dipten yeni çıkmış / weak base
        if "TREND BOZULDU" in status_tag:
            return "REDUCE / EXIT"
        return "HOLD / NO-ADD (WEAK BASE)"
    if "TREND BOZULDU" in status_tag:
        return "REDUCE / EXIT"
    if "UZAMIŞ" in status_tag:
        return "HOLD / NO-ADD"
    if "ALIM BÖLGESİNDE" in status_tag:
        return "HOLD / ADD-ONLY-PULLBACK"
    if "KONSOLİDASYON" in status_tag:
        return "HOLD / WATCH"
    return "HOLD / WATCH"


def build_scores(df_tf: pd.DataFrame, df_daily_for_52w: pd.DataFrame, current_price: float) -> ScorePack:
    last = df_tf.iloc[-1]
    close = float(current_price)

    ema20_v = float(last["ema20"])
    ema50_v = float(last["ema50"])
    ema150_v = float(last["ema150"])
    ema200_v = float(last["ema200"])
    rsi14_v = float(last["rsi14"])
    atr14_v = float(last["atr14"])

    atr_pct = (atr14_v / close) * 100 if close else float("nan")
    dist_ema50_pct = ((close - ema50_v) / ema50_v) * 100 if ema50_v else float("nan")
    dist_ema150_pct = ((close - ema150_v) / ema150_v) * 100 if ema150_v else float("nan")

    # Trend structure
    trend_stack_ok = (ema50_v > ema150_v > ema200_v)
    ema200_slope = slope(df_tf["ema200"], lookback=20)
    long_trend_ok = (ema200_slope > 0)

    trend_broken = (close < ema200_v) or ((not long_trend_ok) and (not trend_stack_ok))

    # Momentum
    momentum_ok = (rsi14_v >= 55)
    momentum_border = (50 <= rsi14_v < 55)

    # Volatility band
    vol_ok = (2.0 <= atr_pct <= 6.0)
    vol_border = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 8.0)

    # Price vs EMA150
    price_above_ema150 = close >= ema150_v
    price_near_ema150 = close >= ema150_v * 0.98

    # Extension
    extended = dist_ema50_pct > 8.0

    # Legacy total (0..100) — eski ekranda kullandığın gibi
    total = 0
    trend_pts = 30 if (trend_stack_ok and long_trend_ok) else (20 if trend_stack_ok else (10 if long_trend_ok else 0))
    total += trend_pts
    p_pts = 20 if price_above_ema150 else (10 if price_near_ema150 else 0)
    total += p_pts
    m_pts = 20 if momentum_ok else (10 if momentum_border else 0)
    total += m_pts
    v_pts = 15 if vol_ok else (7 if vol_border else 0)
    total += v_pts
    e_pts = 15 if not extended else 0
    total += e_pts

    # Entry zone
    entry_low = float(min(ema20_v, ema50_v))
    entry_high = float(max(ema20_v, ema50_v))
    entry_mid = (entry_low + entry_high) / 2.0

    # Suggested stop/tp1 (generic)
    stop_ema = ema50_v * 0.995
    stop_atr = entry_mid - 1.2 * atr14_v
    stop_suggest = float(max(stop_ema, stop_atr))
    if stop_suggest >= entry_mid:
        stop_suggest = float(entry_mid * 0.99)

    risk = entry_mid - stop_suggest
    tp1_suggest = float(entry_mid + 2.0 * risk) if risk > 0 else float(entry_mid * 1.02)
    rr_suggest = (tp1_suggest - entry_mid) / risk if risk > 0 else float("nan")

    # -----------------------------
    # Split scoring:
    # Position Health (0..100) – eldeki pozisyonu taşır mıyım?
    # Add Timing (0..100) – ekleme/giriş uygun mu?
    # -----------------------------
    # Position Health: trend + price position + momentum + 52w rule (Minervini #5) => 100
    # Trend block 0..40
    ph_trend = 40 if (trend_stack_ok and long_trend_ok) else (25 if trend_stack_ok else (15 if long_trend_ok else 0))
    # Price position 0..20
    ph_price = 20 if price_above_ema150 else (10 if price_near_ema150 else 0)
    # Momentum 0..20
    ph_mom = 20 if momentum_ok else (10 if momentum_border else 0)
    # Volatility sanity 0..10 (çok ekstremse kırp)
    ph_vol = 10 if vol_ok else (5 if vol_border else 0)
    # Minervini #5 0..10
    m5 = calc_52w_low_rule(df_daily_for_52w, close)
    ph_m5 = 10 if m5["ok"] else 0

    position_health = int(ph_trend + ph_price + ph_mom + ph_vol + ph_m5)
    position_health = max(0, min(100, position_health))

    # Add Timing: proximity to entry (0..60) + extension (0..40)
    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)
    ext_pts = _extension_points(extended)
    add_timing = int(prox_pts + ext_pts)
    add_timing = max(0, min(100, add_timing))

    in_entry = (entry_low <= close <= entry_high)
    consolidation = _detect_consolidation(atr_pct, rsi14_v)
    setup_ok = position_health >= 60  # basic health gate

    status_tag = status_tag_for_add(
        timing_score=add_timing,
        setup_ok=setup_ok,
        trend_broken=trend_broken,
        extended=extended,
        in_entry=in_entry,
        consolidation=consolidation,
    )

    action = decide_action(position_health, status_tag, m5["ok"])

    watch_level = float(entry_high)

    # Narrative
    trend_text = "güçlü" if (trend_stack_ok and (price_above_ema150 or price_near_ema150)) else ("zayıf" if close < ema200_v else "karışık")
    mom_text = "sağlıklı" if 55 <= rsi14_v <= 75 else ("ısınmış" if rsi14_v > 75 else "zayıf/sınır")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_border else "yüksek")

    scenario = ""
    if "🟢" in status_tag:
        scenario = "Fiyat EMA20–EMA50 bandında. Eklemek için band içinde güç işareti (higher-low / güçlü kapanış) ara."
    elif "🟡" in status_tag:
        scenario = "Fiyat band dışında. Pullback ile EMA20–EMA50 bandına yaklaşmasını bekle; kovalamayı engelle."
    elif "🔵" in status_tag:
        scenario = "Düşük volatilite sıkışması var. Kırılım + hacim artışı görmeden agresif aksiyon yok."
    elif "⚫" in status_tag:
        scenario = "Uzamış. Ekleme yok. Pullback gelmeden yeni risk alma."
    else:
        scenario = "Trend filtresi bozuk. Ekleme yok; stop/pozisyon azaltma planını çalıştır."

    narrative = (
        f"**Güncel Fiyat:** {close:.2f}\n\n"
        f"**Position Health:** {position_health}/100  |  **Add Timing:** {add_timing}/100  |  **Legacy Total:** {total}/100 ({label_from_total(total)})\n"
        f"**Durum:** {status_tag}\n"
        f"**Aksiyon (Portföy):** {action}\n\n"
        f"EMA20: {ema20_v:.2f} | EMA50: {ema50_v:.2f} | EMA150: {ema150_v:.2f} | EMA200: {ema200_v:.2f}\n"
        f"RSI14: {rsi14_v:.1f} ({mom_text}) | ATR%: {atr_pct:.2f} ({vol_text})\n"
        f"EMA50 mesafe: {dist_ema50_pct:+.2f}% | EMA150 mesafe: {dist_ema150_pct:+.2f}%\n\n"
        f"**Minervini #5:** 52W Low = {m5['low_52w']:.2f} | Above Low = {m5['pct_above']:.1f}% | {'PASS' if m5['ok'] else 'FAIL'}\n\n"
        f"**Entry Bandı:** {entry_low:.2f} – {entry_high:.2f}  |  Entry Mesafe: {dist_entry_pct:+.2f}%\n"
        f"**Stop (Öneri):** {stop_suggest:.2f}  |  **TP1 (Öneri):** {tp1_suggest:.2f}  |  **R/R (Öneri):** {('1:'+format(rr_suggest,'.2f')) if np.isfinite(rr_suggest) else '—'}\n"
    )

    return ScorePack(
        position_health=position_health,
        add_timing=add_timing,
        total_legacy=int(total),
        status_tag=status_tag,
        action=action,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        stop_suggest=float(stop_suggest),
        tp1_suggest=float(tp1_suggest),
        rr_suggest=float(rr_suggest) if np.isfinite(rr_suggest) else float("nan"),
        dist_to_entry_pct=float(dist_entry_pct),
        watch_level=float(watch_level),
        narrative=narrative,
        scenario=scenario,
        minervini_low_52w=float(m5["low_52w"]) if np.isfinite(m5["low_52w"]) else float("nan"),
        minervini_pct_above_low=float(m5["pct_above"]) if np.isfinite(m5["pct_above"]) else float("nan"),
        minervini_ok=bool(m5["ok"]),
    )


# =========================================================
# HISTORY / PORTFOLIO STORAGE
# =========================================================
def save_history_row(row: dict):
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def file_bytes(path: str) -> bytes:
    if not os.path.isfile(path):
        return b""
    with open(path, "rb") as f:
        return f.read()


def save_portfolio_df(df_port: pd.DataFrame):
    dfp = df_port.copy()
    dfp["ticker"] = dfp["ticker"].astype(str).str.upper().str.strip()
    dfp.to_csv(PORTFOLIO_FILE, index=False)


def load_portfolio_df() -> pd.DataFrame:
    dfp = read_csv_df(PORTFOLIO_FILE)
    if dfp.empty:
        return st.session_state.portfolio.copy()
    expected = ["ticker", "qty", "avg_cost", "stop", "tp1"]
    for c in expected:
        if c not in dfp.columns:
            dfp[c] = np.nan
    dfp = dfp[expected]
    dfp["ticker"] = dfp["ticker"].astype(str).str.upper().str.strip()
    return dfp


# =========================================================
# PDF EXPORT
# =========================================================
def _wrap_lines(text: str, max_chars: int = 95):
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


def build_pdf_bytes(symbol: str, interval_label: str, bars: int, price: float, pack: ScorePack, quote: dict | None):
    if not PDF_AVAILABLE:
        raise RuntimeError("PDF modülü yok (reportlab). requirements.txt içine reportlab eklemelisin.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    x = 2.0 * cm
    y = h - 2.0 * cm
    lh = 14

    def draw_line(txt, font="Helvetica", size=11, space=lh):
        nonlocal y
        c.setFont(font, size)
        c.drawString(x, y, txt)
        y -= space
        if y < 2.0 * cm:
            c.showPage()
            y = h - 2.0 * cm

    draw_line("Minervini Analiz Raporu (V5 Full)", font="Helvetica-Bold", size=16, space=18)
    draw_line(f"Ticker: {symbol}    Zaman: {interval_label}    Bar: {bars}", size=11)
    draw_line(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", size=10)
    draw_line("", size=10, space=10)

    draw_line("Özet", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"Güncel Fiyat: {price:.2f}", size=11)
    draw_line(f"Position Health: {pack.position_health}/100", size=11)
    draw_line(f"Add Timing: {pack.add_timing}/100", size=11)
    draw_line(f"Legacy Total: {pack.total_legacy}/100", size=11)
    draw_line(f"Durum: {pack.status_tag}", size=11)
    draw_line(f"Aksiyon: {pack.action}", size=11)
    draw_line("", size=10, space=10)

    draw_line("Minervini #5", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"52W Low: {pack.minervini_low_52w:.2f} | Above Low: {pack.minervini_pct_above_low:.1f}% | {'PASS' if pack.minervini_ok else 'FAIL'}", size=11)
    draw_line("", size=10, space=10)

    if quote and isinstance(quote, dict):
        draw_line("Quote (Anlık)", font="Helvetica-Bold", size=13, space=16)
        for k in ["name", "exchange", "currency", "price", "close", "change", "percent_change", "previous_close"]:
            if k in quote:
                draw_line(f"{k}: {quote[k]}", size=10, space=12)
        draw_line("", size=10, space=10)

    draw_line("İşlem Seviyeleri (Öneri)", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"Entry Bandı: {pack.entry_low:.2f} – {pack.entry_high:.2f}  (Mesafe {pack.dist_to_entry_pct:+.2f}%)", size=11)
    draw_line(f"Stop: {pack.stop_suggest:.2f}  |  TP1: {pack.tp1_suggest:.2f}  |  R/R: {('1:'+format(pack.rr_suggest,'.2f')) if np.isfinite(pack.rr_suggest) else '—'}", size=11)
    draw_line("", size=10, space=10)

    draw_line("Senaryo", font="Helvetica-Bold", size=13, space=16)
    for ln in _wrap_lines(pack.scenario, 95):
        draw_line(ln, size=10, space=12)
    draw_line("", size=10, space=10)

    draw_line("Yorum", font="Helvetica-Bold", size=13, space=16)
    plain = pack.narrative.replace("**", "")
    for ln in _wrap_lines(plain, 95):
        draw_line(ln, size=10, space=12)

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
# CHART
# =========================================================
def plot_chart(df: pd.DataFrame, symbol: str, pack: ScorePack, current_price: float):
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

    # Entry band
    fig.add_hrect(
        y0=pack.entry_low,
        y1=pack.entry_high,
        opacity=0.15,
        line_width=0,
        annotation_text="ENTRY",
        annotation_position="top left",
    )

    # Suggested stop/tp
    fig.add_hline(y=pack.stop_suggest, line_dash="dash", annotation_text="STOP", annotation_position="bottom left")
    fig.add_hline(y=pack.tp1_suggest, line_dash="dash", annotation_text="TP1", annotation_position="top left")

    # Current price line
    fig.add_hline(y=float(current_price), line_dash="dot", annotation_text="GÜNCEL", annotation_position="top right")

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
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Genel Ayarlar")
    default_tf_label = st.selectbox("Varsayılan zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=0)
    bars = st.slider("Bar sayısı", min_value=120, max_value=900, value=300, step=10)

    # Quote is extra API call; default off
    use_quote = st.checkbox("Quote (anlık fiyat) kullan", value=False)
    st.caption("Quote açık → hisse başına +1 API çağrısı.")

    st.divider()
    st.subheader("Hafıza (Oturum)")
    if st.session_state.tests:
        df_mem = pd.DataFrame(st.session_state.tests)
        cols = ["timestamp", "ticker", "timeframe", "price", "position_health", "add_timing", "status_tag", "action", "m5_ok", "m5_pct"]
        cols = [c for c in cols if c in df_mem.columns]
        st.dataframe(df_mem[cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Henüz test yok.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Oturumu Temizle", use_container_width=True):
            st.session_state.tests = []
            st.rerun()
    with col2:
        hb = file_bytes(HISTORY_FILE)
        st.download_button(
            "history.csv indir",
            data=hb if hb else b"",
            file_name="history.csv",
            mime="text/csv",
            disabled=(not bool(hb)),
            use_container_width=True,
        )

    with st.expander("history.csv (son 200)"):
        hdf = read_csv_df(HISTORY_FILE)
        if hdf.empty:
            st.info("history.csv yok/boş.")
        else:
            st.dataframe(hdf.tail(200), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Portföy Dosyası")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("portfolio.csv yükle", use_container_width=True):
            st.session_state.portfolio = load_portfolio_df()
            st.rerun()
    with c4:
        pb = file_bytes(PORTFOLIO_FILE)
        st.download_button(
            "portfolio.csv indir",
            data=pb if pb else b"",
            file_name="portfolio.csv",
            mime="text/csv",
            disabled=(not bool(pb)),
            use_container_width=True,
        )


# =========================================================
# MAIN TABS
# =========================================================
tab_single, tab_portfolio = st.tabs(["📈 Tek Hisse", "🧳 Portföy (Eldeki)"])


# =========================================================
# TAB: SINGLE
# =========================================================
with tab_single:
    left, right = st.columns([0.36, 0.64], vertical_alignment="top")

    with left:
        st.subheader("Tek Hisse Analizi")
        ticker = st.text_input("Ticker", placeholder="NVDA / TSLA / PLTR").strip().upper()
        tf_label = st.selectbox("Zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=list(INTERVAL_MAP.keys()).index(default_tf_label))
        run = st.button("Getir & Analiz Et", type="primary", use_container_width=True)

        if run:
            if not ticker:
                st.warning("Ticker gir.")
            else:
                interval = INTERVAL_MAP[tf_label]

                # 52w rule needs daily regardless. We'll fetch daily once.
                with st.spinner("Veriler çekiliyor..."):
                    try:
                        daily_payload = td_time_series(ticker, "1day", max(260, bars))
                        df_daily = parse_ohlcv(daily_payload)
                        df_daily = add_indicators(df_daily)
                    except Exception as e:
                        st.error(f"Daily veri alınamadı: {e}")
                        st.stop()

                    # TF data:
                    try:
                        if interval == "1day":
                            df_tf = df_daily.copy()
                        else:
                            tf_payload = td_time_series(ticker, interval, bars)
                            df_tf = parse_ohlcv(tf_payload)
                            df_tf = add_indicators(df_tf)
                    except Exception as e:
                        st.error(f"{tf_label} veri alınamadı: {e}")
                        st.stop()

                # price
                candle_close = float(df_tf.iloc[-1]["close"])
                quote = {}
                price = candle_close
                if use_quote:
                    try:
                        quote = td_quote(ticker)
                        # TwelveData quote fields vary; use price first
                        if "price" in quote and safe_float(quote["price"]) > 0:
                            price = float(quote["price"])
                        elif "close" in quote and safe_float(quote["close"]) > 0:
                            price = float(quote["close"])
                    except Exception:
                        quote = {}

                pack = build_scores(df_tf=df_tf, df_daily_for_52w=df_daily, current_price=price)

                # Save to session + history
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "timeframe": interval,
                    "price": round(price, 4),
                    "position_health": pack.position_health,
                    "add_timing": pack.add_timing,
                    "legacy_total": pack.total_legacy,
                    "status_tag": pack.status_tag,
                    "action": pack.action,
                    "entry_low": round(pack.entry_low, 4),
                    "entry_high": round(pack.entry_high, 4),
                    "dist_to_entry_pct": round(pack.dist_to_entry_pct, 4),
                    "m5_ok": int(pack.minervini_ok),
                    "m5_low_52w": round(pack.minervini_low_52w, 4) if np.isfinite(pack.minervini_low_52w) else "",
                    "m5_pct": round(pack.minervini_pct_above_low, 2) if np.isfinite(pack.minervini_pct_above_low) else "",
                }
                st.session_state.tests.append(row)
                try:
                    save_history_row(row)
                except Exception:
                    pass

                st.divider()
                st.subheader("Özet")
                st.metric("Güncel Fiyat", f"${price:,.2f}")
                st.metric("Position Health", f"{pack.position_health} / 100")
                st.metric("Add Timing", f"{pack.add_timing} / 100")
                st.metric("Durum", pack.status_tag)
                st.metric("Aksiyon (Portföy Mantığı)", pack.action)

                st.subheader("Minervini #5 (52W Low +25%)")
                if np.isfinite(pack.minervini_low_52w):
                    st.write(f"52W Low: **{pack.minervini_low_52w:.2f}** | Above Low: **{pack.minervini_pct_above_low:.1f}%** | **{'PASS' if pack.minervini_ok else 'FAIL'}**")
                else:
                    st.warning("52W Low hesaplanamadı (daily veri yetersiz olabilir).")

                st.subheader("İşlem Seviyeleri (Öneri)")
                plan_table = pd.DataFrame(
                    {
                        "Parametre": ["Entry Bandı", "Entry Mesafe %", "Takip", "Stop (Öneri)", "TP1 (Öneri)", "R/R (Öneri)"],
                        "Değer": [
                            f"{pack.entry_low:.2f} – {pack.entry_high:.2f}",
                            f"{pack.dist_to_entry_pct:+.2f}%",
                            f"{pack.watch_level:.2f}",
                            f"{pack.stop_suggest:.2f}",
                            f"{pack.tp1_suggest:.2f}",
                            (f"1 : {pack.rr_suggest:.2f}" if np.isfinite(pack.rr_suggest) else "—"),
                        ],
                    }
                )
                st.table(plan_table)

                st.subheader("Senaryo")
                st.write(pack.scenario)

                st.subheader("Otomatik Yorum")
                st.markdown(pack.narrative)

                if use_quote and quote:
                    st.subheader("Quote")
                    keys = ["symbol", "name", "exchange", "currency", "price", "close", "change", "percent_change", "previous_close"]
                    compact = {k: quote[k] for k in keys if k in quote}
                    st.write(compact)

                st.subheader("Rapor (PDF)")
                if not PDF_AVAILABLE:
                    st.warning("PDF için requirements.txt içine `reportlab` eklemelisin (aşağıda not var).")
                else:
                    pdf_bytes = build_pdf_bytes(
                        symbol=ticker,
                        interval_label=tf_label,
                        bars=bars,
                        price=price,
                        pack=pack,
                        quote=(quote if use_quote else None),
                    )
                    st.download_button(
                        "Raporu PDF'e Çevir (İndir)",
                        data=pdf_bytes,
                        file_name=f"{ticker}_{interval}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                # store for chart on right
                st.session_state["__single_df"] = df_tf
                st.session_state["__single_ticker"] = ticker
                st.session_state["__single_pack"] = pack
                st.session_state["__single_price"] = price

    with right:
        st.subheader("Grafik")
        if "__single_df" not in st.session_state:
            st.info("Soldan ticker girip **Getir & Analiz Et** ile başla.")
        else:
            df_tf = st.session_state["__single_df"]
            ticker = st.session_state["__single_ticker"]
            pack = st.session_state["__single_pack"]
            price = float(st.session_state["__single_price"])
            fig = plot_chart(df_tf, ticker, pack, price)
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB: PORTFOLIO (Eldeki)
# =========================================================
with tab_portfolio:
    st.subheader("Portföy Analizi (Eldeki Pozisyon Yönetimi)")
    st.caption("Burada amaç yeni alım değil; **eldeki pozisyonlar** için HOLD / NO-ADD / REDUCE/EXIT kararını netleştirmek.")

    top_left, top_right = st.columns([0.65, 0.35], vertical_alignment="top")

    with top_right:
        st.markdown("### Dosya / Hızlı İşlemler")
        if st.button("Portföyü Kaydet (portfolio.csv)", type="primary", use_container_width=True):
            try:
                save_portfolio_df(st.session_state.portfolio)
                st.success("portfolio.csv kaydedildi.")
            except Exception as e:
                st.error(f"Kaydedilemedi: {e}")

        if st.button("Portföyü Temizle", use_container_width=True):
            st.session_state.portfolio = pd.DataFrame(columns=["ticker", "qty", "avg_cost", "stop", "tp1"])
            try:
                save_portfolio_df(st.session_state.portfolio)
            except Exception:
                pass
            st.rerun()

        st.markdown("### Not")
        st.write("Portföy analiz **default Daily** çalışır. (API tasarrufu + 52W low için şart)")

    with top_left:
        st.markdown("### Portföy Girişi")
        st.session_state.portfolio = st.data_editor(
            st.session_state.portfolio,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "qty": st.column_config.NumberColumn("Adet (opsiyonel)", min_value=0.0, step=1.0),
                "avg_cost": st.column_config.NumberColumn("Alış Ort.", min_value=0.0, step=0.01, format="%.2f"),
                "stop": st.column_config.NumberColumn("Stop", min_value=0.0, step=0.01, format="%.2f"),
                "tp1": st.column_config.NumberColumn("TP1", min_value=0.0, step=0.01, format="%.2f"),
            },
        )

        st.markdown("### Analiz")
        analyze_pf = st.button("Portföyü Analiz Et", type="primary", use_container_width=True)

    if analyze_pf:
        dfp = st.session_state.portfolio.copy()
        if dfp.empty:
            st.warning("Portföy boş. En az 1 satır ekle.")
        else:
            dfp["ticker"] = dfp["ticker"].astype(str).str.upper().str.strip()
            dfp = dfp[dfp["ticker"].str.len() > 0].copy()
            if dfp.empty:
                st.warning("Geçerli ticker yok.")
            else:
                rows = []
                with st.spinner("Portföy verileri çekiliyor..."):
                    for _, r in dfp.iterrows():
                        tkr = str(r.get("ticker", "")).upper().strip()
                        qty = safe_float(r.get("qty"))
                        avg_cost = safe_float(r.get("avg_cost"))
                        user_stop = safe_float(r.get("stop"))
                        user_tp1 = safe_float(r.get("tp1"))

                        try:
                            # Daily only (API friendly + 52W required)
                            payload = td_time_series(tkr, "1day", max(260, bars))
                            df = parse_ohlcv(payload)
                            df = add_indicators(df)

                            candle_close = float(df.iloc[-1]["close"])
                            price = candle_close

                            quote = {}
                            if use_quote:
                                try:
                                    quote = td_quote(tkr)
                                    if "price" in quote and safe_float(quote["price"]) > 0:
                                        price = float(quote["price"])
                                    elif "close" in quote and safe_float(quote["close"]) > 0:
                                        price = float(quote["close"])
                                except Exception:
                                    quote = {}

                            pack = build_scores(df_tf=df, df_daily_for_52w=df, current_price=price)

                            pnl_pct = pct_change(price, avg_cost) if np.isfinite(avg_cost) and avg_cost > 0 else np.nan
                            stop_dist_pct = pct_change(price, user_stop) if np.isfinite(user_stop) and user_stop > 0 else np.nan
                            tp_dist_pct = pct_change(user_tp1, price) if np.isfinite(user_tp1) and user_tp1 > 0 else np.nan
                            rr_user = rr_from_price(price, user_stop, user_tp1)

                            pos_value = (qty * price) if np.isfinite(qty) and np.isfinite(price) else np.nan
                            risk_per_share = (price - user_stop) if (np.isfinite(user_stop) and np.isfinite(price)) else np.nan
                            risk_value = (risk_per_share * qty) if (np.isfinite(risk_per_share) and np.isfinite(qty)) else np.nan

                            # kısa not:
                            if "TREND BOZULDU" in pack.status_tag:
                                note = "Trend bozuk → ekleme yok; stop/riski azalt planı."
                            elif "UZAMIŞ" in pack.status_tag:
                                note = "Uzamış → ekleme yok; pullback bekle."
                            elif "PULLBACK" in pack.status_tag:
                                note = "Pullback → entry bandı yaklaşınca yeniden değerlendir."
                            elif "KONSOLİDASYON" in pack.status_tag:
                                note = "Sıkışma → kırılım/hacim onayı bekle."
                            else:
                                note = "İzle."

                            if not pack.minervini_ok:
                                note = (note + " | Minervini #5 FAIL → weak base (NO-ADD).").strip()

                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": round(price, 2),
                                "Alış Ort.": round(avg_cost, 2) if np.isfinite(avg_cost) else "",
                                "P&L %": round(pnl_pct, 2) if np.isfinite(pnl_pct) else "",
                                "Stop": round(user_stop, 2) if np.isfinite(user_stop) else "",
                                "Stop Mesafe %": round(stop_dist_pct, 2) if np.isfinite(stop_dist_pct) else "",
                                "TP1": round(user_tp1, 2) if np.isfinite(user_tp1) else "",
                                "TP1 Mesafe %": round(tp_dist_pct, 2) if np.isfinite(tp_dist_pct) else "",
                                "R (TP1/Stop)": round(rr_user, 2) if np.isfinite(rr_user) else "",
                                "Position Health": pack.position_health,
                                "Add Timing": pack.add_timing,
                                "Durum": pack.status_tag,
                                "Aksiyon": pack.action,
                                "Minervini #5": ("PASS" if pack.minervini_ok else "FAIL"),
                                "52W Above %": round(pack.minervini_pct_above_low, 1) if np.isfinite(pack.minervini_pct_above_low) else "",
                                "Entry Bandı": f"{pack.entry_low:.2f}–{pack.entry_high:.2f}",
                                "Entry Mesafe %": f"{pack.dist_to_entry_pct:+.2f}",
                                "Poz. Değeri": round(pos_value, 2) if np.isfinite(pos_value) else "",
                                "Risk $": round(risk_value, 2) if np.isfinite(risk_value) else "",
                                "Not": note,
                            })

                        except Exception as e:
                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": "",
                                "Alış Ort.": "",
                                "P&L %": "",
                                "Stop": "",
                                "Stop Mesafe %": "",
                                "TP1": "",
                                "TP1 Mesafe %": "",
                                "R (TP1/Stop)": "",
                                "Position Health": "",
                                "Add Timing": "",
                                "Durum": "HATA",
                                "Aksiyon": "",
                                "Minervini #5": "",
                                "52W Above %": "",
                                "Entry Bandı": "",
                                "Entry Mesafe %": "",
                                "Poz. Değeri": "",
                                "Risk $": "",
                                "Not": f"Veri/analiz hatası: {e}",
                            })

                out = pd.DataFrame(rows)

                st.markdown("### Sonuç Tablosu")
                st.dataframe(out, use_container_width=True, hide_index=True)

                # quick summary
                st.markdown("### Hızlı Özet")
                if not out.empty:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🟢 Alım Bandına Yakın", int((out["Durum"].astype(str).str.contains("ALIM BÖLGESİNDE")).sum()))
                    c2.metric("🟡 Pullback", int((out["Durum"].astype(str).str.contains("PULLBACK")).sum()))
                    c3.metric("⚫ Uzamış", int((out["Durum"].astype(str).str.contains("UZAMIŞ")).sum()))
                    c4.metric("🔴 Trend Bozuk", int((out["Durum"].astype(str).str.contains("TREND BOZULDU")).sum()))

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Portföy Analiz CSV indir",
                    data=csv_bytes,
                    file_name="portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =========================================================
# Footer info
# =========================================================
st.caption(
    "Not: PDF butonu çalışmıyorsa requirements.txt içine `reportlab` ekle. "
    "Quote açıkken her hisse için +1 API çağrısı yapılır."
)
