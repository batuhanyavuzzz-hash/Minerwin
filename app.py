# app.py
# Tek Hisse + Portföy Analiz (V5.0) — Twelve Data
# - Minervini #5 (52W low +%25 filtresi) entegre
# - TP1 + TP2: Minervini uyumlu, hisse “taşıma kapasitesi” + geçmiş impuls + 52W high tavanı ile
# - Elde olan hisse dili: TUT / SAT / STOP YUKARI / KISMİ KAR / İZLE
# - Grafik: mum + EMA + fiyat çizgisi (toggle, çizgi her zaman kalabilir)
# - İşlem yönetimi: Streamlit form ile state korunur (yazarken sayfa refresh hissi azalır)

import io
import os
import csv
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Tek Hisse + Portföy Analiz (V5.0)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Twelve Data (V5.0 | Minervini + TP1/TP2 + İşlem Yönetimi)")

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error('TWELVEDATA_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine ekle: TWELVEDATA_API_KEY="..."')
    st.stop()

BASE_URL = "https://api.twelvedata.com"
HISTORY_FILE = "history.csv"
PORTFOLIO_FILE = "portfolio.csv"

INTERVAL_MAP = {
    "Haftalık (1week)": "1week",
    "Günlük (1day)": "1day",
    "Saatlik (1h)": "1h",
    "15 Dakika (15min)": "15min",
}

DEFAULT_SINGLE_INTERVAL_LABEL = "Günlük (1day)"

# Twelve Data free plan: regular session odaklı; pre/post-market quote genelde yok.
st.caption("Not: Twelve Data Free/BASIC plan genelde pre-market/after-hours fiyatı vermez; piyasa kapalıyken quote son kapanışı döndürebilir.")


# =========================================================
# SESSION STATE INIT
# =========================================================
if "daily_tests" not in st.session_state:
    st.session_state.daily_tests = []  # single tests memory (session)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"])

# per-ticker “işlem yönetimi” state (tek hisse ekranında)
if "trade_mgmt" not in st.session_state:
    st.session_state.trade_mgmt = {}  # dict[ticker] = {"entry":..., "stop":..., "tp1":..., "tp2":...}


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


def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return np.nan
    return (a - b) / b * 100


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return lo


# =========================================================
# DATA (Twelve Data)
# =========================================================
@st.cache_data(ttl=120)
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


@st.cache_data(ttl=120)
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
# HISTORY (CSV) + SESSION MEMORY
# =========================================================
def save_to_history(row: dict):
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_history_df() -> pd.DataFrame:
    if not os.path.isfile(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()


def history_csv_bytes() -> bytes:
    if not os.path.isfile(HISTORY_FILE):
        return b""
    with open(HISTORY_FILE, "rb") as f:
        return f.read()


def clear_today_session():
    st.session_state.daily_tests = []


def save_portfolio_df(df_port: pd.DataFrame):
    df_port = df_port.copy()
    df_port["ticker"] = df_port["ticker"].astype(str).str.upper().str.strip()
    df_port.to_csv(PORTFOLIO_FILE, index=False)


def load_portfolio_df() -> pd.DataFrame:
    if not os.path.isfile(PORTFOLIO_FILE):
        return st.session_state.portfolio.copy()
    try:
        dfp = pd.read_csv(PORTFOLIO_FILE)
        expected = ["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"]
        for c in expected:
            if c not in dfp.columns:
                dfp[c] = np.nan
        dfp = dfp[expected]
        return dfp
    except Exception:
        return st.session_state.portfolio.copy()


def portfolio_csv_bytes() -> bytes:
    if not os.path.isfile(PORTFOLIO_FILE):
        return b""
    with open(PORTFOLIO_FILE, "rb") as f:
        return f.read()


# =========================================================
# MINERVINI (52W LOW/HIGH) HELPERS
# =========================================================
@st.cache_data(ttl=120)
def fetch_daily_52w(symbol: str) -> Tuple[float, float]:
    """
    52 haftalık dip/tepe için günlük veriden yaklaşık 260 bar kullanır.
    Twelve Data free plan için yeterli.
    """
    payload = td_time_series(symbol, "1day", 260)
    df = parse_ohlcv(payload)
    low_52w = float(df["low"].min())
    high_52w = float(df["high"].max())
    return low_52w, high_52w


def minervini_rule5_ok(price: float, low_52w: float) -> bool:
    if not (np.isfinite(price) and np.isfinite(low_52w) and low_52w > 0):
        return False
    return price >= 1.25 * low_52w


# =========================================================
# TP ENGINE (MINERVINI-ALIGNED TARGETS)
# =========================================================
def _trend_capacity_level(
    setup_score: int,
    ema50: float,
    ema150: float,
    ema200: float,
    ema200_slope: float,
    rsi14: float,
    price: float,
) -> str:
    """
    LOW/MID/HIGH — sayı değil, seviye (kendini kandırmayı azaltır).
    """
    votes = 0
    if (ema50 > ema150 > ema200) and (ema200_slope > 0):
        votes += 2
    elif (ema50 > ema150) and (ema200_slope > 0):
        votes += 1

    if setup_score >= 80:
        votes += 2
    elif setup_score >= 70:
        votes += 1

    if 60 <= rsi14 <= 72:
        votes += 2
    elif 55 <= rsi14 < 60:
        votes += 1
    elif rsi14 < 50:
        votes -= 1

    # price positioning: above ema50 helps carrying
    if np.isfinite(price) and np.isfinite(ema50) and price >= ema50:
        votes += 1

    if votes >= 6:
        return "HIGH"
    if votes >= 3:
        return "MID"
    return "LOW"


def _impulse_cap_pct_from_history(df: pd.DataFrame, lookback: int = 90) -> float:
    """
    Geçmiş “sağlıklı taşıma” için kaba ama sağlam üst sınır:
    - Son lookback bar içinde “en iyi dip->tepe” yüzdesi.
    - Kendini kandırma engeli: TP’yi geçmişin üstüne yazma.
    """
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")

    c = df["close"].dropna().astype(float)
    if len(c) < max(lookback, 30):
        return float("nan")

    c = c.iloc[-lookback:].reset_index(drop=True)
    # Running min + future max yaklaşımı (O(n))
    running_min = c.iloc[0]
    best = 0.0
    for i in range(1, len(c)):
        running_min = min(running_min, float(c.iloc[i - 1]))
        if running_min <= 0:
            continue
        move = (float(c.iloc[i]) - running_min) / running_min
        if move > best:
            best = move

    # best is ratio (0.12 => %12)
    return float(best * 100.0)  # percent


def compute_tp1_tp2_minervini(
    df_for_impulse: pd.DataFrame,
    entry: float,
    stop: float,
    close: float,
    atr14: float,
    setup_score: int,
    ema50: float,
    ema150: float,
    ema200: float,
    ema200_slope: float,
    rsi14: float,
    high_52w: float,
) -> Tuple[float, float, float, str, Dict[str, Any]]:
    """
    Çıktı:
      tp1, tp2, expected_move_pct, capacity_level, debug

    Minervini uyumu:
    - TP hedefleri “istek” değil “taşıma kapasitesi” + geçmiş impuls + 52W high tavanı.
    - TP1 karar noktası, TP2 taşıma alanı.
    """
    entry = float(entry)
    stop = float(stop)
    close = float(close)
    atr14 = float(atr14)

    debug = {}
    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(close) and np.isfinite(atr14)):
        tp1 = entry * 1.05
        tp2 = entry * 1.10
        return tp1, tp2, float("nan"), "LOW", {"reason": "NaN input"}

    risk = entry - stop
    if risk <= 0:
        tp1 = entry * 1.05
        tp2 = entry * 1.10
        return tp1, tp2, float("nan"), "LOW", {"reason": "risk<=0"}

    atr_pct = atr14 / close if close > 0 else float("nan")
    atr_pct = clamp(atr_pct, 0.012, 0.085)  # 1.2%..8.5%

    capacity = _trend_capacity_level(setup_score, ema50, ema150, ema200, ema200_slope, rsi14, close)

    # capacity -> step count N and multiplier
    if capacity == "HIGH":
        N = 5.5
        mult = 1.30
    elif capacity == "MID":
        N = 4.5
        mult = 1.10
    else:
        N = 3.5
        mult = 0.95

    # ATR-based “expected move”
    expected_move_pct = (atr_pct * N * mult) * 100.0  # percent

    # impulse cap (percent)
    impulse_cap_pct = _impulse_cap_pct_from_history(df_for_impulse, lookback=90)
    if np.isfinite(impulse_cap_pct) and impulse_cap_pct > 0:
        # TP2 cap: geçmişin %90'ı (kendini kandırmayı engeller)
        expected_move_pct = min(expected_move_pct, impulse_cap_pct * 0.90)

    # Risk floors: TP çok yakın olmasın (ama uçmasın)
    tp1_floor = entry + 2.2 * risk
    tp2_floor = entry + 3.5 * risk

    # Base targets from expected move
    tp1 = entry * (1.0 + (expected_move_pct / 100.0) * 0.55)
    tp2 = entry * (1.0 + (expected_move_pct / 100.0) * 0.90)

    tp1 = max(tp1, tp1_floor)
    tp2 = max(tp2, tp2_floor)

    # Absolute sanity caps (avoid fantasy)
    tp1 = min(tp1, entry * 1.18)
    tp2 = min(tp2, entry * 1.28)

    # 52W high ceiling for TP2 (Minervini disiplin)
    if np.isfinite(high_52w) and high_52w > 0:
        tp2 = min(tp2, high_52w * 0.98)

    # Keep separation
    if tp2 <= tp1:
        tp2 = tp1 * 1.06

    debug = {
        "capacity": capacity,
        "atr_pct": atr_pct * 100.0,
        "N": N,
        "mult": mult,
        "expected_move_pct": expected_move_pct,
        "impulse_cap_pct": impulse_cap_pct,
        "tp1_floor_2_2R": tp1_floor,
        "tp2_floor_3_5R": tp2_floor,
        "high_52w": high_52w,
    }

    return float(tp1), float(tp2), float(expected_move_pct), capacity, debug


# =========================================================
# SCORING / PLAN
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
    total_score: int
    label: str

    setup_score: int
    timing_score: int
    status_tag: str

    entry_low: float
    entry_high: float
    entry_mid: float

    stop: float
    tp1: float
    tp2: float
    rr_tp1: float
    rr_tp2: float

    dist_to_entry_pct: float
    watch_level: float

    # Minervini rule 5
    low_52w: float
    high_52w: float
    minervini5_ok: bool

    # targets meta
    capacity_level: str
    expected_move_pct: float
    targets_reason: str

    # Text
    narrative: str
    scenario: str
    debug: dict
    breakdown: ScoreBreakdown


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
    return (atr_pct < 2.0) and (45 <= rsi14 <= 55)


def _status_tag(
    timing_score: int,
    setup_score: int,
    trend_broken: bool,
    is_extended: bool,
    in_entry: bool,
    consolidation: bool,
    minervini5_ok: bool,
) -> str:
    # Hard veto first (Minervini rule 5)
    if not minervini5_ok:
        return "🟣 52W DİP FİLTRESİ (ZAYIF)"
    if trend_broken or setup_score < 45:
        return "🔴 TREND BOZULDU"
    if consolidation:
        return "🔵 KONSOLİDASYON"
    if in_entry and timing_score >= 70:
        return "🟢 ALIM BÖLGESİNDE"
    if is_extended and timing_score < 50:
        return "⚫ UZAMIŞ — KOVALAMA"
    return "🟡 PULLBACK BEKLENİYOR"


def build_trade_plan(df: pd.DataFrame, low_52w: float, high_52w: float) -> TradePlan:
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

    price_above_ema150 = close >= ema150
    price_near_ema150 = close >= ema150 * 0.98

    extended = dist_ema50_pct > 8.0
    trend_broken = (close < ema200) or (not long_trend_ok and not trend_stack_ok)

    # Minervini #5 (52w low +%25)
    m5_ok = minervini_rule5_ok(close, low_52w)

    # Legacy total (0-100) — keep structure, add rule5 as veto (not as points)
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

    # if minervini fails, label down
    if not m5_ok:
        total = min(total, 55)

    label = label_from_total(total)
    breakdown = ScoreBreakdown(
        trend_stack=trend_pts,
        price_vs_ema150=p_pts,
        momentum_rsi=m_pts,
        volatility_atr=v_pts,
        extension_vs_ema50=e_pts,
    )

    # Entry zone (EMA20-EMA50)
    entry_low = float(min(ema20, ema50))
    entry_high = float(max(ema20, ema50))
    entry_mid = (entry_low + entry_high) / 2.0

    # Stop: EMA50 vs ATR (tightest)
    stop_ema = ema50 * 0.995
    stop_atr = entry_mid - 1.2 * atr14
    stop = float(max(stop_ema, stop_atr))
    if stop >= entry_mid:
        stop = float(entry_mid * 0.99)

    # V4 split scores
    setup_raw = trend_pts + p_pts + m_pts + v_pts  # max 85
    setup_score = int(round(100 * setup_raw / 85)) if setup_raw > 0 else 0

    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)       # 0..60
    ext_pts = _extension_points(extended)              # 0..40
    timing_score = int(ext_pts + prox_pts)             # 0..100

    in_entry = (entry_low <= close <= entry_high)
    consolidation = _detect_consolidation(atr_pct, rsi14)

    status_tag = _status_tag(
        timing_score=timing_score,
        setup_score=setup_score,
        trend_broken=trend_broken,
        is_extended=extended,
        in_entry=in_entry,
        consolidation=consolidation,
        minervini5_ok=m5_ok,
    )

    watch_level = float(entry_high)

    # TP1/TP2 (Minervini-aligned targets)
    tp1, tp2, expected_move_pct, cap_level, targets_debug = compute_tp1_tp2_minervini(
        df_for_impulse=df,   # same timeframe; for better realism you can feed daily df, but keep simple & consistent
        entry=entry_mid,
        stop=stop,
        close=close,
        atr14=atr14,
        setup_score=setup_score,
        ema50=ema50,
        ema150=ema150,
        ema200=ema200,
        ema200_slope=ema200_slope,
        rsi14=rsi14,
        high_52w=high_52w,
    )

    risk = entry_mid - stop
    rr_tp1 = (tp1 - entry_mid) / risk if risk > 0 else float("nan")
    rr_tp2 = (tp2 - entry_mid) / risk if risk > 0 else float("nan")

    # Narrative + scenario
    trend_text = (
        "güçlü" if (trend_stack_ok and (price_above_ema150 or price_near_ema150))
        else ("zayıf" if close < ema200 else "karışık")
    )
    mom_text = "sağlıklı" if 55 <= rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf/sınır")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_border else "yüksek")

    if status_tag.startswith("🟢"):
        timing_cmd = "ALIM ARANIR"
    elif status_tag.startswith(("🟡", "🔵")):
        timing_cmd = "BEKLE / İZLE"
    else:
        timing_cmd = "UZAK DUR / ŞARTLAR OLUŞSUN"

    if status_tag.startswith("🟢"):
        scenario = (
            "Senaryo: Fiyat giriş bandında (EMA20–EMA50). Bu bölgede satış baskısı zayıflayıp küçük gövdeli mumlar + "
            "hacim düşüşü ile sıkışma görülürse, trend yönünde devam denemesi yapılabilir. Stop altına sarkarsa iptal."
        )
    elif status_tag.startswith("🟡"):
        scenario = (
            "Senaryo: Fiyat şu an giriş bandının dışında. EMA20–EMA50 bandına geri çekilme + hacimde düşüş ile "
            "konsolidasyon beklenir. Bu gerçekleşmeden yapılan alım kovalamaya girer."
        )
    elif status_tag.startswith("🔵"):
        scenario = (
            "Senaryo: Düşük volatilite ile yatay sıkışma var. Kırılımı takip et: güçlü kapanış + hacim artışı gelirse "
            "setup aktifleşir; aksi halde zaman kaybı."
        )
    elif status_tag.startswith("⚫"):
        scenario = (
            "Senaryo: Fiyat EMA50’ye göre uzamış. Pullback gelmeden giriş riskli. En iyi plan: giriş bandına yaklaşmasını "
            "bekle ve orada güç işareti (higher low / güçlü kapanış) ara."
        )
    elif status_tag.startswith("🟣"):
        scenario = (
            "Senaryo: Minervini #5 filtresi geçmiyor (fiyat 52W dip +%25 üstünde değil). Dipten yeni çıkan zayıf yapı olabilir. "
            "Önce güç kanıtı (trend + fiyat aksiyonu) gelmeden swing setup yok."
        )
    else:
        scenario = (
            "Senaryo: Trend filtresi bozulmuş. Önce yeniden EMA150/EMA200 üstüne dönüş ve ortalamaların toparlanması gerekir; "
            "aksi halde swing setup yok."
        )

    targets_reason = (
        f"Targets: kapasite={cap_level}, beklenen taşıma ≈ %{expected_move_pct:.1f} "
        f"(ATR/impuls/52W tavanı ile sınırlandı)."
    )

    narrative = (
        f"**Güncel Fiyat:** {close:.2f}  \n"
        f"**Toplam Skor:** {int(total)}/100 → **{label}**  \n"
        f"**Setup Kalitesi:** {setup_score}/100  |  **Zamanlama Skoru:** {timing_score}/100  \n"
        f"**Durum:** {status_tag}  \n\n"
        f"EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | EMA150: {ema150:.2f} | EMA200: {ema200:.2f}  \n"
        f"**Trend:** {trend_text} (EMA200 eğim={ema200_slope:.4f})  \n"
        f"**Fiyat Konumu:** EMA150 uzaklık %{dist_ema150_pct:.2f}  \n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}  \n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}  \n"
        f"**Uzama (EMA50 mesafe):** %{dist_ema50_pct:.2f} → {'uzamış' if extended else 'normal'}  \n\n"
        f"**Minervini #5:** 52W dip={low_52w:.2f} → {'✅ geçiyor' if m5_ok else '❌ geçmiyor'}  \n\n"
        f"**Zamanlama:** **{timing_cmd}**  \n"
        f"**Giriş Bölgesi:** {entry_low:.2f} – {entry_high:.2f}  \n"
        f"**Giriş Bölgesine Mesafe:** {dist_entry_pct:+.2f}%  \n"
        f"**Takip Seviyesi:** {watch_level:.2f} (altına yaklaşınca yeniden değerlendir)  \n\n"
        f"**Stop:** {stop:.2f}  \n"
        f"**TP1:** {tp1:.2f}  (R/R≈1:{rr_tp1:.2f})  \n"
        f"**TP2:** {tp2:.2f}  (R/R≈1:{rr_tp2:.2f})  \n"
        f"{targets_reason}"
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
        "trend_broken": trend_broken,
        "momentum_ok": momentum_ok,
        "vol_ok": vol_ok,
        "extended": extended,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_mid": entry_mid,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "setup_score": setup_score,
        "timing_score": timing_score,
        "dist_to_entry_pct": dist_entry_pct,
        "status_tag": status_tag,
        "consolidation": consolidation,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "minervini5_ok": m5_ok,
        "targets_debug": targets_debug,
    }

    return TradePlan(
        total_score=int(total),
        label=label,
        setup_score=int(setup_score),
        timing_score=int(timing_score),
        status_tag=status_tag,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        entry_mid=float(entry_mid),
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        rr_tp1=float(rr_tp1),
        rr_tp2=float(rr_tp2),
        dist_to_entry_pct=float(dist_entry_pct),
        watch_level=float(watch_level),
        low_52w=float(low_52w) if np.isfinite(low_52w) else float("nan"),
        high_52w=float(high_52w) if np.isfinite(high_52w) else float("nan"),
        minervini5_ok=bool(m5_ok),
        capacity_level=str(cap_level),
        expected_move_pct=float(expected_move_pct) if np.isfinite(expected_move_pct) else float("nan"),
        targets_reason=targets_reason,
        narrative=narrative,
        scenario=scenario,
        debug=debug,
        breakdown=breakdown,
    )


# =========================================================
# PDF EXPORT
# =========================================================
def _wrap_lines(text: str, max_chars: int = 92):
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
    lh = 14

    def draw_line(txt, font="Helvetica", size=11, space=lh):
        nonlocal y
        c.setFont(font, size)
        c.drawString(x, y, txt)
        y -= space
        if y < 2.0 * cm:
            c.showPage()
            y = h - 2.0 * cm

    draw_line("Tek Hisse Teknik Analiz Raporu (V5.0)", font="Helvetica-Bold", size=16, space=18)
    draw_line(f"Ticker: {ticker}    Zaman: {interval_label}    Bar: {bars}", size=11)
    draw_line(f"Tarih: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", font="Helvetica", size=10)
    draw_line("", size=10, space=10)

    draw_line("Özet", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"Güncel Fiyat: {plan.debug.get('close', float('nan')):.2f}", size=11)
    draw_line(f"Toplam Skor: {plan.total_score}/100    Etiket: {plan.label}", size=11)
    draw_line(f"Setup: {plan.setup_score}/100    Timing: {plan.timing_score}/100", size=11)
    draw_line(f"Durum: {plan.status_tag}", size=11)
    draw_line(f"Minervini #5 (52W dip +%25): {'OK' if plan.minervini5_ok else 'FAIL'} | 52W dip={plan.low_52w:.2f} | 52W tepe={plan.high_52w:.2f}", size=10, space=12)
    draw_line(f"Giriş: {plan.entry_low:.2f} – {plan.entry_high:.2f}", size=11)
    draw_line(f"Stop: {plan.stop:.2f}", size=11)
    draw_line(f"TP1: {plan.tp1:.2f} (R/R≈1:{plan.rr_tp1:.2f})", size=11)
    draw_line(f"TP2: {plan.tp2:.2f} (R/R≈1:{plan.rr_tp2:.2f})", size=11)
    draw_line("", size=10, space=10)

    if quote and isinstance(quote, dict):
        draw_line("Quote (Anlık Özet)", font="Helvetica-Bold", size=13, space=16)
        for k in ["name", "exchange", "currency", "close", "price", "change", "percent_change", "previous_close"]:
            if k in quote:
                draw_line(f"{k}: {quote[k]}", size=10, space=12)
        draw_line("", size=10, space=10)

    draw_line("Senaryo", font="Helvetica-Bold", size=13, space=16)
    for ln in _wrap_lines(plan.scenario, max_chars=95):
        draw_line(ln, size=10, space=12)
    draw_line("", size=10, space=10)

    draw_line("Otomatik Teknik Yorum", font="Helvetica-Bold", size=13, space=16)
    plain = plan.narrative.replace("**", "").replace("  \n", "\n")
    for ln in _wrap_lines(plain, max_chars=95):
        draw_line(ln, size=10, space=12)

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
# PLOTTING
# =========================================================
def plot_chart(
    df: pd.DataFrame,
    symbol: str,
    plan: TradePlan,
    last_price_line: float,
    show_candles: bool,
    show_emas: bool,
    show_line: bool,
):
    # If all off, force line on
    if not (show_candles or show_emas or show_line):
        show_line = True

    fig = go.Figure()

    if show_candles:
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

    if show_line:
        fig.add_trace(go.Scatter(x=df["time"], y=df["close"], name="Fiyat (Close)", mode="lines"))

    if show_emas:
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema20"], name="EMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", mode="lines"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema150"], name="EMA150", mode="lines"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", mode="lines"))

    # Entry/Stop/TP overlays
    fig.add_hrect(
        y0=plan.entry_low,
        y1=plan.entry_high,
        opacity=0.12,
        line_width=0,
        annotation_text="ENTRY",
        annotation_position="top left",
    )
    fig.add_hline(y=plan.stop, line_dash="dash", annotation_text="STOP", annotation_position="bottom left")
    fig.add_hline(y=plan.tp1, line_dash="dash", annotation_text="TP1", annotation_position="top left")
    fig.add_hline(y=plan.tp2, line_dash="dash", annotation_text="TP2", annotation_position="top left")
    fig.add_hline(y=float(last_price_line), line_dash="dot", annotation_text="GÜNCEL", annotation_position="top right")

    fig.update_layout(
        title=f"{symbol} — Grafik + EMA + Trade Levels",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =========================================================
# PORTFOLIO HELPERS
# =========================================================
def compute_rr(price: float, stop: float, tp: float) -> float:
    if not (np.isfinite(price) and np.isfinite(stop) and np.isfinite(tp)):
        return np.nan
    risk = price - stop
    reward = tp - price
    if risk <= 0:
        return np.nan
    return reward / risk


def held_action_comment(
    plan: TradePlan,
    price: float,
    avg_cost: float,
    user_stop: float,
    user_tp1: float,
    user_tp2: float,
) -> Tuple[str, str]:
    """
    Elde olan hisse için:
      action_tag, comment
    """
    if not np.isfinite(price):
        return "BİLİNEMİYOR", "Fiyat alınamadı."

    # Hard risk: price below stop
    if np.isfinite(user_stop) and price <= user_stop:
        return "SAT/STOP", "Fiyat stop altı → pozisyon sonlandır (disiplin)."

    # Minervini veto for new adds (held: 'ekleme yok')
    if not plan.minervini5_ok:
        return "TUT/İZLE", "Minervini #5 geçmiyor → ekleme yok; trend kanıtı bekle."

    # Trend broken
    if plan.status_tag.startswith("🔴"):
        return "RİSK AZALT", "Trend bozuk → ekleme yok; stopu sıkılaştır / azaltmayı değerlendir."

    # Extended
    if plan.status_tag.startswith("⚫"):
        return "TUT", "Uzamış → ekleme yok; pullback ile yeniden değerlendir."

    # Consolidation
    if plan.status_tag.startswith("🔵"):
        return "TUT/İZLE", "Sıkışma → kırılım veya bozulma gelene kadar sabır."

    # Profit context
    pnl_pct = pct(price, avg_cost) if np.isfinite(avg_cost) and avg_cost > 0 else np.nan
    if np.isfinite(user_tp1) and price >= user_tp1:
        # TP1 reached: suggest tighten stop
        return "STOP YUKARI", "TP1 bölgesi → stopu yukarı çekerek trend takip et (satış değil, yönetim)."
    if np.isfinite(user_tp2) and price >= user_tp2:
        return "KARAR NOKTASI", "TP2 bölgesi → momentum bozulursa kısmi/çıkış; korunuyorsa trailing stop."

    # If close to stop, warn
    if np.isfinite(user_stop) and (price - user_stop) / price < 0.03:
        return "DİKKAT", "Stop çok yakın → oynaklık stoplatabilir. (Gevşetme yok; pozisyon boyunu düşün.)"

    # Default: hold
    return "TUT", "Koşullar fena değil → plana sadık kal; ekleme için giriş bandı ve timing bekle."


# =========================================================
# SIDEBAR (Global)
# =========================================================
with st.sidebar:
    st.header("Genel Ayarlar")

    default_interval_label = st.selectbox("Varsayılan zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=list(INTERVAL_MAP.keys()).index(DEFAULT_SINGLE_INTERVAL_LABEL))
    bars = st.slider("Bar sayısı", min_value=120, max_value=800, value=300, step=10)

    st.divider()
    st.subheader("Grafik Görünümü")
    show_candles = st.checkbox("Mum (Candlestick) göster", value=True)
    show_emas = st.checkbox("EMA'ları göster", value=True)
    show_line = st.checkbox("Fiyat çizgisi (Close) göster", value=True)
    st.caption("Hepsini kapatırsan çizgi otomatik açılır.")

    st.divider()
    show_quote = st.checkbox("Quote (anlık fiyat) kullan", value=False)
    st.caption("Quote açarsan +1 API çağrısı (ticker başına). Free planda pre/post-market genelde yok.")

    st.divider()
    st.subheader("📌 Tek Hisse Test Hafızası (Oturum)")
    if st.session_state.daily_tests:
        df_mem = pd.DataFrame(st.session_state.daily_tests)
        show_cols = [
            "timestamp", "ticker", "timeframe", "price",
            "setup_score", "timing_score", "total_score", "status_tag",
            "minervini5_ok", "tp1", "tp2"
        ]
        show_cols = [c for c in show_cols if c in df_mem.columns]
        st.dataframe(df_mem[show_cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Henüz tek hisse testi yok.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Oturumu Temizle", use_container_width=True):
            clear_today_session()
            st.rerun()
    with col_b:
        hist_bytes = history_csv_bytes()
        st.download_button(
            "history.csv indir",
            data=hist_bytes if hist_bytes else b"",
            file_name="history.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(not bool(hist_bytes)),
        )

    with st.expander("📚 Geçmiş (history.csv)"):
        hist_df = read_history_df()
        if hist_df.empty:
            st.info("history.csv yok veya boş.")
        else:
            st.dataframe(hist_df.tail(200), use_container_width=True, hide_index=True)


# =========================================================
# MAIN TABS
# =========================================================
tab_single, tab_portfolio = st.tabs(["📈 Tek Hisse Analiz", "🧳 Portföy Analiz"])


# =========================================================
# TAB 1: SINGLE STOCK
# =========================================================
with tab_single:
    left, right = st.columns([0.36, 0.64], vertical_alignment="top")

    with left:
        st.subheader("Hisse")
        ticker = st.text_input("Ticker", placeholder="Örn: NVDA, TSLA, PLTR").strip().upper()
        interval_label = st.selectbox(
            "Zaman çözünürlüğü",
            list(INTERVAL_MAP.keys()),
            index=list(INTERVAL_MAP.keys()).index(default_interval_label),
        )
        run = st.button("Getir & Analiz Et", type="primary", use_container_width=True)

        if run:
            if not ticker:
                st.warning("Ticker gir.")
            else:
                interval = INTERVAL_MAP[interval_label]
                with st.spinner("Veri çekiliyor..."):
                    try:
                        payload = td_time_series(ticker, interval, bars)
                        df = parse_ohlcv(payload)
                    except Exception as e:
                        st.error(f"Veri alınamadı: {e}")
                        st.stop()

                # Indicators
                df["ema20"] = ema(df["close"], 20)
                df["ema50"] = ema(df["close"], 50)
                df["ema150"] = ema(df["close"], 150)
                df["ema200"] = ema(df["close"], 200)
                df["rsi14"] = rsi(df["close"], 14)
                df["atr14"] = atr(df, 14)

                # 52W data (daily)
                low_52w, high_52w = (float("nan"), float("nan"))
                try:
                    low_52w, high_52w = fetch_daily_52w(ticker)
                except Exception:
                    pass

                plan = build_trade_plan(df, low_52w=low_52w, high_52w=high_52w)

                q = {}
                quote_price = None
                if show_quote:
                    try:
                        q = td_quote(ticker)
                        for key in ["price", "close"]:
                            if key in q:
                                try:
                                    quote_price = float(q[key])
                                    break
                                except Exception:
                                    pass
                    except Exception:
                        q = {}

                candle_close = float(df.iloc[-1]["close"])
                last_price_line = quote_price if (quote_price is not None and np.isfinite(quote_price)) else candle_close

                # Save for chart
                st.session_state["__df"] = df
                st.session_state["__ticker"] = ticker
                st.session_state["__plan"] = plan
                st.session_state["__quote"] = q
                st.session_state["__last_price_line"] = float(last_price_line)
                st.session_state["__interval_label"] = interval_label
                st.session_state["__bars"] = bars

                # Memory record (session + csv)
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "timeframe": interval,
                    "price": round(float(last_price_line), 4),
                    "setup_score": int(plan.setup_score),
                    "timing_score": int(plan.timing_score),
                    "total_score": int(plan.total_score),
                    "status_tag": plan.status_tag,
                    "minervini5_ok": bool(plan.minervini5_ok),
                    "entry_low": round(float(plan.entry_low), 4),
                    "entry_high": round(float(plan.entry_high), 4),
                    "stop": round(float(plan.stop), 4),
                    "tp1": round(float(plan.tp1), 4),
                    "tp2": round(float(plan.tp2), 4),
                    "rr_tp1": round(float(plan.rr_tp1), 4) if np.isfinite(plan.rr_tp1) else "",
                    "rr_tp2": round(float(plan.rr_tp2), 4) if np.isfinite(plan.rr_tp2) else "",
                }
                st.session_state.daily_tests.append(record)
                try:
                    save_to_history(record)
                except Exception as e:
                    st.warning(f"history.csv yazılamadı: {e}")

                # UI outputs
                st.divider()
                st.subheader("📊 Strateji Özeti (Görünür)")
                colm1, colm2 = st.columns(2)
                with colm1:
                    st.metric("Güncel Fiyat", f"{float(last_price_line):.2f}")
                    st.metric("Durum", plan.status_tag)
                with colm2:
                    st.metric("Toplam Skor", f"{plan.total_score} / 100")
                    st.metric("Setup / Timing", f"{plan.setup_score} / {plan.timing_score}")

                st.caption(f"Minervini #5: 52W dip {plan.low_52w:.2f} → {'✅ geçiyor' if plan.minervini5_ok else '❌ geçmiyor'}")

                st.subheader("📌 İşlem Planı")
                table = pd.DataFrame(
                    {
                        "Parametre": ["Giriş Bölgesi", "Giriş Mesafesi", "Stop", "TP1", "TP2", "R/R (TP1)", "R/R (TP2)"],
                        "Değer": [
                            f"{plan.entry_low:.2f} – {plan.entry_high:.2f}",
                            f"{plan.dist_to_entry_pct:+.2f}%",
                            f"{plan.stop:.2f}",
                            f"{plan.tp1:.2f}",
                            f"{plan.tp2:.2f}",
                            f"1 : {plan.rr_tp1:.2f}" if np.isfinite(plan.rr_tp1) else "—",
                            f"1 : {plan.rr_tp2:.2f}" if np.isfinite(plan.rr_tp2) else "—",
                        ],
                    }
                )
                st.table(table)

                st.subheader("🧠 Skor Dağılımı (Legacy)")
                b = plan.breakdown
                bdf = pd.DataFrame(
                    {
                        "Bileşen": ["Trend", "Fiyat/EMA150", "Momentum (RSI)", "Volatilite (ATR%)", "Uzama (EMA50)"],
                        "Puan": [b.trend_stack, b.price_vs_ema150, b.momentum_rsi, b.volatility_atr, b.extension_vs_ema50],
                        "Maks": [30, 20, 20, 15, 15],
                    }
                )
                st.table(bdf)

                st.subheader("🧭 Senaryo")
                st.write(plan.scenario)

                st.subheader("📝 Otomatik Teknik Yorum")
                st.markdown(plan.narrative)

                if show_quote and q:
                    st.subheader("⚡ Quote (Anlık Özet)")
                    keys = ["symbol", "name", "exchange", "currency", "price", "close", "change", "percent_change", "previous_close"]
                    compact = {k: q[k] for k in keys if k in q}
                    st.write(compact)
                    if np.isfinite(quote_price) and np.isfinite(candle_close) and abs(quote_price - candle_close) < 1e-9:
                        st.info("Piyasa kapalıysa quote son kapanışı gösterebilir (pre/post-market yok).")

                # ---------- İşlem Yönetimi (FORM) ----------
                st.subheader("🧩 İşlem Yönetimi (Eldeki Hisse)")
                st.caption("Buradaki alanlar yazarken sayfa yenilense bile form gönderene kadar değerler korunur. Stop asla gevşetilmez.")

                # init mgmt state
                if ticker not in st.session_state.trade_mgmt:
                    st.session_state.trade_mgmt[ticker] = {
                        "entry": float(plan.entry_mid),
                        "stop": float(plan.stop),
                        "tp1": float(plan.tp1),
                        "tp2": float(plan.tp2),
                    }

                mg = st.session_state.trade_mgmt[ticker].copy()

                with st.form(key=f"mgmt_form_{ticker}", clear_on_submit=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        entry_in = st.number_input("Entry (maliyet/plan giriş)", value=float(mg.get("entry", plan.entry_mid)), step=0.01, format="%.2f")
                        stop_in = st.number_input("Stop (mevcut)", value=float(mg.get("stop", plan.stop)), step=0.01, format="%.2f")
                    with c2:
                        tp1_in = st.number_input("TP1 (mevcut)", value=float(mg.get("tp1", plan.tp1)), step=0.01, format="%.2f")
                        tp2_in = st.number_input("TP2 (mevcut)", value=float(mg.get("tp2", plan.tp2)), step=0.01, format="%.2f")

                    submitted = st.form_submit_button("Kaydet / Güncelle", use_container_width=True)

                if submitted:
                    # Stop gevşetme yok
                    old_stop = float(mg.get("stop", plan.stop))
                    new_stop = float(stop_in)
                    if new_stop < old_stop:
                        st.warning("Stop geri çekilemez. Eski stop korunuyor.")
                        new_stop = old_stop

                    st.session_state.trade_mgmt[ticker] = {
                        "entry": float(entry_in),
                        "stop": float(new_stop),
                        "tp1": float(tp1_in),
                        "tp2": float(tp2_in),
                    }
                    st.success("İşlem yönetimi değerleri kaydedildi.")

                # Suggestion block (non-destructive)
                mg = st.session_state.trade_mgmt[ticker]
                cur_price = float(last_price_line)
                entry0 = float(mg["entry"])
                stop0 = float(mg["stop"])
                tp1_0 = float(mg["tp1"])
                tp2_0 = float(mg["tp2"])

                # Basic trailing suggestions
                suggestions = []
                # If +5% moved, consider raising stop toward entry
                if np.isfinite(cur_price) and np.isfinite(entry0) and entry0 > 0:
                    move_pct = (cur_price - entry0) / entry0 * 100.0
                    if move_pct >= 5.0:
                        sug_stop = max(stop0, entry0)  # break-even
                        suggestions.append(f"Fiyat entry'ye göre %+{move_pct:.1f}. Stop’u en az **break-even** seviyesine çekmeyi düşünebilirsin: {sug_stop:.2f}")

                if np.isfinite(tp1_0) and cur_price >= tp1_0:
                    # After TP1: trail stop to EMA20 or EMA50 (whichever higher but not above price)
                    ema20_now = float(df.iloc[-1]["ema20"])
                    ema50_now = float(df.iloc[-1]["ema50"])
                    trail = max(stop0, min(cur_price * 0.995, max(ema20_now, ema50_now) * 0.995))
                    suggestions.append(f"TP1 bölgesi: stop’u **EMA bazlı** yukarı taşıyabilirsin (gevşetme yok): {trail:.2f}")

                if np.isfinite(tp2_0) and cur_price >= tp2_0:
                    suggestions.append("TP2 bölgesi: Momentum bozulursa kısmi/çıkış; korunuyorsa trailing stop ile devam.")

                if suggestions:
                    st.info("**Yönetim Önerisi (otomatik):**\n\n- " + "\n- ".join(suggestions))
                else:
                    st.caption("Yönetim önerisi için: fiyatın entry/TP seviyelerine yaklaşmasını bekle.")

                # ---------- Report ----------
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
                    file_name=f"{ticker}_{interval}_rapor.pdf",
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
            last_price_line = float(st.session_state.get("__last_price_line", float(df.iloc[-1]["close"])))
            fig = plot_chart(df, ticker, plan, last_price_line, show_candles, show_emas, show_line)
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2: PORTFOLIO
# =========================================================
with tab_portfolio:
    st.subheader("🧳 Portföy Analiz")
    st.caption("Portföy satırlarını gir: ticker, adet, alış ort., stop, TP1, TP2. Sonra analiz et → eldeki pozisyon diliyle yorum + risk tablosu.")

    top_left, top_right = st.columns([0.65, 0.35], vertical_alignment="top")

    with top_right:
        st.markdown("### Portföy Dosyası")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Portföyü Yükle (portfolio.csv)", use_container_width=True):
                st.session_state.portfolio = load_portfolio_df()
                st.rerun()
        with col2:
            st.download_button(
                "portfolio.csv indir",
                data=portfolio_csv_bytes() if portfolio_csv_bytes() else b"",
                file_name="portfolio.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=(not bool(portfolio_csv_bytes())),
            )

        st.markdown("### Hızlı İşlemler")
        if st.button("Portföyü Kaydet", type="primary", use_container_width=True):
            try:
                save_portfolio_df(st.session_state.portfolio)
                st.success("portfolio.csv kaydedildi.")
            except Exception as e:
                st.error(f"Kaydedilemedi: {e}")

        if st.button("Portföyü Temizle", use_container_width=True):
            st.session_state.portfolio = pd.DataFrame(columns=["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"])
            try:
                save_portfolio_df(st.session_state.portfolio)
            except Exception:
                pass
            st.rerun()

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
                "tp2": st.column_config.NumberColumn("TP2", min_value=0.0, step=0.01, format="%.2f"),
            },
        )

        st.markdown("### Analiz")
        interval_label_pf = st.selectbox(
            "Portföy analiz zaman dilimi",
            list(INTERVAL_MAP.keys()),
            index=list(INTERVAL_MAP.keys()).index("Günlük (1day)"),
            key="pf_interval"
        )
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
                interval = INTERVAL_MAP[interval_label_pf]
                rows = []

                with st.spinner("Portföy verileri çekiliyor ve analiz ediliyor..."):
                    for _, r in dfp.iterrows():
                        tkr = str(r.get("ticker", "")).upper().strip()
                        if not tkr:
                            continue

                        qty = safe_float(r.get("qty"))
                        avg_cost = safe_float(r.get("avg_cost"))
                        user_stop = safe_float(r.get("stop"))
                        user_tp1 = safe_float(r.get("tp1"))
                        user_tp2 = safe_float(r.get("tp2"))

                        try:
                            payload = td_time_series(tkr, interval, bars)
                            dfi = parse_ohlcv(payload)
                            dfi["ema20"] = ema(dfi["close"], 20)
                            dfi["ema50"] = ema(dfi["close"], 50)
                            dfi["ema150"] = ema(dfi["close"], 150)
                            dfi["ema200"] = ema(dfi["close"], 200)
                            dfi["rsi14"] = rsi(dfi["close"], 14)
                            dfi["atr14"] = atr(dfi, 14)

                            # 52W data
                            low_52w, high_52w = (float("nan"), float("nan"))
                            try:
                                low_52w, high_52w = fetch_daily_52w(tkr)
                            except Exception:
                                pass

                            plan = build_trade_plan(dfi, low_52w=low_52w, high_52w=high_52w)

                            candle_close = float(dfi.iloc[-1]["close"])
                            price = candle_close

                            # optional quote (extra API call)
                            if show_quote:
                                try:
                                    q = td_quote(tkr)
                                    if "price" in q:
                                        price = float(q["price"])
                                except Exception:
                                    pass

                            pnl_pct = pct(price, avg_cost) if np.isfinite(avg_cost) and avg_cost > 0 else np.nan
                            dist_stop_pct = pct(price, user_stop) if np.isfinite(user_stop) and user_stop > 0 else np.nan
                            dist_tp1_pct = pct(user_tp1, price) if np.isfinite(user_tp1) and user_tp1 > 0 else np.nan
                            dist_tp2_pct = pct(user_tp2, price) if np.isfinite(user_tp2) and user_tp2 > 0 else np.nan
                            rr_user_tp1 = compute_rr(price, user_stop, user_tp1)
                            rr_user_tp2 = compute_rr(price, user_stop, user_tp2)

                            action, comment = held_action_comment(plan, price, avg_cost, user_stop, user_tp1, user_tp2)

                            pos_value = (qty * price) if np.isfinite(qty) and np.isfinite(price) else np.nan
                            risk_per_share = (price - user_stop) if (np.isfinite(user_stop) and np.isfinite(price)) else np.nan
                            risk_value = (risk_per_share * qty) if (np.isfinite(risk_per_share) and np.isfinite(qty)) else np.nan

                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": round(price, 2),
                                "Alış Ort.": round(avg_cost, 2) if np.isfinite(avg_cost) else "",
                                "P&L %": round(pnl_pct, 2) if np.isfinite(pnl_pct) else "",
                                "Stop": round(user_stop, 2) if np.isfinite(user_stop) else "",
                                "Stop Mesafe %": round(dist_stop_pct, 2) if np.isfinite(dist_stop_pct) else "",
                                "TP1": round(user_tp1, 2) if np.isfinite(user_tp1) else "",
                                "TP1 Mesafe %": round(dist_tp1_pct, 2) if np.isfinite(dist_tp1_pct) else "",
                                "TP2": round(user_tp2, 2) if np.isfinite(user_tp2) else "",
                                "TP2 Mesafe %": round(dist_tp2_pct, 2) if np.isfinite(dist_tp2_pct) else "",
                                "R (TP1/Stop)": round(rr_user_tp1, 2) if np.isfinite(rr_user_tp1) else "",
                                "R (TP2/Stop)": round(rr_user_tp2, 2) if np.isfinite(rr_user_tp2) else "",
                                "Setup": plan.setup_score,
                                "Timing": plan.timing_score,
                                "Durum": plan.status_tag,
                                "Minervini #5": "OK" if plan.minervini5_ok else "FAIL",
                                "Auto Stop": round(plan.stop, 2),
                                "Auto TP1": round(plan.tp1, 2),
                                "Auto TP2": round(plan.tp2, 2),
                                "Aksiyon": action,
                                "Not": comment
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
                                "TP2": "",
                                "TP2 Mesafe %": "",
                                "R (TP1/Stop)": "",
                                "R (TP2/Stop)": "",
                                "Setup": "",
                                "Timing": "",
                                "Durum": "HATA",
                                "Minervini #5": "",
                                "Auto Stop": "",
                                "Auto TP1": "",
                                "Auto TP2": "",
                                "Aksiyon": "HATA",
                                "Not": f"Veri/analiz hatası: {e}"
                            })

                out = pd.DataFrame(rows)

                st.markdown("### Sonuç Tablosu")
                st.dataframe(out, use_container_width=True, hide_index=True)

                st.markdown("### Hızlı Özet")
                if not out.empty:
                    a = out[out["Durum"].astype(str).str.startswith("🟢")]
                    b = out[out["Durum"].astype(str).str.startswith("🟡")]
                    c = out[out["Durum"].astype(str).str.startswith("⚫")]
                    d = out[out["Durum"].astype(str).str.startswith("🔴")]
                    e = out[out["Durum"].astype(str).str.startswith("🟣")]

                    colx, coly, colz, colw, colv = st.columns(5)
                    colx.metric("🟢 Alım Bölgesi", len(a))
                    coly.metric("🟡 Pullback", len(b))
                    colz.metric("⚫ Uzamış", len(c))
                    colw.metric("🔴 Trend Bozuk", len(d))
                    colv.metric("🟣 52W Filtresi", len(e))

                # Download results
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Portföy Analiz CSV indir",
                    data=csv_bytes,
                    file_name="portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
