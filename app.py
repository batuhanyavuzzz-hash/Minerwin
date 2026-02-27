# app.py
# Tek Hisse + Portföy Analiz (V5.2) — Twelve Data
# - Minervini #5: Güncel fiyat, 52W low’un en az %25 üstünde olmalı (veto/etiket)
# - TP1 + TP2: Minervini uyumlu hedefler (taşıma kapasitesi + geçmiş impuls + 52W high tavanı)
# - STOP: “Invalidation + Noise” (swing low / EMA tabanı + ATR gürültü filtresi) + max risk limiti
# - Eldeki hisse dili: TUT / STOP YUKARI / RİSK AZALT / SAT-STOP
# - Grafik: mum + EMA + fiyat çizgisi (toggle; hepsi kapanırsa çizgi zorunlu kalır)
# - İşlem yönetimi: form ile inputlar stabilize (yazarken state sıçraması azalır)
# - Portföy: Şık PDF + Şık Excel (XLSX) çıktı + portföy özet metrikleri
#
# Notlar:
# - Twelve Data free plan çoğu zaman pre/post-market quote sağlamaz; piyasa kapalıyken quote kapanışı döndürebilir.
# - “Alım kararı” için ana timeframe: GÜNLÜK. Haftalık sadece bağlam filtresi olarak kullanılmalı.

import io
import os
import csv
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet

# Excel (openpyxl)
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import CellIsRule


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="MinerWin", layout="wide", initial_sidebar_state="expanded")


# =========================================================
# BRANDING / HEADER
# =========================================================
def _load_logo_b64_multi() -> str:
    """
    Logo kaybolmasın diye birkaç olası yolu dener:
    - minerwin_logo.png (root)
    - assets/minerwin_logo.png
    - static/minerwin_logo.png
    - src/minerwin_logo.png
    """
    candidates = [
        "minerwin_logo.png",
        os.path.join("assets", "minerwin_logo.png"),
        os.path.join("static", "minerwin_logo.png"),
        os.path.join("src", "minerwin_logo.png"),
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except Exception:
            continue
    return ""


def _find_logo_path() -> Optional[str]:
    candidates = [
        "minerwin_logo.png",
        os.path.join("assets", "minerwin_logo.png"),
        os.path.join("static", "minerwin_logo.png"),
        os.path.join("src", "minerwin_logo.png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


logo_b64 = _load_logo_b64_multi()

st.markdown(
    """
<style>
.block-container { padding-top: 3.2rem; }

.header {
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:6px;
}
.header-title { font-size:32px; font-weight:800; line-height:1; }
.sub-title { font-size:13px; color:#8b949e; margin-left:58px; margin-top:-6px; }
.logo { height:42px; }

.card{
  background:#161B22;
  border:1px solid #22262E;
  border-radius:14px;
  padding:16px 18px;
  margin-bottom:14px;
}
.small-muted{ color:#8b949e; font-size:12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header">
    {"<img class='logo' src='data:image/png;base64," + logo_b64 + "' />" if logo_b64 else ""}
    <div class="header-title">MinerWin</div>
</div>
<div class="sub-title">Minervini-Based Technical Trading Engine</div>
""",
    unsafe_allow_html=True,
)

st.divider()

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

st.caption(
    "Not: Twelve Data Free/BASIC plan genelde pre-market/after-hours fiyatı vermez; "
    "piyasa kapalıyken quote son kapanışı döndürebilir."
)

# =========================================================
# SESSION STATE INIT
# =========================================================
if "daily_tests" not in st.session_state:
    st.session_state.daily_tests = []

if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"])

if "trade_mgmt" not in st.session_state:
    st.session_state.trade_mgmt = {}  # dict[ticker] = {"entry":..., "stop":..., "tp1":..., "tp2":...}


# =========================================================
# BASIC HELPERS
# =========================================================
def safe_float(x) -> float:
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


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:+.2f}%"


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
# STOP ENGINE (INVALIDATION + NOISE)
# =========================================================
def _recent_pivot_low(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Son lookback bar içinde en son pivot low (lokal minimum) bul.
    Pivot tanımı: low[i] < low[i-1] ve low[i] < low[i+1]
    Bulunamazsa NaN.
    """
    if df is None or df.empty or "low" not in df.columns:
        return float("nan")
    d = df.tail(max(lookback + 2, 10)).reset_index(drop=True)
    lows = d["low"].astype(float).values
    if len(lows) < 5:
        return float("nan")

    pivots = []
    for i in range(1, len(lows) - 1):
        if np.isfinite(lows[i - 1]) and np.isfinite(lows[i]) and np.isfinite(lows[i + 1]):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                pivots.append((i, float(lows[i])))

    if not pivots:
        return float("nan")
    return float(pivots[-1][1])


def _noise_factor_from_atr_pct(atr_pct: float) -> float:
    """
    ATR% (yüzde) → noise_factor
    Sakin: <2%       => 1.25
    Normal: 2-4%     => 1.55
    Agresif: 4-6%    => 1.85
    Çok agresif: >6% => 2.15
    """
    if not np.isfinite(atr_pct):
        return 1.55
    if atr_pct < 2.0:
        return 1.25
    if atr_pct < 4.0:
        return 1.55
    if atr_pct < 6.0:
        return 1.85
    return 2.15


def compute_stop_invalidation_plus_noise(
    entry: float,
    ema50: float,
    atr14: float,
    atr_pct: float,
    pivot_low: float,
    max_risk_pct: float = 7.0,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Returns:
      (stop_active, stop_structural, stop_noise, debug)

    stop_structural: invalidation base (pivot/EMA)
    stop_noise:      ATR-based noise stop
    stop_active:     operational stop after applying room logic + risk cap

    Mantık:
    - stop_structural: pivot_low varsa pivot altı, yoksa EMA50 altı (toleranslı)
    - stop_noise: entry - ATR * noise_factor  (oynaklığı taşımak için)
    - stop_candidate: min(structural, noise)  => daha aşağıdaki (daha geniş) stop
    - risk cap: entry*(1-max_risk_pct) altına inemez (çok genişlemeyi engeller)
    """
    if not (np.isfinite(entry) and np.isfinite(ema50) and np.isfinite(atr14)) or entry <= 0:
        stop_fallback = float(entry * 0.93) if np.isfinite(entry) and entry > 0 else float("nan")
        return stop_fallback, float("nan"), float("nan"), {"reason": "NaN entry/ema50/atr14"}

    # --- Structural invalidation base ---
    inv_from_ema = float(ema50 * 0.995)  # EMA50 altına biraz tolerans
    inv_from_pivot = float(pivot_low * 0.995) if (np.isfinite(pivot_low) and pivot_low > 0) else float("nan")

    if np.isfinite(inv_from_pivot):
        stop_structural = float(min(inv_from_ema, inv_from_pivot))
        inv_src = "pivot_or_ema"
    else:
        stop_structural = float(inv_from_ema)
        inv_src = "ema50"

    # --- Noise stop (ATR room) ---
    nf = _noise_factor_from_atr_pct(atr_pct)
    stop_noise = float(entry - nf * atr14)

    # --- Operational room logic (wider stop to avoid whipsaw) ---
    stop_candidate = float(min(stop_structural, stop_noise))

    # --- Cap maximum risk (avoid too-wide) ---
    cap_stop = float(entry * (1.0 - max_risk_pct / 100.0))
    if stop_candidate < cap_stop:
        stop_active = cap_stop
        capped = True
    else:
        stop_active = stop_candidate
        capped = False

    # Safety: never >= entry
    if np.isfinite(stop_active) and stop_active >= entry:
        stop_active = float(entry * 0.99)

    dbg = {
        "inv_src": inv_src,
        "pivot_low": pivot_low,
        "stop_structural": stop_structural,
        "noise_factor": nf,
        "stop_noise": stop_noise,
        "stop_candidate": stop_candidate,
        "cap_stop": cap_stop,
        "capped": capped,
        "max_risk_pct": max_risk_pct,
    }
    return float(stop_active), float(stop_structural), float(stop_noise), dbg


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

    if np.isfinite(price) and np.isfinite(ema50) and price >= ema50:
        votes += 1

    if votes >= 6:
        return "HIGH"
    if votes >= 3:
        return "MID"
    return "LOW"


def _impulse_cap_pct_from_history(df: pd.DataFrame, lookback: int = 90) -> float:
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")

    c = df["close"].dropna().astype(float)
    if len(c) < max(lookback, 30):
        return float("nan")

    c = c.iloc[-lookback:].reset_index(drop=True)
    running_min = float(c.iloc[0])
    best = 0.0
    for i in range(1, len(c)):
        running_min = min(running_min, float(c.iloc[i - 1]))
        if running_min <= 0:
            continue
        move = (float(c.iloc[i]) - running_min) / running_min
        if move > best:
            best = move
    return float(best * 100.0)


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
    entry = float(entry)
    stop = float(stop)
    close = float(close)
    atr14 = float(atr14)

    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(close) and np.isfinite(atr14)):
        return entry * 1.06, entry * 1.12, float("nan"), "LOW", {"reason": "NaN input"}

    risk = entry - stop
    if risk <= 0:
        return entry * 1.06, entry * 1.12, float("nan"), "LOW", {"reason": "risk<=0"}

    atr_pct_ratio = atr14 / close if close > 0 else float("nan")
    atr_pct_ratio = clamp(atr_pct_ratio, 0.012, 0.085)  # 1.2%..8.5%

    capacity = _trend_capacity_level(setup_score, ema50, ema150, ema200, ema200_slope, rsi14, close)

    if capacity == "HIGH":
        N, mult = 5.5, 1.30
    elif capacity == "MID":
        N, mult = 4.5, 1.10
    else:
        N, mult = 3.5, 0.95

    expected_move_pct = (atr_pct_ratio * N * mult) * 100.0  # percent

    impulse_cap_pct = _impulse_cap_pct_from_history(df_for_impulse, lookback=90)
    if np.isfinite(impulse_cap_pct) and impulse_cap_pct > 0:
        expected_move_pct = min(expected_move_pct, impulse_cap_pct * 0.90)

    tp1_floor = entry + 2.2 * risk
    tp2_floor = entry + 3.5 * risk

    tp1 = entry * (1.0 + (expected_move_pct / 100.0) * 0.55)
    tp2 = entry * (1.0 + (expected_move_pct / 100.0) * 0.90)

    tp1 = max(tp1, tp1_floor)
    tp2 = max(tp2, tp2_floor)

    tp1 = min(tp1, entry * 1.18)
    tp2 = min(tp2, entry * 1.28)

    if np.isfinite(high_52w) and high_52w > 0:
        tp2 = min(tp2, high_52w * 0.98)

    if tp2 <= tp1:
        tp2 = tp1 * 1.06

    dbg = {
        "capacity": capacity,
        "atr_pct": atr_pct_ratio * 100.0,
        "N": N,
        "mult": mult,
        "expected_move_pct": expected_move_pct,
        "impulse_cap_pct": impulse_cap_pct,
        "tp1_floor_2_2R": tp1_floor,
        "tp2_floor_3_5R": tp2_floor,
        "high_52w": high_52w,
    }
    return float(tp1), float(tp2), float(expected_move_pct), capacity, dbg


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

    low_52w: float
    high_52w: float
    minervini5_ok: bool

    capacity_level: str
    expected_move_pct: float
    targets_reason: str

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

    m5_ok = minervini_rule5_ok(close, low_52w)

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

    entry_low = float(min(ema20, ema50))
    entry_high = float(max(ema20, ema50))
    entry_mid = float((entry_low + entry_high) / 2.0)

    setup_raw = trend_pts + p_pts + m_pts + v_pts  # max 85
    setup_score = int(round(100 * setup_raw / 85)) if setup_raw > 0 else 0

    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)
    ext_pts = _extension_points(extended)
    timing_score = int(ext_pts + prox_pts)

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

    pivot_low = _recent_pivot_low(df, lookback=20)
    stop, stop_structural, stop_noise, stop_dbg = compute_stop_invalidation_plus_noise(
        entry=entry_mid,
        ema50=ema50,
        atr14=atr14,
        atr_pct=atr_pct,
        pivot_low=pivot_low,
        max_risk_pct=7.0,
    )

    tp1, tp2, expected_move_pct, cap_level, targets_dbg = compute_tp1_tp2_minervini(
        df_for_impulse=df,
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

    stop_reason = (
        f"Stop (aktif): noise(ATR) + yapısal(invalidation:{stop_dbg.get('inv_src')}) + max_risk_cap "
        f"(capped={stop_dbg.get('capped')}). "
        f"Yapısal={stop_structural:.2f} | Noise={stop_noise:.2f}"
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
        f"**Takip Seviyesi:** {watch_level:.2f}  \n\n"
        f"**Stop:** {stop:.2f}  \n"
        f"**TP1:** {tp1:.2f}  (R/R≈1:{rr_tp1:.2f})  \n"
        f"**TP2:** {tp2:.2f}  (R/R≈1:{rr_tp2:.2f})  \n"
        f"{targets_reason}  \n"
        f"{stop_reason}"
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
        "pivot_low": pivot_low,
        "stop_debug": stop_dbg,
        "stop_structural": stop_structural,
        "stop_noise": stop_noise,
        "targets_debug": targets_dbg,
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
# PDF EXPORT — SINGLE
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


def build_pdf_bytes_single(
    ticker: str,
    interval_label: str,
    bars: int,
    plan: TradePlan,
    quote: dict | None,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("Tek Hisse Teknik Analiz Raporu (V5.2)", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Ticker:</b> {ticker} &nbsp;&nbsp; <b>Zaman:</b> {interval_label} &nbsp;&nbsp; <b>Bar:</b> {bars}", styles["Normal"]))
    story.append(Paragraph(f"<b>Tarih:</b> {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Summary
    story.append(Paragraph("Özet", styles["Heading2"]))
    story.append(Paragraph(f"Güncel Fiyat: <b>{plan.debug.get('close', float('nan')):.2f}</b>", styles["Normal"]))
    story.append(Paragraph(f"Toplam Skor: <b>{plan.total_score}/100</b> &nbsp;&nbsp; Etiket: <b>{plan.label}</b>", styles["Normal"]))
    story.append(Paragraph(f"Setup: <b>{plan.setup_score}/100</b> &nbsp;&nbsp; Timing: <b>{plan.timing_score}/100</b>", styles["Normal"]))
    story.append(Paragraph(f"Durum: {plan.status_tag}", styles["Normal"]))
    story.append(Paragraph(
        f"Minervini #5: {'OK' if plan.minervini5_ok else 'FAIL'} | 52W dip={plan.low_52w:.2f} | 52W tepe={plan.high_52w:.2f}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Giriş: <b>{plan.entry_low:.2f} – {plan.entry_high:.2f}</b>", styles["Normal"]))
    story.append(Paragraph(f"Stop: <b>{plan.stop:.2f}</b>", styles["Normal"]))
    story.append(Paragraph(f"TP1: <b>{plan.tp1:.2f}</b> (R/R≈1:{plan.rr_tp1:.2f})", styles["Normal"]))
    story.append(Paragraph(f"TP2: <b>{plan.tp2:.2f}</b> (R/R≈1:{plan.rr_tp2:.2f})", styles["Normal"]))
    story.append(Spacer(1, 10))

    if quote and isinstance(quote, dict):
        story.append(Paragraph("Quote (Anlık Özet)", styles["Heading2"]))
        keys = ["name", "exchange", "currency", "close", "price", "change", "percent_change", "previous_close"]
        for k in keys:
            if k in quote:
                story.append(Paragraph(f"{k}: {quote[k]}", styles["Normal"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Senaryo", styles["Heading2"]))
    story.append(Paragraph(plan.scenario, styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Otomatik Teknik Yorum", styles["Heading2"]))
    plain = plan.narrative.replace("**", "").replace("  \n", "<br/>")
    story.append(Paragraph(plain, styles["Normal"]))

    doc.build(story)
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
# PORTFOLIO ANALYSIS HELPERS (Blue Sky + trailing)
# =========================================================
def rolling_52w_levels(df: pd.DataFrame, bars_1day: int = 260) -> tuple[float, float]:
    """
    52W High/Low approximation:
    - For daily data: last 260 bars.
    - For intraday: we still approximate using last 260 bars of that interval (less meaningful),
      but we will ONLY use Blue Sky logic in portfolio when interval is daily OR when we have enough bars.
    """
    if df is None or df.empty:
        return (np.nan, np.nan)
    n = min(len(df), int(bars_1day))
    window = df.iloc[-n:]
    hi = float(window["high"].max()) if "high" in window.columns else np.nan
    lo = float(window["low"].min()) if "low" in window.columns else np.nan
    return hi, lo


def is_blue_sky(price: float, high_52w: float, threshold: float = 0.98) -> bool:
    """
    Blue Sky = price is within 2% of 52W High or above.
    threshold=0.98 means price >= 0.98 * high_52w
    """
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return False
    return price >= (threshold * high_52w)


def trailing_structure_status(price: float, ema20: float, ema50: float) -> tuple[str, str]:
    """
    Returns (headline, detail_text)
    - No commands, just state.
    """
    if not (np.isfinite(price) and np.isfinite(ema20) and np.isfinite(ema50)):
        return ("—", "İz süren yapı için veri eksik.")

    above20 = price >= ema20
    above50 = price >= ema50

    if above20 and above50:
        return ("İz süren yapı korunuyor.", f"EMA20: ÜZERİNDE | EMA50: ÜZERİNDE")
    if (not above20) and above50:
        return ("Kısa vadeli iz süren yapı zayıflıyor.", f"EMA20: ALTINDA | EMA50: ÜZERİNDE")
    return ("İz süren yapı bozulma sinyali veriyor.", f"EMA20: ALTINDA | EMA50: ALTINDA")


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
    if not np.isfinite(price):
        return "BİLİNEMİYOR", "Fiyat alınamadı."

    if np.isfinite(user_stop) and price <= user_stop:
        return "SAT/STOP", "Fiyat stop altı → pozisyonu disiplinle kapat."

    if not plan.minervini5_ok:
        return "TUT/İZLE", "Minervini #5 geçmiyor → ekleme yok; güç kanıtı bekle."

    if plan.status_tag.startswith("🔴"):
        return "RİSK AZALT", "Trend bozuk → ekleme yok; stopu sıkılaştır / pozisyon azaltmayı düşün."

    if plan.status_tag.startswith("⚫"):
        return "TUT", "Uzamış → ekleme yok; pullback ile yeniden değerlendir."

    if plan.status_tag.startswith("🔵"):
        return "TUT/İZLE", "Sıkışma → kırılım/bozulma gelene kadar sabır."

    if np.isfinite(user_tp2) and price >= user_tp2:
        return "KARAR NOKTASI", "TP2 bölgesi → momentum bozulursa kısmi/çıkış; korunuyorsa trailing stop."
    if np.isfinite(user_tp1) and price >= user_tp1:
        return "STOP YUKARI", "TP1 bölgesi → stopu yukarı çekerek trend takip et (satış değil, yönetim)."

    if np.isfinite(user_stop) and (price - user_stop) / price < 0.03:
        return "DİKKAT", "Stop çok yakın → oynaklık stoplatabilir. (Gevşetme yok; pozisyon boyunu düşün.)"

    return "TUT", "Koşullar fena değil → plana sadık kal; ekleme için giriş bandı ve timing bekle."


# =========================================================
# PORTFOLIO SUMMARY + EXPORTS (NEW)
# =========================================================
def compute_portfolio_summary(out: pd.DataFrame) -> Dict[str, float]:
    """
    out tablosu üstünden portföy metrikleri.
    Beklenen kolonlar: Adet, Fiyat, Alış Ort., Stop, TP1
    """
    if out is None or out.empty:
        return {
            "portfolio_value": np.nan,
            "cost_basis": np.nan,
            "pnl_now": np.nan,
            "pnl_now_pct": np.nan,
            "tp1_scn": np.nan,
            "stop_scn": np.nan,
            "rr_portfolio": np.nan,
        }

    def _to_float_series(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series([np.nan] * len(out))
        return pd.to_numeric(out[col], errors="coerce")

    qty = _to_float_series("Adet")
    price = _to_float_series("Fiyat")
    avg = _to_float_series("Alış Ort.")
    stop = _to_float_series("Stop")
    tp1 = _to_float_series("TP1")

    qty = qty.fillna(0.0)
    price = price.astype(float)
    avg = avg.astype(float)
    stop = stop.astype(float)
    tp1 = tp1.astype(float)

    portfolio_value = float(np.nansum(qty * price))
    cost_basis = float(np.nansum(qty * avg))

    pnl_now = float(np.nansum(qty * (price - avg)))
    pnl_now_pct = float((pnl_now / cost_basis) * 100.0) if np.isfinite(cost_basis) and cost_basis > 0 else np.nan

    tp1_scn = float(np.nansum(qty * (tp1 - avg)))  # TP1 senaryosu (maks kâr gibi düşünme: TP1 hepsi çalışırsa)
    stop_scn = float(np.nansum(qty * (stop - avg)))  # hepsi stop olursa (negatif çıkar)

    # Portfolio risk/reward (TP1 vs Stop) - sadece anlamlı satırlar
    mask = (
        np.isfinite(qty) & (qty > 0) &
        np.isfinite(avg) & (avg > 0) &
        np.isfinite(stop) & np.isfinite(tp1)
    )
    risk_total = float(np.nansum(qty[mask] * (avg[mask] - stop[mask])))  # pozitif risk = avg-stop
    reward_total = float(np.nansum(qty[mask] * (tp1[mask] - avg[mask])))
    rr_portfolio = float(reward_total / risk_total) if (np.isfinite(risk_total) and risk_total > 0) else np.nan

    return {
        "portfolio_value": portfolio_value,
        "cost_basis": cost_basis,
        "pnl_now": pnl_now,
        "pnl_now_pct": pnl_now_pct,
        "tp1_scn": tp1_scn,
        "stop_scn": stop_scn,
        "rr_portfolio": rr_portfolio,
    }


def build_portfolio_excel_bytes(out: pd.DataFrame, summary: Dict[str, float]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Portfolio"

    # Title row
    ws["A1"] = "MinerWin — Portföy Analiz Raporu (V5.2)"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, min(20, out.shape[1] if out is not None else 1)))
    ws["A2"] = f"Rapor Tarihi (UTC): {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')}"
    ws["A2"].font = Font(size=10, color="666666")

    start_row = 4

    if out is None or out.empty:
        ws["A4"] = "Portföy çıktısı boş."
    else:
        # Write dataframe
        for r_idx, row in enumerate(dataframe_to_rows(out, index=False, header=True), start=start_row):
            ws.append(row)

        header_row = start_row
        thin = Side(style="thin", color="2B2F36")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        # Header style
        header_fill = PatternFill("solid", fgColor="1F2937")
        for c in range(1, out.shape[1] + 1):
            cell = ws.cell(row=header_row, column=c)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

        # Body style + borders
        for r in range(header_row + 1, header_row + 1 + out.shape[0]):
            for c in range(1, out.shape[1] + 1):
                cell = ws.cell(row=r, column=c)
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                cell.border = border

        # Auto column widths (safe)
        for c in range(1, out.shape[1] + 1):
            max_len = 10
            col_letter = ws.cell(row=header_row, column=c).column_letter
            for r in range(header_row, header_row + 1 + out.shape[0]):
                val = ws.cell(row=r, column=c).value
                if val is None:
                    continue
                val_str = str(val)[:40]
                max_len = max(max_len, len(val_str))
    ws.column_dimensions[col_letter].width = min(28, max(10, max_len + 2))

        # Conditional formatting for P&L %
        try:
            if "P&L %" in out.columns:
                c_idx = out.columns.get_loc("P&L %") + 1
                col_letter = ws.cell(row=header_row, column=c_idx).column_letter
                data_range = f"{col_letter}{header_row+1}:{col_letter}{header_row+out.shape[0]}"
                ws.conditional_formatting.add(data_range, CellIsRule(operator="greaterThan", formula=["0"], font=Font(color="0A8A0A")))
                ws.conditional_formatting.add(data_range, CellIsRule(operator="lessThan", formula=["0"], font=Font(color="B00020")))
        except Exception:
            pass

        # Summary sheet
    ws2 = wb.create_sheet("Summary")
    ws2["A1"] = "Portföy Özeti"
    ws2["A1"].font = Font(bold=True, size=14)

    items = [
        ("Portföy Değeri", summary.get("portfolio_value", np.nan)),
        ("Toplam Maliyet (Cost Basis)", summary.get("cost_basis", np.nan)),
        ("Anlık P&L", summary.get("pnl_now", np.nan)),
        ("Anlık P&L %", summary.get("pnl_now_pct", np.nan)),
        ("TP1 Senaryosu (Hepsi TP1)", summary.get("tp1_scn", np.nan)),
        ("Stop Senaryosu (Hepsi Stop)", summary.get("stop_scn", np.nan)),
        ("Portföy R/R (TP1 vs Stop)", summary.get("rr_portfolio", np.nan)),
    ]

    r = 3
    for k, v in items:
        ws2.cell(row=r, column=1).value = k
        ws2.cell(row=r, column=1).font = Font(bold=True)
        if "P&L %" in k:
            ws2.cell(row=r, column=2).value = float(v) if np.isfinite(v) else None
            ws2.cell(row=r, column=2).number_format = '0.00"%"'
        elif "R/R" in k:
            ws2.cell(row=r, column=2).value = float(v) if np.isfinite(v) else None
            ws2.cell(row=r, column=2).number_format = "0.00"
        else:
            ws2.cell(row=r, column=2).value = float(v) if np.isfinite(v) else None
            ws2.cell(row=r, column=2).number_format = '"$"#,##0.00;[Red]-"$"#,##0.00'
        r += 1

    ws2.column_dimensions["A"].width = 34
    ws2.column_dimensions["B"].width = 20

    # Bytes
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def build_portfolio_pdf_bytes(out: pd.DataFrame, summary: Dict[str, float]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # Header row with logo if exists
    logo_path = _find_logo_path()
    header_tbl = None
    try:
        if logo_path:
            img = RLImage(logo_path, width=2.0 * cm, height=2.0 * cm)
            header_tbl = Table(
                [[img, Paragraph("<b>MinerWin — Portföy Analiz Raporu (V5.2)</b>", styles["Title"])]],
                colWidths=[2.3 * cm, 24.0 * cm],
            )
        else:
            header_tbl = Table([[Paragraph("<b>MinerWin — Portföy Analiz Raporu (V5.2)</b>", styles["Title"])]])
        header_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(header_tbl)
    except Exception:
        story.append(Paragraph("MinerWin — Portföy Analiz Raporu (V5.2)", styles["Title"]))

    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<span color='#666666'>Rapor Tarihi (UTC): {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')}</span>", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Summary box
    sum_rows = [
        ["Portföy Değeri", fmt_money(summary.get("portfolio_value", np.nan)),
         "Anlık P&L", fmt_money(summary.get("pnl_now", np.nan)) + f" ({fmt_pct(summary.get('pnl_now_pct', np.nan))})"],
        ["TP1 Senaryosu (Hepsi TP1)", fmt_money(summary.get("tp1_scn", np.nan)),
         "Stop Senaryosu (Hepsi Stop)", fmt_money(summary.get("stop_scn", np.nan))],
        ["Toplam Maliyet (Cost Basis)", fmt_money(summary.get("cost_basis", np.nan)),
         "Portföy R/R (TP1 vs Stop)", "—" if not np.isfinite(summary.get("rr_portfolio", np.nan)) else f"1 : {summary.get('rr_portfolio', np.nan):.2f}"],
    ]
    sum_tbl = Table(sum_rows, colWidths=[6.3 * cm, 6.2 * cm, 6.3 * cm, 6.2 * cm])
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#2B2F36")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F4F6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Portföy Tablosu", styles["Heading2"]))
    story.append(Spacer(1, 6))

    if out is None or out.empty:
        story.append(Paragraph("Portföy çıktısı boş.", styles["Normal"]))
    else:
        # Select a presentable column set (keep core + new columns)
        preferred = [
            "Ticker", "Adet", "Fiyat", "Alış Ort.", "P&L %", "Poz. Değeri",
            "Stop", "Risk $", "TP1", "TP2",
            "Setup", "Timing", "Durum", "Minervini #5",
            "52W High", "Blue Sky", "İz Süren Yapı",
            "Aksiyon", "Not"
        ]
        cols = [c for c in preferred if c in out.columns]
        small = out[cols].copy()

        # Convert to strings for safe PDF table rendering
        def _safe_str(x):
            if x is None:
                return ""
            if isinstance(x, float) and (not np.isfinite(x)):
                return ""
            return str(x)

        data = [cols] + [[_safe_str(v) for v in row] for row in small.values.tolist()]

        # Column widths tuned for landscape A4
        # (Some columns will still wrap; that's OK)
        base_widths = []
        for c in cols:
            if c in ("Ticker", "Blue Sky"):
                base_widths.append(1.4 * cm)
            elif c in ("Adet", "Setup", "Timing", "Minervini #5"):
                base_widths.append(1.7 * cm)
            elif c in ("Fiyat", "Alış Ort.", "Stop", "TP1", "TP2", "Poz. Değeri", "Risk $", "52W High", "P&L %"):
                base_widths.append(2.2 * cm)
            elif c in ("Durum", "İz Süren Yapı", "Aksiyon"):
                base_widths.append(3.2 * cm)
            else:  # Not
                base_widths.append(6.0 * cm)

        tbl = Table(data, colWidths=base_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#2B2F36")),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F3F4F6")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(tbl)

    story.append(Spacer(1, 10))
    story.append(Paragraph("<span color='#666666'>Not: Bu PDF emir dili içermez; portföy fotoğrafı + senaryo metrikleri verir.</span>", styles["Normal"]))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Genel Ayarlar")
    default_interval_label = st.selectbox(
        "Varsayılan zaman çözünürlüğü",
        list(INTERVAL_MAP.keys()),
        index=list(INTERVAL_MAP.keys()).index(DEFAULT_SINGLE_INTERVAL_LABEL),
    )
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
            "minervini5_ok", "stop", "tp1", "tp2"
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
# TAB 1: SINGLE STOCK (STRATEJİ DEĞİŞMEDİ)
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

                df["ema20"] = ema(df["close"], 20)
                df["ema50"] = ema(df["close"], 50)
                df["ema150"] = ema(df["close"], 150)
                df["ema200"] = ema(df["close"], 200)
                df["rsi14"] = rsi(df["close"], 14)
                df["atr14"] = atr(df, 14)

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

                st.session_state["__df"] = df
                st.session_state["__ticker"] = ticker
                st.session_state["__plan"] = plan
                st.session_state["__quote"] = q
                st.session_state["__last_price_line"] = float(last_price_line)
                st.session_state["__interval_label"] = interval_label
                st.session_state["__bars"] = bars

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

                st.divider()

                st.subheader("📊 Strateji Özeti (Görünür)")
                colm1, colm2, colm3 = st.columns(3)
                with colm1:
                    st.metric("Güncel Fiyat", f"{float(last_price_line):.2f}")
                    st.metric("Durum", plan.status_tag)
                with colm2:
                    st.metric("Toplam Skor", f"{plan.total_score} / 100")
                    st.metric("Setup / Timing", f"{plan.setup_score} / {plan.timing_score}")
                with colm3:
                    st.metric("Stop (Aktif)", f"{plan.stop:.2f}")
                    st.metric("TP1 / TP2", f"{plan.tp1:.2f} / {plan.tp2:.2f}")
                    st.caption(f"Yapısal: {plan.debug.get('stop_structural', float('nan')):.2f} | Noise: {plan.debug.get('stop_noise', float('nan')):.2f}")

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
                    if quote_price is not None and np.isfinite(quote_price) and abs(quote_price - candle_close) < 1e-9:
                        st.info("Piyasa kapalıysa quote son kapanışı gösterebilir (pre/post-market yok).")

                st.subheader("🧩 İşlem Yönetimi (Eldeki Hisse)")
                st.caption("Stop asla gevşetilmez. Buradaki değerler senin pozisyon yönetimindir; otomatik planı ezmez.")

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

                mg = st.session_state.trade_mgmt[ticker]
                cur_price = float(last_price_line)
                entry0 = float(mg["entry"])
                stop0 = float(mg["stop"])
                tp1_0 = float(mg["tp1"])
                tp2_0 = float(mg["tp2"])

                suggestions = []
                if np.isfinite(cur_price) and np.isfinite(entry0) and entry0 > 0:
                    move_pct = (cur_price - entry0) / entry0 * 100.0
                    if move_pct >= 5.0:
                        sug_stop = max(stop0, entry0)
                        suggestions.append(f"Entry’ye göre %+{move_pct:.1f}. Stop’u en az **break-even** seviyesine çekmeyi düşünebilirsin: {sug_stop:.2f}")

                if np.isfinite(tp1_0) and cur_price >= tp1_0:
                    ema20_now = float(df.iloc[-1]["ema20"])
                    ema50_now = float(df.iloc[-1]["ema50"])
                    trail = max(stop0, min(cur_price * 0.995, max(ema20_now, ema50_now) * 0.995))
                    suggestions.append(f"TP1 bölgesi: stop’u **EMA bazlı** yukarı taşı (gevşetme yok): {trail:.2f}")

                if np.isfinite(tp2_0) and cur_price >= tp2_0:
                    suggestions.append("TP2 bölgesi: Momentum bozulursa kısmi/çıkış; korunuyorsa trailing stop.")

                if suggestions:
                    st.info("**Yönetim Önerisi:**\n\n- " + "\n- ".join(suggestions))
                else:
                    st.caption("Yönetim önerileri için: fiyatın entry/TP seviyelerine yaklaşmasını bekle.")

                st.subheader("📄 Rapor")
                pdf_bytes = build_pdf_bytes_single(
                    ticker=ticker,
                    interval_label=interval_label,
                    bars=bars,
                    plan=plan,
                    quote=(q if show_quote else None),
                )
                st.download_button(
                    label="Tek Hisse PDF İndir",
                    data=pdf_bytes,
                    file_name=f"{ticker}_{INTERVAL_MAP[interval_label]}_rapor.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                with st.expander("Detay (debug)"):
                    st.json(plan.debug)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 2: PORTFOLIO (NEW EXPORTS + SUMMARY)
# =========================================================
with tab_portfolio:
    st.subheader("🧳 Portföy Analiz")
    st.caption("Portföy satırlarını gir: ticker, adet, alış ort., stop, TP1, TP2. Analiz → eldeki pozisyon dili + risk + auto kıyas.")

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
            "Portföy analiz zaman dilimi (ana karar için günlük önerilir)",
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

                            low_52w, high_52w = (float("nan"), float("nan"))
                            try:
                                low_52w, high_52w = fetch_daily_52w(tkr)
                            except Exception:
                                pass

                            plan = build_trade_plan(dfi, low_52w=low_52w, high_52w=high_52w)

                            candle_close = float(dfi.iloc[-1]["close"])
                            price = candle_close

                            if show_quote:
                                try:
                                    q = td_quote(tkr)
                                    if "price" in q:
                                        price = float(q["price"])
                                except Exception:
                                    pass

                            high_52w_roll, low_52w_roll = rolling_52w_levels(dfi, bars_1day=260)
                            blue = is_blue_sky(price, high_52w_roll, threshold=0.98)

                            ema20_now = float(dfi.iloc[-1]["ema20"])
                            ema50_now = float(dfi.iloc[-1]["ema50"])
                            trail_head, trail_detail = trailing_structure_status(price, ema20_now, ema50_now)

                            in_profit = np.isfinite(avg_cost) and avg_cost > 0 and np.isfinite(price) and (price > avg_cost)
                            show_blue_box = bool(in_profit and blue)

                            pnl_pct = pct(price, avg_cost) if np.isfinite(avg_cost) and avg_cost > 0 else np.nan
                            dist_stop_pct = pct(price, user_stop) if np.isfinite(user_stop) and user_stop > 0 else np.nan
                            dist_tp1_pct = pct(user_tp1, price) if np.isfinite(user_tp1) and user_tp1 > 0 else np.nan
                            dist_tp2_pct = pct(user_tp2, price) if np.isfinite(user_tp2) and user_tp2 > 0 else np.nan
                            rr_user_tp1 = compute_rr(price, user_stop, user_tp1)
                            rr_user_tp2 = compute_rr(price, user_stop, user_tp2)

                            action, comment = held_action_comment(plan, price, avg_cost, user_stop, user_tp1, user_tp2)

                            pos_value = (qty * price) if np.isfinite(qty) and np.isfinite(price) else np.nan
                            risk_per_share = (avg_cost - user_stop) if (np.isfinite(user_stop) and np.isfinite(avg_cost)) else np.nan
                            risk_value = (risk_per_share * qty) if (np.isfinite(risk_per_share) and np.isfinite(qty)) else np.nan

                            rows.append({
                                "Ticker": tkr,
                                "Adet": round(qty, 2) if np.isfinite(qty) else "",
                                "Fiyat": round(price, 2),
                                "Alış Ort.": round(avg_cost, 2) if np.isfinite(avg_cost) else "",
                                "P&L %": round(pnl_pct, 2) if np.isfinite(pnl_pct) else "",
                                "Poz. Değeri": round(pos_value, 2) if np.isfinite(pos_value) else "",
                                "Stop": round(user_stop, 2) if np.isfinite(user_stop) else "",
                                "Stop Mesafe %": round(dist_stop_pct, 2) if np.isfinite(dist_stop_pct) else "",
                                "Risk $": round(risk_value, 2) if np.isfinite(risk_value) else "",
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
                                "52W High": round(high_52w_roll, 2) if np.isfinite(high_52w_roll) else "",
                                "Blue Sky": "🔵" if show_blue_box else "",
                                "İz Süren Yapı": trail_head if show_blue_box else "",
                                "Auto Yapısal Stop": round(plan.debug.get("stop_structural", np.nan), 2) if np.isfinite(plan.debug.get("stop_structural", np.nan)) else "",
                                "Auto Noise Stop": round(plan.debug.get("stop_noise", np.nan), 2) if np.isfinite(plan.debug.get("stop_noise", np.nan)) else "",
                                "Aksiyon": action,
                                "Not": comment,
                            })

                        except Exception as e:
                            rows.append({
                                "Ticker": tkr,
                                "Adet": round(qty, 2) if np.isfinite(qty) else "",
                                "Fiyat": "",
                                "Alış Ort.": round(avg_cost, 2) if np.isfinite(avg_cost) else "",
                                "P&L %": "",
                                "Poz. Değeri": "",
                                "Stop": round(user_stop, 2) if np.isfinite(user_stop) else "",
                                "Stop Mesafe %": "",
                                "Risk $": "",
                                "TP1": round(user_tp1, 2) if np.isfinite(user_tp1) else "",
                                "TP1 Mesafe %": "",
                                "TP2": round(user_tp2, 2) if np.isfinite(user_tp2) else "",
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
                                "52W High": "",
                                "Blue Sky": "",
                                "İz Süren Yapı": "",
                                "Auto Yapısal Stop": "",
                                "Auto Noise Stop": "",
                                "Aksiyon": "HATA",
                                "Not": f"Veri/analiz hatası: {e}",
                            })

                out = pd.DataFrame(rows)

                # ---- SUMMARY METRICS (NEW) ----
                summary = compute_portfolio_summary(out)

                st.markdown("### 📦 Portföy Özeti")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Portföy Değeri", fmt_money(summary.get("portfolio_value", np.nan)))
                c2.metric("Anlık P&L", fmt_money(summary.get("pnl_now", np.nan)), fmt_pct(summary.get("pnl_now_pct", np.nan)))
                c3.metric("TP1 Senaryosu (Hepsi TP1)", fmt_money(summary.get("tp1_scn", np.nan)))
                c4.metric("Stop Senaryosu (Hepsi Stop)", fmt_money(summary.get("stop_scn", np.nan)))

                rr_val = summary.get("rr_portfolio", np.nan)
                st.caption(f"Portföy R/R (TP1 vs Stop): {'—' if not np.isfinite(rr_val) else f'1 : {rr_val:.2f}'}")

                st.markdown("### Sonuç Tablosu")
                st.dataframe(out, use_container_width=True, hide_index=True)

                # ---- EXPORTS (NEW) ----
                st.markdown("### 📤 İndirilebilir Raporlar")
                pdf_bytes = build_portfolio_pdf_bytes(out, summary)
                xlsx_bytes = build_portfolio_excel_bytes(out, summary)

                d1, d2, d3 = st.columns([0.34, 0.33, 0.33])
                with d1:
                    st.download_button(
                        "Portföy PDF İndir",
                        data=pdf_bytes,
                        file_name=f"minerwin_portfolio_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "Portföy Excel (XLSX) İndir",
                        data=xlsx_bytes,
                        file_name=f"minerwin_portfolio_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                with d3:
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Opsiyonel: CSV İndir",
                        data=csv_bytes,
                        file_name="portfolio_analysis.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                # --- Blue Sky informational box ---
                st.markdown("### 🔵 Blue Sky Evresi (Bilgilendirme)")
                st.caption("Sadece kârda olan ve 52W zirve bölgesindeki pozisyonlar için görünür.")

                if not out.empty:
                    blue_rows = out[out["Blue Sky"].astype(str).str.contains("🔵", na=False)].copy()
                    if blue_rows.empty:
                        st.info("Şu an Blue Sky koşulunda pozisyon yok.")
                    else:
                        for _, rr in blue_rows.iterrows():
                            st.markdown(f"**{rr['Ticker']}**")
                            st.write("• Fiyat, 52 haftalık zirve bölgesinde işlem görüyor.")
                            st.write("• Geçmiş direnç seviyeleri bulunmuyor.")
                            st.write("• Bu evrede hedef seviyeler yerine trend yapısı izlenir.")
                            if str(rr.get("İz Süren Yapı", "")).strip():
                                st.write(f"📐 İz Süren Yapı: {rr['İz Süren Yapı']}")
                            st.divider()

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
