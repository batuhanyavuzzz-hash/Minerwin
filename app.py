# app.py
# MinerWin — Tek Hisse + Portföy Analiz (V6.1) — Twelve Data
#
# V6.1 Düzeltmeleri:
# 1. 52W high/low her durumda daily veriden hesaplanır
# 2. Portföyde 52W tuple sırası düzeltildi
# 3. Blue Sky mantığı doğru high_52w ile çalışır
# 4. RSI yönü artık skora dahil edilir (+5 / 0 / -5)
# 5. Skor normalize tavanı 130'a göre güncellendi
# 6. TP2 52W cap breakout/blue-sky durumunda esnetildi
# 7. Risk $ hesabı avg_cost - stop bazına alındı
# 8. Max zarar KPI pozitif büyüklük olarak gösterilir
# 9. Günlük dışı timeframe’lerde de Minervini #5 / 52W uzaklık doğru çalışır

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
from typing import Any, Dict, Tuple

# PDF (WeasyPrint)
try:
    from weasyprint import HTML as _WeasyprintHTML
    _WEASYPRINT_OK = True
except ImportError:
    _WEASYPRINT_OK = False

# Excel (openpyxl)
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.table import Table as XLTable, TableStyleInfo


# =========================================================
# SABİTLER
# =========================================================
MINERVINI5_THRESHOLD = 1.25
MAX_RISK_PCT_DEFAULT = 7.0
DRYUP_RATIO_THRESHOLD = 0.60
BREAKOUT_VOL_MULTIPLIER = 1.50
NEAR_HIGH_THRESHOLD = 0.25
BLUE_SKY_THRESHOLD = 0.98
EXTENDED_EMA50_PCT = 8.0
PIVOT_LOOKBACK = 20
RSI_MOMENTUM_LOOKBACK = 5

TP_CAP_MOMENTUM = {
    "HIGH": (0.50, 0.85),
    "MID":  (0.30, 0.50),
    "LOW":  (0.18, 0.28),
}


def dynamic_stop_cap(atr_pct: float) -> float:
    if not np.isfinite(atr_pct):
        return MAX_RISK_PCT_DEFAULT
    if atr_pct < 2.0:
        return 5.0
    if atr_pct < 4.0:
        return 7.0
    if atr_pct < 6.0:
        return 9.0
    return 11.0


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="MinerWin – Portföy Analizi",
    page_icon="minerwin_favicon.png",
    layout="wide",
)


def _load_logo_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


logo_b64 = _load_logo_b64("minerwin_logo.png")

st.markdown(
    """
<style>
.block-container { padding-top: 3.2rem; }
.header { display:flex; align-items:center; gap:14px; margin-bottom:6px; }
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
<div class="sub-title">Minervini-Based Technical Trading Engine — V6.1</div>
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
    st.session_state.portfolio = pd.DataFrame(
        columns=["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"]
    )

if "trade_mgmt" not in st.session_state:
    st.session_state.trade_mgmt = {}


# =========================================================
# TEMEL YARDIMCILAR
# =========================================================
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


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}"


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:+.2f}%"


# =========================================================
# İNDİKATÖRLER
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


def rsi_slope(rsi_series: pd.Series, lookback: int = RSI_MOMENTUM_LOOKBACK) -> float:
    s = rsi_series.dropna()
    if len(s) < lookback + 1:
        return float("nan")
    y = s.iloc[-lookback:].values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


# =========================================================
# VERİ (Twelve Data)
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
        raise RuntimeError(
            f"TwelveData: {payload.get('message')} (code={payload.get('code')})"
        )

    values = payload.get("values")
    if not values:
        raise RuntimeError(
            "TwelveData: 'values' boş döndü (ticker/interval desteklenmiyor olabilir)."
        )

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

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df


# =========================================================
# GÜNLÜK VERİ / 52W
# =========================================================
@st.cache_data(ttl=300)
def _fetch_daily_df(symbol: str, outputsize: int = 320) -> pd.DataFrame:
    payload = td_time_series(symbol, "1day", int(outputsize))
    return parse_ohlcv(payload)


def compute_52w_levels(df: pd.DataFrame, bars_1day: int = 260) -> Tuple[float, float]:
    if df is None or df.empty:
        return float("nan"), float("nan")
    n = min(len(df), int(bars_1day))
    window = df.iloc[-n:]
    low_52w = float(window["low"].min()) if "low" in window.columns else float("nan")
    high_52w = float(window["high"].max()) if "high" in window.columns else float("nan")
    return low_52w, high_52w


def get_daily_52w_levels(symbol: str, interval: str, current_df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """
    52W seviyeleri her zaman daily veriden hesaplanır.
    Eğer zaten interval=1day ise mevcut df kullanılır.
    """
    if interval == "1day" and current_df is not None and not current_df.empty:
        daily_df = current_df.copy()
    else:
        daily_df = _fetch_daily_df(symbol, 320)
    low_52w, high_52w = compute_52w_levels(daily_df, bars_1day=260)
    return low_52w, high_52w, daily_df


# =========================================================
# GEÇMİŞ (CSV) + OTURUM HAFIZASI
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
# MİNERVİNİ KURAL 5
# =========================================================
def minervini_rule5_ok(price: float, low_52w: float) -> bool:
    if not (np.isfinite(price) and np.isfinite(low_52w) and low_52w > 0):
        return False
    return price >= MINERVINI5_THRESHOLD * low_52w


# =========================================================
# STOP MOTORU
# =========================================================
def _recent_pivot_low(df: pd.DataFrame, lookback: int = PIVOT_LOOKBACK) -> float:
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


BAZ_LOOKBACK = 20
PIVOT_BREAK_LOOKBACK = 20
ATR_CONTRACT_RATIO = 0.80
VOL_DRY_RATIO = 0.75

BASE_BONUS_PTS = 7
BREAKOUT_BONUS_PTS = 8


def detect_base_and_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "base_detected": False,
        "breakout_detected": False,
        "base_pts": 0,
        "breakout_pts": 0,
        "total_bonus_pts": 0,
        "details": {},
    }

    if df is None or len(df) < BAZ_LOOKBACK + 10:
        return result

    atr_full = float(df["atr14"].dropna().mean())
    atr_base = float(df["atr14"].iloc[-BAZ_LOOKBACK:].mean())
    atr_contracted = (
        np.isfinite(atr_full) and np.isfinite(atr_base)
        and atr_full > 0
        and atr_base <= atr_full * ATR_CONTRACT_RATIO
    )

    vol = df["volume"].astype(float).fillna(0.0)
    vol_full_mean = float(vol.mean())
    vol_base_mean = float(vol.iloc[-BAZ_LOOKBACK:].mean())
    vol_dried = (
        np.isfinite(vol_full_mean) and np.isfinite(vol_base_mean)
        and vol_full_mean > 0
        and vol_base_mean <= vol_full_mean * VOL_DRY_RATIO
    )

    base_detected = atr_contracted and vol_dried

    if len(df) >= PIVOT_BREAK_LOOKBACK + 2:
        pivot_high = float(df["high"].iloc[-(PIVOT_BREAK_LOOKBACK + 1):-1].max())
        last_close = float(df["close"].iloc[-1])
        last_vol = float(df["volume"].iloc[-1])
        vol_50mean = float(vol.rolling(50).mean().iloc[-1])

        price_broke = np.isfinite(pivot_high) and np.isfinite(last_close) and last_close > pivot_high
        vol_confirmed = (
            np.isfinite(last_vol) and np.isfinite(vol_50mean)
            and vol_50mean > 0
            and last_vol >= 1.4 * vol_50mean
        )
        breakout_detected = price_broke and vol_confirmed
    else:
        pivot_high = float("nan")
        last_close = float(df["close"].iloc[-1])
        price_broke = False
        vol_confirmed = False
        breakout_detected = False

    base_pts = BASE_BONUS_PTS if base_detected else 0
    breakout_pts = BREAKOUT_BONUS_PTS if breakout_detected else 0
    total_bonus = base_pts + breakout_pts

    result.update({
        "base_detected": base_detected,
        "breakout_detected": breakout_detected,
        "base_pts": base_pts,
        "breakout_pts": breakout_pts,
        "total_bonus_pts": total_bonus,
        "details": {
            "atr_full_mean": atr_full,
            "atr_base_mean": atr_base,
            "atr_contracted": atr_contracted,
            "vol_full_mean": vol_full_mean,
            "vol_base_mean": vol_base_mean,
            "vol_dried": vol_dried,
            "pivot_high": pivot_high,
            "last_close": last_close,
            "price_broke_pivot": price_broke,
            "vol_confirmed": vol_confirmed,
        },
    })
    return result


def _noise_factor_from_atr_pct(atr_pct: float) -> float:
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
) -> Tuple[float, float, float, Dict[str, Any]]:
    max_risk_pct = dynamic_stop_cap(atr_pct)

    if not (np.isfinite(entry) and np.isfinite(ema50) and np.isfinite(atr14)) or entry <= 0:
        stop_fallback = float(entry * 0.93)
        return stop_fallback, float("nan"), float("nan"), {"reason": "NaN entry/ema50/atr14"}

    inv_from_ema = float(ema50 * 0.995)
    inv_from_pivot = (
        float(pivot_low * 0.995)
        if (np.isfinite(pivot_low) and pivot_low > 0)
        else float("nan")
    )

    if np.isfinite(inv_from_pivot):
        stop_structural = float(min(inv_from_ema, inv_from_pivot))
        inv_src = "pivot_or_ema"
    else:
        stop_structural = float(inv_from_ema)
        inv_src = "ema50"

    nf = _noise_factor_from_atr_pct(atr_pct)
    stop_noise = float(entry - nf * atr14)

    stop_candidate = float(min(stop_structural, stop_noise))

    cap_stop = float(entry * (1.0 - max_risk_pct / 100.0))
    if stop_candidate < cap_stop:
        stop_active = cap_stop
        capped = True
    else:
        stop_active = stop_candidate
        capped = False

    if stop_active >= entry:
        stop_active = float(entry * 0.99)

    high_vol_warning = capped and (atr_pct > 5.0)

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
        "high_vol_warning": high_vol_warning,
    }
    return float(stop_active), float(stop_structural), float(stop_noise), dbg


# =========================================================
# TP MOTORU
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
    capacity: str,
    dist_to_52w_high_pct: float,
    breakout_detected: bool,
) -> Tuple[float, float, float, str, Dict[str, Any]]:
    entry = float(entry)
    stop = float(stop)
    close = float(close)
    atr14 = float(atr14)

    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(close) and np.isfinite(atr14)):
        return entry * 1.06, entry * 1.12, float("nan"), capacity, {"reason": "NaN input"}

    risk = entry - stop
    if risk <= 0:
        return entry * 1.06, entry * 1.12, float("nan"), capacity, {"reason": "risk<=0"}

    atr_pct_ratio = atr14 / close if close > 0 else float("nan")
    atr_pct_ratio = clamp(atr_pct_ratio, 0.012, 0.085)

    if capacity == "HIGH":
        N, mult = 5.5, 1.30
    elif capacity == "MID":
        N, mult = 4.5, 1.10
    else:
        N, mult = 3.5, 0.95

    expected_move_pct = (atr_pct_ratio * N * mult) * 100.0

    impulse_cap_pct = _impulse_cap_pct_from_history(df_for_impulse, lookback=90)
    if np.isfinite(impulse_cap_pct) and impulse_cap_pct > 0:
        expected_move_pct = min(expected_move_pct, impulse_cap_pct * 0.90)

    tp1_floor = entry + 2.2 * risk
    tp2_floor = entry + 3.5 * risk

    tp1 = entry * (1.0 + (expected_move_pct / 100.0) * 0.55)
    tp2 = entry * (1.0 + (expected_move_pct / 100.0) * 0.90)

    tp1 = max(tp1, tp1_floor)
    tp2 = max(tp2, tp2_floor)

    tp1_cap_pct, tp2_cap_pct = TP_CAP_MOMENTUM.get(capacity, (0.18, 0.28))
    tp1 = min(tp1, entry * (1.0 + tp1_cap_pct))
    tp2 = min(tp2, entry * (1.0 + tp2_cap_pct))

    # 52W tavanı: sadece hâlâ belirgin biçimde 52W high altında ise baskılansın
    # breakout/blue-sky yakınında fazla erkenci kesmesin
    if np.isfinite(high_52w) and high_52w > 0:
        allow_looser_cap = breakout_detected or (
            np.isfinite(dist_to_52w_high_pct) and dist_to_52w_high_pct <= 1.0
        )
        if not allow_looser_cap and close < high_52w * 0.995:
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
        "tp1_cap_pct": tp1_cap_pct * 100,
        "tp2_cap_pct": tp2_cap_pct * 100,
        "high_52w": high_52w,
        "dist_to_52w_high_pct": dist_to_52w_high_pct,
        "breakout_detected": breakout_detected,
    }
    return float(tp1), float(tp2), float(expected_move_pct), capacity, dbg


# =========================================================
# SKOR / PLAN
# =========================================================
@dataclass
class ScoreBreakdown:
    trend_stack: int
    price_vs_ema150: int
    momentum_rsi: int
    volatility_atr: int
    extension_vs_ema50: int
    near_52w_high: int
    rsi_direction: int
    base_bonus: int
    breakout_bonus: int


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

    base_detected: bool
    breakout_detected: bool
    base_bonus_pts: int
    breakout_bonus_pts: int
    rsi_slope_val: float
    rsi_direction_label: str
    high_vol_warning: bool
    dist_to_52w_high_pct: float

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


def _rsi_direction_label(slope_val: float) -> str:
    if not np.isfinite(slope_val):
        return "Bilinmiyor"
    if slope_val > 0.5:
        return "Yükseliyor ↑"
    if slope_val < -0.5:
        return "Düşüyor ↓"
    return "Yatay →"


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

    base_result = detect_base_and_breakout(df)
    base_detected = base_result["base_detected"]
    breakout_detected = base_result["breakout_detected"]
    base_bonus_pts = base_result["base_pts"]
    breakout_bonus_pts = base_result["breakout_pts"]
    total_bonus_pts = base_result["total_bonus_pts"]

    rsi_slope_val = rsi_slope(df["rsi14"], lookback=RSI_MOMENTUM_LOOKBACK)
    rsi_dir_label = _rsi_direction_label(rsi_slope_val)

    if np.isfinite(rsi_slope_val):
        if rsi_slope_val > 0.5:
            rsi_dir_pts = 5
        elif rsi_slope_val < -0.5:
            rsi_dir_pts = -5
        else:
            rsi_dir_pts = 0
    else:
        rsi_dir_pts = 0

    if np.isfinite(high_52w) and high_52w > 0 and np.isfinite(close):
        dist_to_52w_high_pct = ((high_52w - close) / high_52w) * 100.0
    else:
        dist_to_52w_high_pct = float("nan")

    if np.isfinite(dist_to_52w_high_pct):
        if dist_to_52w_high_pct <= 10.0:
            near_52w_pts = 10
        elif dist_to_52w_high_pct <= 25.0:
            near_52w_pts = 5
        else:
            near_52w_pts = 0
    else:
        near_52w_pts = 0

    trend_stack_ok = ema50 > ema150 > ema200
    ema200_slope = slope(df["ema200"], lookback=20)
    long_trend_ok = ema200_slope > 0

    momentum_ok = rsi14 >= 55
    momentum_border = 50 <= rsi14 < 55

    vol_ok = 2.0 <= atr_pct <= 6.0
    vol_border = (1.5 <= atr_pct < 2.0) or (6.0 < atr_pct <= 8.0)

    price_above_ema150 = close >= ema150
    price_near_ema150 = close >= ema150 * 0.98

    extended = dist_ema50_pct > EXTENDED_EMA50_PCT
    trend_broken = (close < ema200) or (not long_trend_ok and not trend_stack_ok)

    m5_ok = minervini_rule5_ok(close, low_52w)

    trend_pts = (
        30 if (trend_stack_ok and long_trend_ok)
        else (20 if trend_stack_ok else (10 if long_trend_ok else 0))
    )
    p_pts = 20 if price_above_ema150 else (10 if price_near_ema150 else 0)
    m_pts = 20 if momentum_ok else (10 if momentum_border else 0)
    v_pts = 15 if vol_ok else (7 if vol_border else 0)
    e_pts = 15 if not extended else 0

    # Maks teorik toplam:
    # 30 + 20 + 20 + 15 + 15 + 10 + 5 + 7 + 8 = 130
    raw_total = (
        trend_pts + p_pts + m_pts + v_pts + e_pts +
        near_52w_pts + rsi_dir_pts + total_bonus_pts
    )

    total = int(round(clamp(raw_total / 130.0 * 100.0, 0, 100)))

    if not m5_ok:
        total = min(total, 55)

    label = label_from_total(total)
    breakdown = ScoreBreakdown(
        trend_stack=trend_pts,
        price_vs_ema150=p_pts,
        momentum_rsi=m_pts,
        volatility_atr=v_pts,
        extension_vs_ema50=e_pts,
        near_52w_high=near_52w_pts,
        rsi_direction=rsi_dir_pts,
        base_bonus=base_bonus_pts,
        breakout_bonus=breakout_bonus_pts,
    )

    entry_low = float(min(ema20, ema50))
    entry_high = float(max(ema20, ema50))
    entry_mid = float((entry_low + entry_high) / 2.0)

    setup_raw = trend_pts + p_pts + m_pts + v_pts
    setup_score = int(round(100 * setup_raw / 85)) if setup_raw > 0 else 0

    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)
    ext_pts = _extension_points(extended)
    timing_score = int(ext_pts + prox_pts)

    in_entry = entry_low <= close <= entry_high
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

    pivot_low = _recent_pivot_low(df, lookback=PIVOT_LOOKBACK)
    stop, stop_structural, stop_noise, stop_dbg = compute_stop_invalidation_plus_noise(
        entry=entry_mid,
        ema50=ema50,
        atr14=atr14,
        atr_pct=atr_pct,
        pivot_low=pivot_low,
    )
    high_vol_warning = bool(stop_dbg.get("high_vol_warning", False))

    capacity = _trend_capacity_level(
        setup_score, ema50, ema150, ema200, ema200_slope, rsi14, close
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
        capacity=capacity,
        dist_to_52w_high_pct=dist_to_52w_high_pct,
        breakout_detected=breakout_detected,
    )

    risk = entry_mid - stop
    rr_tp1 = (tp1 - entry_mid) / risk if risk > 0 else float("nan")
    rr_tp2 = (tp2 - entry_mid) / risk if risk > 0 else float("nan")

    trend_text = (
        "güçlü"
        if (trend_stack_ok and (price_above_ema150 or price_near_ema150))
        else ("zayıf" if close < ema200 else "karışık")
    )
    mom_text = (
        "sağlıklı" if 55 <= rsi14 <= 75
        else ("ısınmış" if rsi14 > 75 else "zayıf/sınır")
    )
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
            "Senaryo: Fiyat EMA50'ye göre uzamış. Pullback gelmeden giriş riskli. En iyi plan: giriş bandına yaklaşmasını "
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
        f"(ATR/impuls/52W tavanı ile sınırlandı). "
        f"TP tavan: TP1≤%{targets_dbg.get('tp1_cap_pct', 0):.0f} / TP2≤%{targets_dbg.get('tp2_cap_pct', 0):.0f}"
    )

    stop_reason = (
        f"Stop (aktif): noise(ATR) + yapısal(invalidation:{stop_dbg.get('inv_src')}) + dinamik_cap(%{stop_dbg.get('max_risk_pct'):.0f}) "
        f"(capped={stop_dbg.get('capped')}). "
        f"Yapısal={stop_structural:.2f} | Noise={stop_noise:.2f}"
    )

    vol_warning_text = ""
    if high_vol_warning:
        vol_warning_text = (
            f"\n⚠️ **Yüksek Volatilite Uyarısı:** ATR% yüksek ({atr_pct:.1f}%). "
            "Stop cap devreye girdi — gerçek yapısal stop daha aşağıda olabilir. "
            "Pozisyon boyunu buna göre küçült."
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
        f"**RSI Yönü (Son {RSI_MOMENTUM_LOOKBACK} Bar):** {rsi_dir_label} (eğim={rsi_slope_val:.2f})  \n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}  \n"
        f"**Uzama (EMA50 mesafe):** %{dist_ema50_pct:.2f} → {'uzamış' if extended else 'normal'}  \n\n"
        f"**Minervini #5:** 52W dip={low_52w:.2f} → {'✅ geçiyor' if m5_ok else '❌ geçmiyor'}  \n"
        f"**52W Zirveye Uzaklık:** %{dist_to_52w_high_pct:.1f} ({'+' if near_52w_pts > 0 else ''}{near_52w_pts} puan)  \n\n"
        f"**Zamanlama:** **{timing_cmd}**  \n"
        f"**Giriş Bölgesi:** {entry_low:.2f} – {entry_high:.2f}  \n"
        f"**Giriş Bölgesine Mesafe:** {dist_entry_pct:+.2f}%  \n"
        f"**Takip Seviyesi:** {watch_level:.2f}  \n\n"
        f"**Stop:** {stop:.2f}  \n"
        f"**TP1:** {tp1:.2f}  (R/R≈1:{rr_tp1:.2f})  \n"
        f"**TP2:** {tp2:.2f}  (R/R≈1:{rr_tp2:.2f})  \n"
        f"{targets_reason}  \n"
        f"{stop_reason}"
        f"{vol_warning_text}"
    )

    debug = {
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema150": ema150,
        "ema200": ema200,
        "rsi14": rsi14,
        "rsi_slope": rsi_slope_val,
        "rsi_direction": rsi_dir_label,
        "rsi_dir_pts": rsi_dir_pts,
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
        "dist_to_52w_high_pct": dist_to_52w_high_pct,
        "near_52w_pts": near_52w_pts,
        "minervini5_ok": m5_ok,
        "pivot_low": pivot_low,
        "stop_debug": stop_dbg,
        "stop_structural": stop_structural,
        "stop_noise": stop_noise,
        "high_vol_warning": high_vol_warning,
        "targets_debug": targets_dbg,
        "capacity": capacity,
        "base_detected": base_detected,
        "breakout_detected": breakout_detected,
        "base_bonus_pts": base_bonus_pts,
        "breakout_bonus_pts": breakout_bonus_pts,
        "base_breakout_details": base_result["details"],
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
        base_detected=bool(base_detected),
        breakout_detected=bool(breakout_detected),
        base_bonus_pts=int(base_bonus_pts),
        breakout_bonus_pts=int(breakout_bonus_pts),
        rsi_slope_val=float(rsi_slope_val) if np.isfinite(rsi_slope_val) else float("nan"),
        rsi_direction_label=rsi_dir_label,
        high_vol_warning=high_vol_warning,
        dist_to_52w_high_pct=float(dist_to_52w_high_pct) if np.isfinite(dist_to_52w_high_pct) else float("nan"),
        narrative=narrative,
        scenario=scenario,
        debug=debug,
        breakdown=breakdown,
    )


# =========================================================

# =========================================================
# PDF EXPORT (Tek Hisse)  — WeasyPrint
# =========================================================

_PDF_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Helvetica Neue',Arial,sans-serif; font-size:10px; color:#0F172A; background:#fff; }
.header { background:#0F172A; display:flex; align-items:stretch; height:80px; }
.header-logo { background:#0F172A; display:flex; align-items:center;
  justify-content:center; padding:0 14px; border-right:1px solid #1E293B; flex-shrink:0; }
.header-logo img { height:60px; width:auto; }
.header-body { flex:1; display:flex; align-items:center;
  justify-content:space-between; padding:0 24px; }
.header-title { font-size:14px; font-weight:700; color:#F1F5F9; margin-bottom:5px; }
.header-badge { display:inline-block; background:#1E3A5F; color:#93C5FD;
  font-size:9px; font-weight:700; padding:2px 10px; border-radius:12px;
  border:1px solid #2563EB; margin-right:5px; }
.header-meta { text-align:right; color:#64748B; font-size:8px; line-height:1.9; }
.header-meta strong { color:#94A3B8; }
.content { padding:14px 24px 20px; }
.status-band { padding:8px 14px; border-radius:5px; margin-bottom:10px; border-left:4px solid; }
.status-label { font-size:10px; font-weight:700; text-transform:uppercase; }
.warning-box { background:#FFF7ED; border-left:3px solid #F97316; padding:5px 11px;
  border-radius:4px; font-size:8px; font-weight:700; color:#9A3412; margin-bottom:9px; }
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:5px; margin-bottom:12px; }
.kpi-card { background:#F8FAFC; border:1px solid #CBD5E1; border-radius:5px; padding:8px 9px; }
.kpi-label { font-size:7px; font-weight:700; color:#64748B; text-transform:uppercase;
  letter-spacing:.4px; margin-bottom:3px; }
.kpi-value { font-size:13px; font-weight:800; color:#0F172A; line-height:1.1; }
.bar-wrap { display:flex; margin-top:5px; border-radius:3px; overflow:hidden; height:7px; }
.bar-fill { height:7px; }
.bar-empty { height:7px; background:#CBD5E1; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
.box { border:1px solid #E2E8F0; border-radius:5px; padding:9px; }
.sec-title { font-size:8px; font-weight:700; color:#1E40AF; text-transform:uppercase;
  letter-spacing:.5px; border-bottom:1.5px solid #BFDBFE; padding-bottom:3px; margin-bottom:7px; }
table { width:100%; border-collapse:collapse; }
table th { background:#EFF6FF; font-size:7.5px; font-weight:700; color:#1E40AF;
  text-align:left; padding:4px 7px; }
table td { font-size:8.5px; padding:4px 7px; border-bottom:1px solid #F1F5F9;
  color:#1E293B; vertical-align:middle; }
table tr:last-child td { border-bottom:none; }
table tr:nth-child(even) td { background:#F8FAFC; }
.td-param { color:#64748B; font-weight:600; width:42%; }
.td-val { color:#0F172A; font-weight:700; }
.td-red { color:#991B1B; font-weight:700; }
.td-green { color:#166534; font-weight:700; }
.td-muted { color:#94A3B8; }
.td-right { text-align:right; }
.score-bar { width:65px; height:7px; background:#E2E8F0; border-radius:3px;
  border:1px solid #CBD5E1; overflow:hidden; display:inline-block; vertical-align:middle; }
.score-bar-fill { height:100%; background:#3B82F6; }
.score-bar-bonus { background:#10B981; }
.scenario-box { background:#F0F9FF; border-left:3px solid #3B82F6;
  border-radius:4px; padding:9px 12px; font-size:8.5px; line-height:1.6; color:#1E3A5F; }
.footer { margin-top:12px; padding-top:6px; border-top:1px solid #E2E8F0;
  display:flex; justify-content:space-between; font-size:7px; color:#94A3B8; }
"""


def _status_colors_wp(tag):
    t = tag.upper()
    if "ALIM" in t:       return "#166534", "#DCFCE7"
    if "PULLBACK" in t or "BEKLE" in t: return "#92400E", "#FEF9C3"
    if "KONSOL" in t:     return "#1E40AF", "#DBEAFE"
    if "BOZULDU" in t or "UZAMI" in t or "52W" in t: return "#991B1B", "#FEE2E2"
    return "#475569", "#F1F5F9"


def _header_html(logo_b64, ticker, interval_label, bars, date_str):
    logo = (f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else "")
    return (
        '<div class="header">'
        f'<div class="header-logo">{logo}</div>'
        '<div class="header-body">'
        "<div>"
        '<div class="header-title">Teknik Analiz Raporu</div>'
        "<div>"
        f'<span class="header-badge">{html.escape(ticker)}</span>'
        f'<span class="header-badge" style="background:#1E293B;color:#94A3B8;border-color:#334155">'
        f'{html.escape(interval_label)}</span>'
        "</div></div>"
        '<div class="header-meta">'
        f"Bar: <strong>{bars}</strong><br>"
        f"<strong>{html.escape(date_str)}</strong>"
        "</div></div></div>"
    )


def _bar(pct, color="#3B82F6"):
    p = max(0, min(100, round(pct)))
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill" style="width:{p}%;background:{color}"></div>'
        f'<div class="bar-empty" style="width:{100-p}%"></div>'
        "</div>"
    )


def _score_bar(pts, mx, bonus=False):
    p = int(pts / mx * 100) if mx else 0
    cls = "score-bar-bonus" if bonus else ""
    return (
        f'<div class="score-bar">'
        f'<div class="score-bar-fill {cls}" style="width:{p}%"></div>'
        "</div>"
    )


def _render_pdf_wp(body_html):
    if not _WEASYPRINT_OK:
        raise RuntimeError("weasyprint kurulu degil -- pip install weasyprint")
    doc = (
        '<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">'
        f"<style>{_PDF_CSS}</style></head><body>{body_html}</body></html>"
    )
    return _WeasyprintHTML(string=doc, base_url="/").write_pdf()


def build_pdf_bytes_single(
    ticker: str,
    interval_label: str,
    bars: int,
    plan: "TradePlan",
    quote: "dict | None",
    logo_b64_str: str = "",
) -> bytes:
    sc, sb    = _status_colors_wp(plan.status_tag)
    date_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    close_val = plan.debug.get("close", float("nan"))

    def scolor(v):
        return "#166534" if v >= 75 else ("#1D4ED8" if v >= 50 else "#64748B")

    # KPI Grid
    kpi_items = [
        ("Guncel Fiyat",  f"${close_val:.2f}" if np.isfinite(close_val) else "---", None, None),
        ("Toplam Skor",   f"{plan.total_score}/100",  plan.total_score, "#3B82F6"),
        ("Setup",         f"{plan.setup_score}/100",  plan.setup_score, "#3B82F6"),
        ("Timing",        f"{plan.timing_score}/100", plan.timing_score,"#166534"),
        ("Kapasite",      plan.capacity_level,         None, None),
        ("Minervini #5",  "GECTI" if plan.minervini5_ok else "GECMEDI", None, None),
    ]
    m5_color = "#166534" if plan.minervini5_ok else "#991B1B"
    kpi_html = '<div class="kpi-grid">'
    for i, (lbl, val, bar_pct, bar_c) in enumerate(kpi_items):
        vc = scolor(bar_pct) if bar_pct is not None else (m5_color if i == 5 else "#0F172A")
        bar = _bar(bar_pct, bar_c) if bar_pct is not None else ""
        kpi_html += (
            '<div class="kpi-card">'
            f'<div class="kpi-label">{lbl}</div>'
            f'<div class="kpi-value" style="color:{vc}">{html.escape(str(val))}</div>'
            f"{bar}</div>"
        )
    kpi_html += "</div>"

    # Warning
    warn = (
        '<div class="warning-box">&#9888; Yuksek volatilite &mdash; '
        "stop cap devrede, pozisyon boyunu kucult.</div>"
        if plan.high_vol_warning else ""
    )

    # Plan table
    rr1  = f"1:{plan.rr_tp1:.2f}" if np.isfinite(plan.rr_tp1) else "---"
    rr2  = f"1:{plan.rr_tp2:.2f}" if np.isfinite(plan.rr_tp2) else "---"
    dist = f"%{plan.dist_to_52w_high_pct:.1f}" if np.isfinite(plan.dist_to_52w_high_pct) else "---"
    plan_rows = [
        ("Giris Bolgesi", f"${plan.entry_low:.2f} &ndash; ${plan.entry_high:.2f}"),
        ("Stop",          f'<span class="td-red">${plan.stop:.2f}</span>'),
        ("TP1",           f'<span class="td-green">${plan.tp1:.2f}</span> <span class="td-muted">({rr1})</span>'),
        ("TP2",           f'<span class="td-green">${plan.tp2:.2f}</span> <span class="td-muted">({rr2})</span>'),
        ("52W Dip/Zirve", f"${plan.low_52w:.2f} / ${plan.high_52w:.2f}" if np.isfinite(plan.low_52w) else "---"),
        ("Zirve Uzaklik", dist),
        ("RSI Yonu",      html.escape(plan.rsi_direction_label)),
        ("Dar Baz",       "Var" if plan.base_detected else "Yok"),
        ("Kirilim",       "Var" if plan.breakout_detected else "Yok"),
    ]
    plan_tbl = (
        "<table><tr><th>Parametre</th><th>Deger</th></tr>"
        + "".join(f'<tr><td class="td-param">{r}</td><td class="td-val">{v}</td></tr>' for r, v in plan_rows)
        + "</table>"
    )

    # Score table
    b = plan.breakdown
    skor_items = [
        ("Trend",           b.trend_stack,        30, False),
        ("Fiyat/EMA150",    b.price_vs_ema150,    20, False),
        ("Momentum (RSI)",  b.momentum_rsi,       20, False),
        ("Volatilite(ATR)", b.volatility_atr,     15, False),
        ("Uzama (EMA50)",   b.extension_vs_ema50, 15, False),
        ("52W Zirve",       b.near_52w_high,      10, False),
        ("RSI Yonu",        b.rsi_direction,       5, False),
        ("Dar Baz(bonus)",  b.base_bonus,          7, True),
        ("Kirilim(bonus)",  b.breakout_bonus,      8, True),
    ]
    skor_tbl = (
        "<table><tr><th>Bilesen</th>"
        '<th class="td-right">Puan</th><th class="td-right">Maks</th><th>Ilerleme</th></tr>'
        + "".join(
            f"<tr><td>{html.escape(n)}</td>"
            f'<td class="td-right td-val">{p}</td>'
            f'<td class="td-right td-muted">{m}</td>'
            f"<td>{_score_bar(p, m, bonus)}</td></tr>"
            for n, p, m, bonus in skor_items
        )
        + "</table>"
    )

    scenario = html.escape(plan.scenario.replace("**", ""))

    # Quote
    q_html = ""
    if quote and isinstance(quote, dict):
        q_rows = "".join(
            f'<tr><td class="td-param">{html.escape(k)}</td><td class="td-val">{html.escape(str(quote[k]))}</td></tr>'
            for k in ["name","exchange","currency","price","change","percent_change","previous_close"]
            if k in quote
        )
        if q_rows:
            q_html = (
                '<div class="box" style="margin-bottom:12px">'
                '<div class="sec-title">Quote</div>'
                f"<table><tr><th>Alan</th><th>Deger</th></tr>{q_rows}</table></div>"
            )

    body = (
        _header_html(logo_b64_str, ticker, interval_label, bars, date_str)
        + '<div class="content">'
        + f'<div class="status-band" style="border-color:{sc};background:{sb}">'
        + f'<div class="status-label" style="color:{sc}">{html.escape(plan.status_tag)}</div></div>'
        + warn + kpi_html
        + '<div class="two-col">'
        + f'<div class="box"><div class="sec-title">Islem Plani</div>{plan_tbl}</div>'
        + f'<div class="box"><div class="sec-title">Skor Dagilimi</div>{skor_tbl}</div>'
        + "</div>"
        + '<div class="box" style="margin-bottom:12px">'
        + '<div class="sec-title">Senaryo</div>'
        + f'<div class="scenario-box">{scenario}</div></div>'
        + q_html
        + '<div class="footer">'
        + "<span>MinerWin V6.2 &nbsp;&middot;&nbsp; Minervini-Based Technical Trading Engine</span>"
        + "<span>Bu rapor otomatik teknik analiz amaciyla uretilmistir. Yatirim tavsiyesi degildir.</span>"
        + "</div></div>"
    )
    return _render_pdf_wp(body)


# GRAFİK
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
# LİDERLİK MODÜLÜ
# =========================================================
@st.cache_data(ttl=300)
def _fetch_spy_daily(outputsize: int = 320) -> pd.DataFrame:
    payload = td_time_series("SPY", "1day", int(outputsize))
    return parse_ohlcv(payload)


def _aligned_ratio_series(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.Series:
    s = stock_df[["time", "close"]].dropna().copy().rename(columns={"close": "s_close"}).set_index("time")
    m = spy_df[["time", "close"]].dropna().copy().rename(columns={"close": "m_close"}).set_index("time")
    j = s.join(m, how="inner")
    if j.empty:
        return pd.Series(dtype=float)
    rs = j["s_close"] / j["m_close"]
    rs.name = "rs_line"
    return rs


def _perf_pct_over_days(df: pd.DataFrame, days: int) -> float:
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")
    c = df["close"].dropna().astype(float)
    if len(c) <= days:
        return float("nan")
    start = float(c.iloc[-(days + 1)])
    end = float(c.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start - 1.0) * 100.0


def analyze_volume_profile(daily_df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "vol_ma50": float("nan"),
        "vol_last10": float("nan"),
        "dryup_ratio": float("nan"),
        "dryup_ok": False,
        "breakout_ok": False,
        "pivot_level": float("nan"),
        "volume_today": float("nan"),
    }
    if daily_df is None or daily_df.empty:
        return out

    d = daily_df.copy()
    if "volume" not in d.columns:
        d["volume"] = 0.0

    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)

    v = d["volume"].astype(float).fillna(0.0)
    if len(v.dropna()) < 60:
        return out

    vol_ma50 = float(v.rolling(50).mean().iloc[-1])
    vol_last10 = float(v.tail(10).mean())
    volume_today = float(v.iloc[-1])

    out["vol_ma50"] = vol_ma50
    out["vol_last10"] = vol_last10
    out["volume_today"] = volume_today

    if np.isfinite(vol_ma50) and vol_ma50 > 0:
        out["dryup_ratio"] = float(vol_last10 / vol_ma50)

    last = d.iloc[-1]
    close_ = float(last["close"])
    ema20_ = float(last["ema20"])
    ema50_ = float(last["ema50"])
    band_lo = min(ema20_, ema50_)
    band_hi = max(ema20_, ema50_)
    in_band = (
        np.isfinite(close_) and np.isfinite(band_lo) and np.isfinite(band_hi)
        and (band_lo <= close_ <= band_hi)
    )

    if in_band and np.isfinite(out["dryup_ratio"]):
        out["dryup_ok"] = bool(out["dryup_ratio"] <= DRYUP_RATIO_THRESHOLD)

    if len(d) >= 25:
        pivot = float(d["high"].astype(float).rolling(20).max().shift(1).iloc[-1])
        out["pivot_level"] = pivot
        if np.isfinite(pivot) and close_ > pivot and np.isfinite(vol_ma50) and vol_ma50 > 0:
            out["breakout_ok"] = bool(volume_today >= BREAKOUT_VOL_MULTIPLIER * vol_ma50)

    return out


def analyze_relative_strength(daily_df: pd.DataFrame, spy_df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "rs_line_new_high_60d": False,
        "rs_rating": float("nan"),
        "edge_3m": float("nan"),
        "edge_6m": float("nan"),
        "edge_12m": float("nan"),
    }
    if daily_df is None or daily_df.empty or spy_df is None or spy_df.empty:
        return out

    rs_line = _aligned_ratio_series(daily_df, spy_df)
    if len(rs_line) >= 70:
        window = rs_line.tail(60)
        out["rs_line_new_high_60d"] = bool(window.iloc[-1] >= window.max())

    stock_3m = _perf_pct_over_days(daily_df, 63)
    spy_3m = _perf_pct_over_days(spy_df, 63)
    stock_6m = _perf_pct_over_days(daily_df, 126)
    spy_6m = _perf_pct_over_days(spy_df, 126)
    stock_12m = _perf_pct_over_days(daily_df, 252)
    spy_12m = _perf_pct_over_days(spy_df, 252)

    out["edge_3m"] = stock_3m - spy_3m if (np.isfinite(stock_3m) and np.isfinite(spy_3m)) else float("nan")
    out["edge_6m"] = stock_6m - spy_6m if (np.isfinite(stock_6m) and np.isfinite(spy_6m)) else float("nan")
    out["edge_12m"] = stock_12m - spy_12m if (np.isfinite(stock_12m) and np.isfinite(spy_12m)) else float("nan")

    score = 50.0
    for edge, w in [(out["edge_3m"], 0.30), (out["edge_6m"], 0.35), (out["edge_12m"], 0.35)]:
        if np.isfinite(edge):
            score += clamp(edge, -30.0, 30.0) * w

    if out["rs_line_new_high_60d"]:
        score += 8.0

    out["rs_rating"] = float(clamp(score, 0.0, 100.0))
    return out


def analyze_52w_high_proximity(price: float, high_52w: float) -> Dict[str, Any]:
    out = {"dist_to_52w_high_pct": float("nan"), "near_high_ok": False}
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return out
    dist = ((high_52w - price) / high_52w) * 100.0
    out["dist_to_52w_high_pct"] = float(dist)
    out["near_high_ok"] = bool(dist <= NEAR_HIGH_THRESHOLD * 100)
    return out


def leadership_pack(
    symbol: str,
    interval: str,
    df_interval: pd.DataFrame,
    low_52w: float,
    high_52w: float,
    spy_df: pd.DataFrame | None = None,
    daily_df_override: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    try:
        if daily_df_override is not None and not daily_df_override.empty:
            daily_df = daily_df_override
        else:
            daily_df = df_interval if interval == "1day" else _fetch_daily_df(symbol, 320)
    except Exception:
        daily_df = pd.DataFrame()

    if spy_df is None or spy_df.empty:
        try:
            spy_df = _fetch_spy_daily(320)
        except Exception:
            spy_df = pd.DataFrame()

    vol = analyze_volume_profile(daily_df) if not daily_df.empty else {}
    rs = analyze_relative_strength(daily_df, spy_df) if (not daily_df.empty and not spy_df.empty) else {}
    price = float(df_interval.iloc[-1]["close"]) if (df_interval is not None and not df_interval.empty) else float("nan")
    near = analyze_52w_high_proximity(price, high_52w)

    leader = "—"
    rr = rs.get("rs_rating", np.nan)
    if np.isfinite(rr):
        if rr >= 70 and rs.get("rs_line_new_high_60d"):
            leader = "LİDER ADAYI"
        elif rr >= 60:
            leader = "ORTA"
        else:
            leader = "ZAYIF"

    return {
        "leader_label": leader,
        "rs_rating": rr,
        "rs_new_high_60d": bool(rs.get("rs_line_new_high_60d", False)),
        "edge_3m": rs.get("edge_3m", np.nan),
        "edge_6m": rs.get("edge_6m", np.nan),
        "edge_12m": rs.get("edge_12m", np.nan),
        "dryup_ok": bool(vol.get("dryup_ok", False)),
        "breakout_ok": bool(vol.get("breakout_ok", False)),
        "dryup_ratio": vol.get("dryup_ratio", np.nan),
        "dist_to_52w_high_pct": near.get("dist_to_52w_high_pct", np.nan),
        "near_high_ok": bool(near.get("near_high_ok", False)),
    }


# =========================================================
# PORTFÖY YARDIMCILARI
# =========================================================
def rolling_52w_levels(df: pd.DataFrame, bars_1day: int = 260) -> Tuple[float, float]:
    return compute_52w_levels(df, bars_1day)


def is_blue_sky(price: float, high_52w: float, threshold: float = BLUE_SKY_THRESHOLD) -> bool:
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return False
    return price >= (threshold * high_52w)


def trailing_structure_status(price: float, ema20: float, ema50: float) -> Tuple[str, str]:
    if not (np.isfinite(price) and np.isfinite(ema20) and np.isfinite(ema50)):
        return ("—", "İz süren yapı için veri eksik.")

    above20 = price >= ema20
    above50 = price >= ema50

    if above20 and above50:
        return ("İz süren yapı korunuyor.", "EMA20: ÜZERİNDE | EMA50: ÜZERİNDE")
    if (not above20) and above50:
        return ("Kısa vadeli iz süren yapı zayıflıyor.", "EMA20: ALTINDA | EMA50: ÜZERİNDE")
    return ("İz süren yapı bozulma sinyali veriyor.", "EMA20: ALTINDA | EMA50: ALTINDA")


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
# PORTFÖY KPI
# =========================================================
def compute_portfolio_kpis(out: pd.DataFrame) -> Dict[str, float]:
    k = {
        "portfolio_value": np.nan,
        "cost_basis": np.nan,
        "pnl_value": np.nan,
        "pnl_pct": np.nan,
        "max_profit_tp1": np.nan,
        "max_loss_stop": np.nan,
    }
    if out is None or out.empty:
        return k

    needed = ["Qty", "Fiyat", "Alış Ort.", "Stop", "TP1"]
    for c in needed:
        if c not in out.columns:
            return k

    df = out.copy()

    def to_num(x):
        try:
            if x == "" or x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    df["Qty_n"] = df["Qty"].apply(to_num)
    df["Price_n"] = df["Fiyat"].apply(to_num)
    df["Avg_n"] = df["Alış Ort."].apply(to_num)
    df["Stop_n"] = df["Stop"].apply(to_num)
    df["TP1_n"] = df["TP1"].apply(to_num)

    valid = df[np.isfinite(df["Qty_n"]) & (df["Qty_n"] > 0) & np.isfinite(df["Price_n"])].copy()
    if valid.empty:
        return k

    valid["pos_value"] = valid["Qty_n"] * valid["Price_n"]
    k["portfolio_value"] = float(valid["pos_value"].sum())

    valid_cost = valid[np.isfinite(valid["Avg_n"]) & (valid["Avg_n"] > 0)].copy()
    if not valid_cost.empty:
        valid_cost["cost_value"] = valid_cost["Qty_n"] * valid_cost["Avg_n"]
        k["cost_basis"] = float(valid_cost["cost_value"].sum())

        pnl_val = float((valid_cost["pos_value"] - valid_cost["cost_value"]).sum())
        k["pnl_value"] = pnl_val

        if k["cost_basis"] and np.isfinite(k["cost_basis"]) and k["cost_basis"] != 0:
            k["pnl_pct"] = float((pnl_val / k["cost_basis"]) * 100.0)

        tp1v = valid_cost[np.isfinite(valid_cost["TP1_n"]) & (valid_cost["TP1_n"] > 0)].copy()
        if not tp1v.empty:
            k["max_profit_tp1"] = float((tp1v["Qty_n"] * (tp1v["TP1_n"] - tp1v["Avg_n"])).sum())

        stv = valid_cost[np.isfinite(valid_cost["Stop_n"]) & (valid_cost["Stop_n"] > 0)].copy()
        if not stv.empty:
            raw_stop_pnl = float((stv["Qty_n"] * (stv["Stop_n"] - stv["Avg_n"])).sum())
            k["max_loss_stop"] = abs(raw_stop_pnl)

    return k


# =========================================================

# =========================================================
# PDF EXPORT (Portföy)  — WeasyPrint
# =========================================================
def build_portfolio_pdf_bytes(
    title: str,
    out: "pd.DataFrame",
    kpis: "Dict[str, float]",
    interval_label: str,
    bars: int,
    logo_b64_str: str = "",
) -> bytes:
    sc, sb   = "#1E40AF", "#DBEAFE"
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def fm(v): return f"${v:,.0f}" if np.isfinite(v) else "---"
    def fp(v): return f"%{v:.1f}"  if np.isfinite(v) else "---"

    pv   = kpis.get("portfolio_value", float("nan"))
    pnlv = kpis.get("pnl_value",      float("nan"))
    pnlp = kpis.get("pnl_pct",        float("nan"))
    mxp  = kpis.get("max_profit_tp1", float("nan"))
    mxl  = kpis.get("max_loss_stop",  float("nan"))
    pc   = "#166534" if (np.isfinite(pnlv) and pnlv >= 0) else "#991B1B"

    kpi_html = (
        '<div class="kpi-grid" style="grid-template-columns:repeat(5,1fr)">'
        + f'<div class="kpi-card"><div class="kpi-label">Portfoy Degeri</div><div class="kpi-value">{fm(pv)}</div></div>'
        + f'<div class="kpi-card"><div class="kpi-label">Anlik P&L ($)</div><div class="kpi-value" style="color:{pc}">{fm(pnlv)}</div></div>'
        + f'<div class="kpi-card"><div class="kpi-label">Anlik P&L (%)</div><div class="kpi-value" style="color:{pc}">{fp(pnlp)}</div></div>'
        + f'<div class="kpi-card"><div class="kpi-label">Max Kar TP1</div><div class="kpi-value" style="color:#166534">{fm(mxp)}</div></div>'
        + f'<div class="kpi-card"><div class="kpi-label">Max Zarar Stop</div><div class="kpi-value" style="color:#991B1B">{fm(mxl)}</div></div>'
        + "</div>"
    )

    tbl_html = '<p style="color:#64748B;font-size:9px">Pozisyon tablosu bos.</p>'
    if out is not None and not out.empty:
        preferred = [
            "Ticker","Fiyat","Qty","Alis Ort.","P&L %",
            "Stop","TP1","TP2","Setup","Timing",
            "Durum","RS Rating","52W Zirve Uzaklik %","Blue Sky","RSI Yonu",
        ]
        col_map = {
            "Alış Ort.":          "Alis Ort.",
            "52W Zirve Uzaklık %": "52W Zirve Uzaklik %",
            "RSI Yönü":           "RSI Yonu",
        }
        df = out.rename(columns=col_map).copy()
        cols = [c for c in preferred if c in df.columns]
        df = df[cols]

        def cell(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "---"
            s = str(v)
            return s.encode("ascii","ignore").decode() or s

        th = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
        tr = "".join(
            "<tr>" + "".join(f"<td>{cell(row[c])}</td>" for c in df.columns) + "</tr>"
            for _, row in df.iterrows()
        )
        tbl_html = f'<table style="font-size:8px"><tr>{th}</tr>{tr}</table>'

    body = (
        _header_html(logo_b64_str, "Portfoy", interval_label, bars, date_str)
        + '<div class="content">'
        + f'<div class="status-band" style="border-color:{sc};background:{sb}">'
        + f'<div class="status-label" style="color:{sc}">Portfoy Ozeti</div></div>'
        + kpi_html
        + '<div class="box" style="margin-bottom:12px">'
        + '<div class="sec-title">Pozisyonlar</div>'
        + tbl_html + "</div>"
        + '<div class="footer">'
        + "<span>MinerWin V6.2 &nbsp;&middot;&nbsp; Minervini-Based Technical Trading Engine</span>"
        + "<span>Bu rapor otomatik teknik analiz amaciyla uretilmistir. Yatirim tavsiyesi degildir.</span>"
        + "</div></div>"
    )
    return _render_pdf_wp(body)


# EXCEL EXPORT (Portföy)
# =========================================================
def build_portfolio_excel_bytes(
    title: str,
    out: pd.DataFrame,
    kpis: Dict[str, float],
    interval_label: str,
    bars: int,
) -> bytes:
    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Özet"
    ws_pos = wb.create_sheet("Pozisyonlar")

    font_title = Font(name="Calibri", size=16, bold=True, color="0F172A")
    font_sub = Font(name="Calibri", size=10, color="475569")
    font_hdr = Font(name="Calibri", size=11, bold=True, color="0F172A")
    font_body = Font(name="Calibri", size=10, color="111827")
    font_kpi = Font(name="Calibri", size=14, bold=True, color="0F172A")
    font_kpi_lbl = Font(name="Calibri", size=10, bold=True, color="475569")

    fill_hdr = PatternFill("solid", fgColor="F1F5F9")
    fill_card = PatternFill("solid", fgColor="FFFFFF")

    thin = Side(style="thin", color="E5E7EB")
    border_thin = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws_sum["A1"] = title
    ws_sum["A1"].font = font_title
    ws_sum["A2"] = f"Zaman dilimi: {interval_label}  |  Bar: {bars}  |  Oluşturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ws_sum["A2"].font = font_sub

    cards = [
        ("Portföy Değeri", kpis.get("portfolio_value", np.nan), "money"),
        ("Anlık P&L ($)", kpis.get("pnl_value", np.nan), "money"),
        ("Anlık P&L (%)", kpis.get("pnl_pct", np.nan), "pct"),
        ("TP1 Hepsi Olursa (Max Kâr)", kpis.get("max_profit_tp1", np.nan), "money"),
        ("Stop Hepsi Olursa (Max Zarar)", kpis.get("max_loss_stop", np.nan), "money"),
        ("Not", "Hesaplar Qty ve Alış Ort. girilmiş satırlarda geçerlidir.", "text"),
    ]
    pos = [("A4", "C6"), ("E4", "G6"), ("I4", "K6"), ("A8", "C10"), ("E8", "G10"), ("I8", "K10")]

    for (lbl, val, kind), (tl, br) in zip(cards, pos):
        tl_col = tl[0]
        tl_row = int(tl[1:])
        br_col = br[0]
        br_row = int(br[1:])

        for r in range(tl_row, br_row + 1):
            for c in range(ord(tl_col), ord(br_col) + 1):
                cell = ws_sum[f"{chr(c)}{r}"]
                cell.fill = fill_card
                cell.border = border_thin

        ws_sum[tl] = lbl
        ws_sum[tl].font = font_kpi_lbl
        ws_sum[tl].alignment = Alignment(horizontal="left", vertical="top")

        value_cell = f"{tl_col}{tl_row+1}"
        if kind == "money":
            ws_sum[value_cell] = float(val) if np.isfinite(val) else ""
            ws_sum[value_cell].number_format = '#,##0.00'
            ws_sum[value_cell].font = font_kpi
            ws_sum[value_cell].alignment = Alignment(horizontal="left", vertical="center")
        elif kind == "pct":
            ws_sum[value_cell] = (float(val) / 100.0) if np.isfinite(val) else ""
            ws_sum[value_cell].number_format = '0.00%'
            ws_sum[value_cell].font = font_kpi
            ws_sum[value_cell].alignment = Alignment(horizontal="left", vertical="center")
        else:
            ws_sum[value_cell] = str(val)
            ws_sum[value_cell].font = font_sub
            ws_sum[value_cell].alignment = Alignment(wrap_text=True, vertical="top")

    for col, w in [("A", 22), ("B", 16), ("C", 16), ("E", 22), ("F", 16), ("G", 16), ("I", 22), ("J", 16), ("K", 16)]:
        ws_sum.column_dimensions[col].width = w
    ws_sum.row_dimensions[1].height = 24
    ws_sum.row_dimensions[2].height = 16

    if out is None or out.empty:
        ws_pos["A1"] = "Pozisyon tablosu boş."
        ws_pos["A1"].font = font_body
    else:
        df = out.copy()
        preferred_cols = [
            "Ticker", "Fiyat", "Qty", "Alış Ort.", "P&L %", "Stop", "Stop Mesafe %",
            "TP1", "TP1 Mesafe %", "TP2", "TP2 Mesafe %",
            "R (TP1/Stop)", "R (TP2/Stop)",
            "Setup", "Timing", "Durum", "Minervini #5", "Liderlik", "RS Rating", "RS Yeni Zirve",
            "Endekse Üstünlük 3A", "Hacim Kuruması", "Kuruma Oranı", "52W Zirve Uzaklık %",
            "Blue Sky", "İz Süren Yapı", "RSI Yönü", "Yüksek Vol Uyarı",
            "Poz. Değeri", "Risk $",
            "Aksiyon", "Not"
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        df = df[cols].copy()

        header_row = 1
        for ci, col_name in enumerate(df.columns, start=1):
            cell = ws_pos.cell(row=header_row, column=ci, value=col_name)
            cell.font = font_hdr
            cell.fill = fill_hdr
            cell.border = border_thin
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for ri in range(df.shape[0]):
            for ci, col_name in enumerate(df.columns, start=1):
                v = df.iloc[ri, ci - 1]
                cell = ws_pos.cell(row=header_row + 1 + ri, column=ci, value=v)
                cell.font = font_body
                cell.border = border_thin
                if col_name in ("Not", "Durum", "İz Süren Yapı", "Aksiyon", "RSI Yönü"):
                    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)

        num_money = {"Fiyat", "Alış Ort.", "Stop", "TP1", "TP2", "Poz. Değeri", "Risk $"}
        num_pct = {"P&L %", "Stop Mesafe %", "TP1 Mesafe %", "TP2 Mesafe %"}
        num_rr = {"R (TP1/Stop)", "R (TP2/Stop)"}
        num_int = {"Setup", "Timing"}

        for ci, col_name in enumerate(df.columns, start=1):
            for r in range(header_row + 1, header_row + 1 + df.shape[0]):
                cell = ws_pos.cell(row=r, column=ci)
                if col_name in num_money:
                    cell.number_format = '#,##0.00'
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif col_name in num_pct:
                    try:
                        vv = float(cell.value)
                        cell.value = vv / 100.0
                        cell.number_format = '0.00%'
                        cell.alignment = Alignment(horizontal="right", vertical="center")
                    except Exception:
                        pass
                elif col_name in num_rr:
                    cell.number_format = '0.00'
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif col_name in num_int:
                    cell.number_format = '0'
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif col_name in ("Ticker", "Blue Sky", "Minervini #5"):
                    cell.alignment = Alignment(horizontal="center", vertical="center")

        if "P&L %" in df.columns:
            pnl_col = df.columns.tolist().index("P&L %") + 1
            col_letter = get_column_letter(pnl_col)
            rng = f"{col_letter}{header_row+1}:{col_letter}{header_row+df.shape[0]}"
            ws_pos.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"], font=Font(color="166534", bold=True)))
            ws_pos.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"], font=Font(color="991B1B", bold=True)))

        ws_pos.freeze_panes = "A2"
        ws_pos.auto_filter.ref = f"A1:{get_column_letter(df.shape[1])}{df.shape[0]+1}"

        tab = XLTable(displayName="PozisyonlarTablosu", ref=ws_pos.auto_filter.ref)
        style = TableStyleInfo(name="TableStyleLight9", showFirstColumn=False, showLastColumn=False, showRowStripes=False, showColumnStripes=False)
        tab.tableStyleInfo = style
        ws_pos.add_table(tab)

        for c in range(1, df.shape[1] + 1):
            col_letter = get_column_letter(c)
            name = ws_pos.cell(row=1, column=c).value
            base = 12
            if name in ("Not",):
                base = 44
            elif name in ("Durum", "İz Süren Yapı", "Aksiyon", "RSI Yönü"):
                base = 26
            elif name in ("Ticker", "Blue Sky", "Minervini #5"):
                base = 10
            elif name in ("Poz. Değeri", "Risk $"):
                base = 16
            ws_pos.column_dimensions[col_letter].width = base

        for r in range(2, df.shape[0] + 2):
            ws_pos.row_dimensions[r].height = 18
        ws_pos.row_dimensions[1].height = 22

    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)
    return out_buf.getvalue()


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
# ANA SEKMELER
# =========================================================
tab_single, tab_portfolio = st.tabs(["📈 Tek Hisse Analiz", "🧳 Portföy Analiz"])


# =========================================================
# SEKME 1: TEK HİSSE
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
                df = pd.DataFrame()
                q = {}
                quote_price = None

                with st.spinner("Veri çekiliyor..."):
                    try:
                        payload = td_time_series(ticker, interval, bars)
                        df = parse_ohlcv(payload)
                    except Exception as e:
                        st.error(f"Veri alınamadı: {e}")

                if not df.empty:
                    df["ema20"] = ema(df["close"], 20)
                    df["ema50"] = ema(df["close"], 50)
                    df["ema150"] = ema(df["close"], 150)
                    df["ema200"] = ema(df["close"], 200)
                    df["rsi14"] = rsi(df["close"], 14)
                    df["atr14"] = atr(df, 14)

                    try:
                        low_52w, high_52w, daily_df_for_52w = get_daily_52w_levels(ticker, interval, df)
                    except Exception as e:
                        st.error(f"Daily veri / 52W hesap hatası: {e}")
                        daily_df_for_52w = pd.DataFrame()
                        low_52w, high_52w = float("nan"), float("nan")

                    plan = build_trade_plan(df, low_52w=low_52w, high_52w=high_52w)

                    lead = leadership_pack(
                        ticker,
                        interval,
                        df,
                        low_52w=low_52w,
                        high_52w=high_52w,
                        daily_df_override=daily_df_for_52w,
                    )

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
                    last_price_line = (
                        quote_price
                        if (quote_price is not None and np.isfinite(quote_price))
                        else candle_close
                    )

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
                        "rsi_direction": plan.rsi_direction_label,
                        "dist_to_52w_high_pct": round(float(plan.dist_to_52w_high_pct), 2) if np.isfinite(plan.dist_to_52w_high_pct) else "",
                        "high_vol_warning": plan.high_vol_warning,
                        "entry_low": round(float(plan.entry_low), 4),
                        "entry_high": round(float(plan.entry_high), 4),
                        "stop": round(float(plan.stop), 4),
                        "tp1": round(float(plan.tp1), 4),
                        "tp2": round(float(plan.tp2), 4),
                        "rr_tp1": round(float(plan.rr_tp1), 4) if np.isfinite(plan.rr_tp1) else "",
                        "rr_tp2": round(float(plan.rr_tp2), 4) if np.isfinite(plan.rr_tp2) else "",
                        "capacity": plan.capacity_level,
                    }
                    st.session_state.daily_tests.append(record)
                    try:
                        save_to_history(record)
                    except Exception as e:
                        st.warning(f"history.csv yazılamadı: {e}")

                    st.divider()
                    st.subheader("📊 Strateji Özeti")
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
                        st.caption(
                            f"Yapısal: {plan.debug.get('stop_structural', float('nan')):.2f} | "
                            f"Noise: {plan.debug.get('stop_noise', float('nan')):.2f} | "
                            f"Cap: %{plan.debug.get('stop_debug', {}).get('max_risk_pct', 7):.0f}"
                        )

                    if plan.high_vol_warning:
                        st.warning(
                            f"⚠️ **Yüksek Volatilite:** Stop cap devreye girdi. "
                            f"ATR% yüksek — gerçek yapısal stop daha aşağıda. Pozisyon boyunu küçült."
                        )

                    col_baz, col_kir = st.columns(2)
                    with col_baz:
                        st.metric(
                            "Dar Baz",
                            "✅ Tespit Edildi" if plan.base_detected else "— Yok",
                            help="Son 20 barda ATR daralması + hacim kuruması birlikte varsa baz oluşmuştur."
                        )
                    with col_kir:
                        st.metric(
                            "Pivot Kırılımı",
                            "✅ Kırıldı + Hacim" if plan.breakout_detected else "— Yok",
                            help="Son 20 barın zirvesi kırıldı + hacim 50g ortalamasının %140 üstünde."
                        )

                    col_rsi, col_52w = st.columns(2)
                    with col_rsi:
                        st.metric(
                            f"RSI Yönü (Son {RSI_MOMENTUM_LOOKBACK} Bar)",
                            plan.rsi_direction_label,
                            help="RSI yükseliyorsa momentum artıyor, düşüyorsa zayıflıyor."
                        )
                    with col_52w:
                        dist_label = f"%{plan.dist_to_52w_high_pct:.1f} uzakta" if np.isfinite(plan.dist_to_52w_high_pct) else "—"
                        st.metric("52W Zirveye Uzaklık", dist_label)

                    st.caption(
                        f"Minervini #5: 52W dip={plan.low_52w:.2f} → "
                        f"{'✅ geçiyor' if plan.minervini5_ok else '❌ geçmiyor'} | "
                        f"Kapasite: {plan.capacity_level}"
                    )

                    st.subheader("🏁 Liderlik (Hacim + RS)")
                    cL1, cL2, cL3, cL4 = st.columns(4)
                    cL1.metric("Liderlik", str(lead.get("leader_label", "—")))
                    rsr = lead.get("rs_rating", np.nan)
                    cL2.metric("RS Rating", f"{rsr:.0f}" if np.isfinite(rsr) else "—")
                    cL3.metric("RS Yeni Zirve (60g)", "✅" if lead.get("rs_new_high_60d") else "—")
                    d52 = lead.get("dist_to_52w_high_pct", np.nan)
                    cL4.metric("52W Zirve Uzaklık", f"%{d52:.1f}" if np.isfinite(d52) else "—")

                    with st.expander("Detay (Hacim/RS)", expanded=False):
                        dr = lead.get("dryup_ratio", np.nan)
                        st.write({
                            "Hacim Kuruması": "✅" if lead.get("dryup_ok") else "—",
                            "Kuruma Oranı (10g/50g)": f"{dr:.2f}" if np.isfinite(dr) else "—",
                            "Kırılım Hacmi": "✅" if lead.get("breakout_ok") else "—",
                            "Endekse Üstünlük 3A": f"{lead.get('edge_3m'):+.1f}%" if np.isfinite(lead.get('edge_3m', np.nan)) else "—",
                            "Endekse Üstünlük 6A": f"{lead.get('edge_6m'):+.1f}%" if np.isfinite(lead.get('edge_6m', np.nan)) else "—",
                            "Endekse Üstünlük 12A": f"{lead.get('edge_12m'):+.1f}%" if np.isfinite(lead.get('edge_12m', np.nan)) else "—",
                        })

                    st.subheader("📌 İşlem Planı")
                    table = pd.DataFrame({
                        "Parametre": ["Giriş Bölgesi", "Giriş Mesafesi", "Stop", "TP1", "TP2", "R/R (TP1)", "R/R (TP2)", "Kapasite"],
                        "Değer": [
                            f"{plan.entry_low:.2f} – {plan.entry_high:.2f}",
                            f"{plan.dist_to_entry_pct:+.2f}%",
                            f"{plan.stop:.2f}",
                            f"{plan.tp1:.2f}",
                            f"{plan.tp2:.2f}",
                            f"1 : {plan.rr_tp1:.2f}" if np.isfinite(plan.rr_tp1) else "—",
                            f"1 : {plan.rr_tp2:.2f}" if np.isfinite(plan.rr_tp2) else "—",
                            plan.capacity_level,
                        ],
                    })
                    st.table(table)

                    st.subheader("🧠 Skor Dağılımı")
                    b = plan.breakdown
                    bdf = pd.DataFrame({
                        "Bileşen": [
                            "Trend", "Fiyat/EMA150", "Momentum (RSI)",
                            "Volatilite (ATR%)", "Uzama (EMA50)",
                            "52W Zirve Yakınlığı", "RSI Yönü", "Dar Baz (bonus)", "Pivot Kırılımı (bonus)"
                        ],
                        "Puan": [
                            b.trend_stack, b.price_vs_ema150, b.momentum_rsi,
                            b.volatility_atr, b.extension_vs_ema50,
                            b.near_52w_high, b.rsi_direction, b.base_bonus, b.breakout_bonus
                        ],
                        "Maks": [30, 20, 20, 15, 15, 10, 5, 7, 8],
                    })
                    st.table(bdf)
                    st.caption("Toplam 130 maks → 100'e normalize edilir. Dar baz (+7), pivot kırılımı (+8) bonus puandır. RSI yönü (+5 / 0 / -5) skora dahildir. Minervini #5 geçmezse tavan 55.")

                    st.subheader("🧭 Senaryo")
                    st.write(plan.scenario)

                    st.subheader("📝 Otomatik Teknik Yorum")
                    st.markdown(plan.narrative)

                    if show_quote and q:
                        st.subheader("⚡ Quote (Anlık Özet)")
                        keys = ["symbol", "name", "exchange", "currency", "price", "close", "change", "percent_change", "previous_close"]
                        compact = {k: q[k] for k in keys if k in q}
                        st.write(compact)

                    st.subheader("🧩 İşlem Yönetimi (Eldeki Hisse)")
                    st.caption("Stop asla gevşetilmez.")

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
                            suggestions.append(f"Entry'ye göre %+{move_pct:.1f}. Stop'u en az **break-even** seviyesine çek: {sug_stop:.2f}")

                    if np.isfinite(tp1_0) and cur_price >= tp1_0:
                        ema20_now = float(df.iloc[-1]["ema20"])
                        ema50_now = float(df.iloc[-1]["ema50"])
                        trail = max(stop0, min(cur_price * 0.995, max(ema20_now, ema50_now) * 0.995))
                        suggestions.append(f"TP1 bölgesi: stop'u **EMA bazlı** yukarı taşı: {trail:.2f}")

                    if np.isfinite(tp2_0) and cur_price >= tp2_0:
                        suggestions.append("TP2 bölgesi: Momentum bozulursa kısmi/çıkış; korunuyorsa trailing stop.")

                    if suggestions:
                        st.info("**Yönetim Önerisi:**\n\n- " + "\n- ".join(suggestions))
                    else:
                        st.caption("Yönetim önerileri için fiyatın entry/TP seviyelerine yaklaşmasını bekle.")

                    st.subheader("📄 Rapor")
                    pdf_bytes = build_pdf_bytes_single(ticker=ticker, interval_label=interval_label, bars=bars, plan=plan, quote=(q if show_quote else None))
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
# SEKME 2: PORTFÖY
# =========================================================
with tab_portfolio:
    st.subheader("🧳 Portföy Analiz")
    st.caption("Portföy satırlarını gir: ticker, adet, alış ort., stop, TP1, TP2.")

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
                "qty": st.column_config.NumberColumn("Adet", min_value=0.0, step=1.0),
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
            key="pf_interval",
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

                with st.spinner("Portföy verileri çekiliyor..."):
                    spy_df_shared = pd.DataFrame()
                    try:
                        spy_df_shared = _fetch_spy_daily(320)
                    except Exception:
                        pass

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

                            low_52w, high_52w, daily_df_for_52w = get_daily_52w_levels(tkr, interval, dfi)
                            plan = build_trade_plan(dfi, low_52w=low_52w, high_52w=high_52w)

                            lead = leadership_pack(
                                tkr, interval, dfi,
                                low_52w=low_52w, high_52w=high_52w,
                                spy_df=spy_df_shared,
                                daily_df_override=daily_df_for_52w,
                            )

                            candle_close = float(dfi.iloc[-1]["close"])
                            price = candle_close
                            if show_quote:
                                try:
                                    q = td_quote(tkr)
                                    if "price" in q:
                                        price = float(q["price"])
                                except Exception:
                                    pass

                            # DÜZELTME: tuple sırası low, high
                            low_52w_roll, high_52w_roll = rolling_52w_levels(daily_df_for_52w, bars_1day=260)
                            blue = is_blue_sky(price, high_52w_roll)

                            ema20_now = float(dfi.iloc[-1]["ema20"])
                            ema50_now = float(dfi.iloc[-1]["ema50"])
                            trail_head, _trail_detail = trailing_structure_status(price, ema20_now, ema50_now)

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

                            # DÜZELTME: Risk $ = avg_cost - stop bazlı
                            risk_per_share = (avg_cost - user_stop) if (np.isfinite(user_stop) and np.isfinite(avg_cost)) else np.nan
                            risk_value = (risk_per_share * qty) if (np.isfinite(risk_per_share) and np.isfinite(qty)) else np.nan

                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": round(price, 2),
                                "Qty": round(qty, 2) if np.isfinite(qty) else "",
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
                                "RSI Yönü": plan.rsi_direction_label,
                                "Yüksek Vol Uyarı": "⚠️" if plan.high_vol_warning else "",
                                "52W Zirve Uzaklık %": round(float(plan.dist_to_52w_high_pct), 1) if np.isfinite(plan.dist_to_52w_high_pct) else "",
                                "RS Rating": round(float(lead.get("rs_rating", np.nan)), 0) if np.isfinite(lead.get("rs_rating", np.nan)) else "",
                                "RS Yeni Zirve": "✅" if lead.get("rs_new_high_60d") else "",
                                "Endekse Üstünlük 3A": round(float(lead.get("edge_3m", np.nan)), 1) if np.isfinite(lead.get("edge_3m", np.nan)) else "",
                                "Hacim Kuruması": "✅" if lead.get("dryup_ok") else "",
                                "Kuruma Oranı": round(float(lead.get("dryup_ratio", np.nan)), 2) if np.isfinite(lead.get("dryup_ratio", np.nan)) else "",
                                "Liderlik": str(lead.get("leader_label", "—")),
                                "Auto Stop": round(plan.stop, 2),
                                "Auto TP1": round(plan.tp1, 2),
                                "Auto TP2": round(plan.tp2, 2),
                                "Poz. Değeri": round(pos_value, 2) if np.isfinite(pos_value) else "",
                                "Risk $": round(risk_value, 2) if np.isfinite(risk_value) else "",
                                "Aksiyon": action,
                                "Not": comment,
                                "52W High": round(high_52w_roll, 2) if np.isfinite(high_52w_roll) else "",
                                "Blue Sky": "🔵" if show_blue_box else "",
                                "İz Süren Yapı": trail_head if show_blue_box else "",
                                "Auto Yapısal Stop": round(plan.debug.get("stop_structural", np.nan), 2) if np.isfinite(plan.debug.get("stop_structural", np.nan)) else "",
                                "Auto Noise Stop": round(plan.debug.get("stop_noise", np.nan), 2) if np.isfinite(plan.debug.get("stop_noise", np.nan)) else "",
                            })

                        except Exception as e:
                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": "", "Qty": round(qty, 2) if np.isfinite(qty) else "",
                                "Alış Ort.": round(avg_cost, 2) if np.isfinite(avg_cost) else "",
                                "P&L %": "", "Stop": "", "Stop Mesafe %": "",
                                "TP1": "", "TP1 Mesafe %": "", "TP2": "", "TP2 Mesafe %": "",
                                "R (TP1/Stop)": "", "R (TP2/Stop)": "",
                                "Setup": "", "Timing": "", "Durum": "HATA",
                                "Minervini #5": "", "RSI Yönü": "", "Yüksek Vol Uyarı": "",
                                "52W Zirve Uzaklık %": "",
                                "Auto Stop": "", "Auto TP1": "", "Auto TP2": "",
                                "Poz. Değeri": "", "Risk $": "",
                                "Aksiyon": "HATA", "Not": f"Veri/analiz hatası: {e}",
                                "52W High": "", "Blue Sky": "", "İz Süren Yapı": "",
                            })

                out = pd.DataFrame(rows)

                st.markdown("### Sonuç Tablosu")
                st.dataframe(out, use_container_width=True, hide_index=True)

                kpis = compute_portfolio_kpis(out)

                st.markdown("### 📌 Portföy Özeti")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Portföy Değeri", fmt_money(kpis.get("portfolio_value", np.nan)))
                c2.metric("Anlık P&L ($)", fmt_money(kpis.get("pnl_value", np.nan)))
                c3.metric("Anlık P&L (%)", fmt_pct(kpis.get("pnl_pct", np.nan)))
                c4.metric("TP1 Hepsi Olursa", fmt_money(kpis.get("max_profit_tp1", np.nan)))
                c5.metric("Stop Hepsi Olursa", fmt_money(kpis.get("max_loss_stop", np.nan)))

                st.markdown("### ⬇️ İndir")
                title = "MinerWin – Portföy Analizi V6.1"
                pdf_bytes = build_portfolio_pdf_bytes(title=title, out=out, kpis=kpis, interval_label=interval_label_pf, bars=bars)
                xls_bytes = build_portfolio_excel_bytes(title=title, out=out, kpis=kpis, interval_label=interval_label_pf, bars=bars)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button("📄 Portföy Raporu (PDF) indir", data=pdf_bytes, file_name=f"MinerWin_Portfoy_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)
                with d2:
                    st.download_button("📊 Portföy Raporu (Excel) indir", data=xls_bytes, file_name=f"MinerWin_Portfoy_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

                st.markdown("### 🔵 Blue Sky Evresi")
                if not out.empty and "Blue Sky" in out.columns:
                    blue_rows = out[out["Blue Sky"].astype(str).str.contains("🔵", na=False)].copy()
                    if blue_rows.empty:
                        st.info("Şu an Blue Sky koşulunda pozisyon yok.")
                    else:
                        for _, rr in blue_rows.iterrows():
                            st.markdown(f"**{rr.get('Ticker','')}**")
                            st.write("• Fiyat, 52 haftalık zirve bölgesinde işlem görüyor.")
                            if str(rr.get("İz Süren Yapı", "")).strip():
                                st.write(f"📐 İz Süren Yapı: {rr['İz Süren Yapı']}")
                            st.divider()

                st.markdown("### Hızlı Özet")
                if not out.empty and "Durum" in out.columns:
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

                if not out.empty and "Yüksek Vol Uyarı" in out.columns:
                    vol_warn_tickers = out[out["Yüksek Vol Uyarı"] == "⚠️"]["Ticker"].tolist()
                    if vol_warn_tickers:
                        st.warning(f"⚠️ Yüksek volatilite uyarısı: **{', '.join(vol_warn_tickers)}** — stop cap devrede, pozisyon boylarını kontrol et.")
