# app.py
# MinerWin — Tek Hisse + Portföy Analiz (V5.3 / FAZ-1)
# Twelve Data + Streamlit
#
# ✅ KİLİTLİ MEKANİK (BOZULMAZ):
# - Entry (EMA20–EMA50 bandı), Timing/Setup ayrımı
# - Stop: Invalidation + Noise (ATR) + max risk cap
# - TP1/TP2: Minervini uyumlu hedefleme
# - Eldeki hisse dili / yönetim formu
#
# ✅ FAZ-1 (Bilgi + kalite katmanı):
# - Hacim Onayı: sıkışmada “kuruma” + breakout’ta “hacim artışı” (bilgi/puan)
# - Relative Strength (SPY kıyası): hisse/SPY RS hattı + yeni zirve kontrolü (bilgi/puan)
# - 52W High’a yakınlık etiketi: zirveye uzaksa “ZAYIF YAPI” etiketi (bilgi/puan)
# - Skorlama: RS + Hacim + Zirve yakınlığı puanı eklenir (hard veto yok; mekaniği bozmaz)
#
# Notlar:
# - Twelve Data Free/BASIC plan çoğu zaman pre/post-market quote vermez.
# - “Alım kararı” için ana timeframe: GÜNLÜK. Haftalık sadece bağlam filtresi.

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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# APP CONFIG (Name + Icon)
# =========================================================
# Not: Streamlit “uygulama adı/ikon” (PWA/home-screen) çoğu zaman ayrıca
# .streamlit/config.toml ve (varsa) static/manifest ayarları ister.
# Ama app.py içinde de en azından tarayıcı başlığı + favicon'u düzeltiyoruz.
#
# Eğer dosyan varsa: minerwin_icon.png
PAGE_ICON = "minerwin_icon.png" if os.path.isfile("minerwin_icon.png") else "📈"
st.set_page_config(page_title="MinerWin – Portföy Analizi", page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")


# =========================================================
# MINERWIN UI (Branding + Professional Header)
# =========================================================
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

.small-muted { color:#8b949e; font-size:12px; }
.badge {
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid #2a2f39;
  background:#0f141b;
  font-size:12px;
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
<div class="sub-title">Minervini-Based Technical Trading Engine • V5.3 (FAZ-1)</div>
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
def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct(a: float, b: float) -> float:
    """% change from b to a"""
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return np.nan
    return (a - b) / b * 100


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return lo


def _fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}"


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

    # volume (FAZ-1 için kritik)
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
# 52W HELPERS (Minervini)
# =========================================================
@st.cache_data(ttl=120)
def fetch_daily_52w(symbol: str) -> Tuple[float, float]:
    """52W dip/tepe için günlük veriden ~260 bar."""
    payload = td_time_series(symbol, "1day", 260)
    df = parse_ohlcv(payload)
    low_52w = float(df["low"].min())
    high_52w = float(df["high"].max())
    return low_52w, high_52w


def minervini_rule5_ok(price: float, low_52w: float) -> bool:
    """Minervini #5: Güncel fiyat, 52W low’un en az %25 üstünde olmalı."""
    if not (np.isfinite(price) and np.isfinite(low_52w) and low_52w > 0):
        return False
    return price >= 1.25 * low_52w


def near_52w_high_ok(price: float, high_52w: float, max_below_pct: float = 25.0) -> bool:
    """FAZ-1: Fiyat 52W high’ın en fazla %25 altında olsun (lider odak)."""
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return False
    return price >= (1.0 - max_below_pct / 100.0) * high_52w


def high_proximity_band(price: float, high_52w: float) -> str:
    """Etiket: Zirveye yakınlık bandı."""
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return "—"
    dist = (high_52w - price) / high_52w * 100.0
    if dist <= 2.0:
        return "BLUE-SKY ZİRVE"
    if dist <= 10.0:
        return "ZİRVEYE YAKIN"
    if dist <= 25.0:
        return "ORTA"
    return "ZİRVEYE UZAK"


# =========================================================
# FAZ-1: VOLUME CONFIRMATION
# =========================================================
def analyze_volume_profile(
    df: pd.DataFrame,
    in_entry_band: bool,
    pivot_lookback: int = 20,
    dry_lookback: int = 10,
    vol_ma: int = 50,
    dry_ratio_max: float = 0.60,  # son 10 gün ort. hacim <= 0.60 * MA50 → “kuruma”
    breakout_ratio_min: float = 1.50,  # breakout barı hacim >= 1.50 * MA50
) -> Dict[str, Any]:
    """
    FAZ-1:
    - Kuruma (VCP benzeri): entry bandındayken son 5-10 gün hacmi, 50g ortalamanın belirgin altında olmalı.
    - Breakout hacmi: pivot high geçilirken hacim, MA50'nin üstünde olmalı.

    Not: Bu modül hard veto yapmaz; sadece puan + bilgi üretir.
    """
    out: Dict[str, Any] = {
        "vol_ma50": np.nan,
        "dry_ratio": np.nan,
        "dry_ok": False,
        "pivot_high": np.nan,
        "breakout": False,
        "breakout_ok": False,
        "breakout_ratio": np.nan,
        "note": "",
    }

    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns or "high" not in df.columns:
        out["note"] = "Hacim verisi eksik."
        return out

    d = df.copy()
    d["vol_ma50"] = d["volume"].rolling(vol_ma, min_periods=max(10, vol_ma // 2)).mean()

    vol_ma50 = float(d.iloc[-1]["vol_ma50"]) if np.isfinite(d.iloc[-1]["vol_ma50"]) else np.nan
    out["vol_ma50"] = vol_ma50

    # kuruma: son dry_lookback ort hacim / vol_ma50
    tail = d.tail(max(dry_lookback, 5))
    vol_avg_short = float(tail["volume"].mean()) if len(tail) > 0 else np.nan
    dry_ratio = (vol_avg_short / vol_ma50) if (np.isfinite(vol_avg_short) and np.isfinite(vol_ma50) and vol_ma50 > 0) else np.nan
    out["dry_ratio"] = float(dry_ratio) if np.isfinite(dry_ratio) else np.nan

    dry_ok = bool(in_entry_band and np.isfinite(dry_ratio) and (dry_ratio <= dry_ratio_max))
    out["dry_ok"] = dry_ok

    # pivot high: son pivot_lookback içindeki en yüksek high (son bar hariç)
    d2 = d.tail(max(pivot_lookback + 2, 25)).reset_index(drop=True)
    if len(d2) >= 5:
        pivot_high = float(d2.iloc[:-1]["high"].max())  # last bar excluded
        out["pivot_high"] = pivot_high

        last_close = float(d2.iloc[-1]["close"])
        last_vol = float(d2.iloc[-1]["volume"])
        breakout = bool(np.isfinite(last_close) and np.isfinite(pivot_high) and last_close > pivot_high)

        out["breakout"] = breakout
        if breakout and np.isfinite(vol_ma50) and vol_ma50 > 0 and np.isfinite(last_vol):
            br_ratio = last_vol / vol_ma50
            out["breakout_ratio"] = float(br_ratio)
            out["breakout_ok"] = bool(br_ratio >= breakout_ratio_min)
        else:
            out["breakout_ratio"] = np.nan
            out["breakout_ok"] = False

    return out


# =========================================================
# FAZ-1: RELATIVE STRENGTH (SPY)
# =========================================================
@st.cache_data(ttl=180)
def fetch_daily_df(symbol: str, bars: int = 300) -> pd.DataFrame:
    payload = td_time_series(symbol, "1day", int(bars))
    return parse_ohlcv(payload)


def compute_rs_metrics(
    df_stock_daily: pd.DataFrame,
    df_spy_daily: pd.DataFrame,
    lookback_high: int = 252,
) -> Dict[str, Any]:
    """
    RS line = stock_close / spy_close (aligned by date)
    - RS new high? (son lookback_high içinde)
    - RS slope (son 20 bar)
    - 3m/6m/12m relatif performans (yaklaşık)
    """
    out: Dict[str, Any] = {
        "rs_new_high": False,
        "rs_slope20": np.nan,
        "rel_3m": np.nan,
        "rel_6m": np.nan,
        "rel_12m": np.nan,
        "note": "",
    }

    if df_stock_daily is None or df_spy_daily is None or df_stock_daily.empty or df_spy_daily.empty:
        out["note"] = "RS için veri eksik."
        return out

    a = df_stock_daily[["time", "close"]].copy().rename(columns={"close": "c_stock"})
    b = df_spy_daily[["time", "close"]].copy().rename(columns={"close": "c_spy"})
    m = pd.merge(a, b, on="time", how="inner")
    m = m.dropna().sort_values("time").reset_index(drop=True)
    if len(m) < 80:
        out["note"] = "RS hesaplamak için bar sayısı az."
        return out

    m["rs"] = m["c_stock"] / m["c_spy"].replace(0, np.nan)

    # RS new high
    n = min(len(m), int(lookback_high))
    rs_win = m.iloc[-n:]["rs"].dropna()
    if len(rs_win) >= 30:
        rs_now = float(rs_win.iloc[-1])
        rs_high = float(rs_win.max())
        out["rs_new_high"] = bool(np.isfinite(rs_now) and np.isfinite(rs_high) and rs_now >= rs_high * 0.995)

    # RS slope 20
    out["rs_slope20"] = slope(m["rs"], lookback=20)

    # Rel perf windows (yaklaşık günlük bar)
    def _rel_perf(days: int) -> float:
        if len(m) < days + 5:
            return np.nan
        p0s = float(m.iloc[-days]["c_stock"])
        p1s = float(m.iloc[-1]["c_stock"])
        p0i = float(m.iloc[-days]["c_spy"])
        p1i = float(m.iloc[-1]["c_spy"])
        if p0s <= 0 or p0i <= 0:
            return np.nan
        r_stock = (p1s / p0s) - 1.0
        r_spy = (p1i / p0i) - 1.0
        return float((r_stock - r_spy) * 100.0)

    out["rel_3m"] = _rel_perf(63)   # ~3 ay
    out["rel_6m"] = _rel_perf(126)  # ~6 ay
    out["rel_12m"] = _rel_perf(252) # ~12 ay

    return out


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

    - stop_structural: invalidation base (pivot/EMA)
    - stop_noise:      ATR-based “room”
    - stop_active:     operational stop after applying max risk cap
    """
    if not (np.isfinite(entry) and np.isfinite(ema50) and np.isfinite(atr14)) or entry <= 0:
        stop_fallback = float(entry * 0.93)
        return stop_fallback, float("nan"), float("nan"), {"reason": "NaN entry/ema50/atr14"}

    # Structural invalidation base
    inv_from_ema = float(ema50 * 0.995)  # EMA50 altına tolerans
    inv_from_pivot = float(pivot_low * 0.995) if (np.isfinite(pivot_low) and pivot_low > 0) else float("nan")

    if np.isfinite(inv_from_pivot):
        stop_structural = float(min(inv_from_ema, inv_from_pivot))
        inv_src = "pivot_or_ema"
    else:
        stop_structural = float(inv_from_ema)
        inv_src = "ema50"

    # Noise stop (ATR room)
    nf = _noise_factor_from_atr_pct(atr_pct)
    stop_noise = float(entry - nf * atr14)

    # Choose wider stop to avoid “shake-out”
    stop_candidate = float(min(stop_structural, stop_noise))

    # Cap maximum risk
    cap_stop = float(entry * (1.0 - max_risk_pct / 100.0))
    if stop_candidate < cap_stop:
        stop_active = cap_stop
        capped = True
    else:
        stop_active = stop_candidate
        capped = False

    # Safety: never >= entry
    if stop_active >= entry:
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
    """Son lookback barda en iyi dip->tepe yüzdesi (kaba ama güvenli üst sınır)."""
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
    rs_leadership: int
    volume_confirmation: int
    high_proximity: int


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

    # FAZ-1 extras
    high_proximity_ok: bool
    high_proximity_band: str

    rs_new_high: bool
    rs_slope20: float
    rel_3m: float
    rel_6m: float
    rel_12m: float

    vol_ma50: float
    dry_ratio: float
    dry_ok: bool
    pivot_high: float
    breakout: bool
    breakout_ok: bool
    breakout_ratio: float

    capacity_level: str
    expected_move_pct: float
    targets_reason: str

    narrative: str
    scenario: str
    debug: dict
    breakdown: ScoreBreakdown


def label_from_total(score: int) -> str:
    if score >= 78:
        return "UYGUN"
    if score >= 62:
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
    high_proximity_ok: bool,
    rs_new_high: bool,
    dry_ok: bool,
) -> str:
    # KİLİTLİ (öncelikler): minervini5 + trend bozulması aynı mantık
    if not minervini5_ok:
        return "🟣 52W DİP FİLTRESİ (ZAYIF)"
    if trend_broken or setup_score < 45:
        return "🔴 TREND BOZULDU"
    if consolidation:
        return "🔵 KONSOLİDASYON"
    if in_entry and timing_score >= 70:
        # FAZ-1 bilgi etiketi: ALIM bölgesi ama kalite bayrakları
        if (not high_proximity_ok) or (not rs_new_high) or (not dry_ok):
            return "🟢 ALIM BÖLGESİNDE (KALİTE KONTROL)"
        return "🟢 ALIM BÖLGESİNDE"
    if is_extended and timing_score < 50:
        return "⚫ UZAMIŞ — KOVALAMA"
    # FAZ-1: zirveye uzaksa “zayıf yapı” etiketi (mekaniğe dokunmadan)
    if not high_proximity_ok and setup_score >= 55:
        return "🟠 ZİRVEYE UZAK (ZAYIF YAPI)"
    return "🟡 PULLBACK BEKLENİYOR"


def build_trade_plan(
    df: pd.DataFrame,
    low_52w: float,
    high_52w: float,
    rs_metrics: Optional[Dict[str, Any]] = None,
    vol_metrics: Optional[Dict[str, Any]] = None,
) -> TradePlan:
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

    # Minervini #5
    m5_ok = minervini_rule5_ok(close, low_52w)

    # FAZ-1: 52W high proximity
    hi_ok = near_52w_high_ok(close, high_52w, max_below_pct=25.0)
    hi_band = high_proximity_band(close, high_52w)

    # Entry zone (EMA20-EMA50)
    entry_low = float(min(ema20, ema50))
    entry_high = float(max(ema20, ema50))
    entry_mid = float((entry_low + entry_high) / 2.0)
    in_entry = (entry_low <= close <= entry_high)

    consolidation = _detect_consolidation(atr_pct, rsi14)

    # FAZ-1 RS / Volume metrics defaults
    rs_metrics = rs_metrics or {}
    rs_new_high = bool(rs_metrics.get("rs_new_high", False))
    rs_slope20 = safe_float(rs_metrics.get("rs_slope20"))
    rel_3m = safe_float(rs_metrics.get("rel_3m"))
    rel_6m = safe_float(rs_metrics.get("rel_6m"))
    rel_12m = safe_float(rs_metrics.get("rel_12m"))

    vol_metrics = vol_metrics or {}
    vol_ma50 = safe_float(vol_metrics.get("vol_ma50"))
    dry_ratio = safe_float(vol_metrics.get("dry_ratio"))
    dry_ok = bool(vol_metrics.get("dry_ok", False))
    pivot_high = safe_float(vol_metrics.get("pivot_high"))
    breakout = bool(vol_metrics.get("breakout", False))
    breakout_ok = bool(vol_metrics.get("breakout_ok", False))
    breakout_ratio = safe_float(vol_metrics.get("breakout_ratio"))

    # --- Legacy total (keep core shape) ---
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

    # Minervini #5 fails -> cap (KİLİTLİ)
    if not m5_ok:
        total = min(total, 55)

    # --- FAZ-1 scoring add-ons (NO HARD VETO) ---
    # RS Leadership: 0..12
    rs_pts = 0
    if rs_new_high:
        rs_pts += 8
    if np.isfinite(rs_slope20) and rs_slope20 > 0:
        rs_pts += 4
    rs_pts = int(clamp(rs_pts, 0, 12))

    # Volume Confirmation: 0..10
    vol_pts = 0
    if dry_ok:
        vol_pts += 6
    if breakout and breakout_ok:
        vol_pts += 4
    vol_pts = int(clamp(vol_pts, 0, 10))

    # High proximity: 0..8
    hi_pts = 0
    if hi_ok:
        hi_pts = 8
    else:
        # orta bandda bir miktar puan
        if hi_band == "ORTA":
            hi_pts = 3
        else:
            hi_pts = 0

    # toplamı genişlet (toplam 100 kalacak şekilde tavanla)
    total = int(clamp(total + rs_pts + vol_pts + hi_pts, 0, 100))

    label = label_from_total(total)

    # Split scores (KİLİTLİ yaklaşım: setup/timing ayrı)
    setup_raw = trend_pts + p_pts + m_pts + v_pts  # max 85 (core)
    setup_score = int(round(100 * setup_raw / 85)) if setup_raw > 0 else 0

    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)       # 0..60
    ext_pts = _extension_points(extended)              # 0..40
    timing_score = int(ext_pts + prox_pts)             # 0..100

    status_tag = _status_tag(
        timing_score=timing_score,
        setup_score=setup_score,
        trend_broken=trend_broken,
        is_extended=extended,
        in_entry=in_entry,
        consolidation=consolidation,
        minervini5_ok=m5_ok,
        high_proximity_ok=hi_ok,
        rs_new_high=rs_new_high,
        dry_ok=dry_ok,
    )

    watch_level = float(entry_high)

    # STOP (Invalidation + Noise)
    pivot_low = _recent_pivot_low(df, lookback=20)
    stop, stop_structural, stop_noise, stop_dbg = compute_stop_invalidation_plus_noise(
        entry=entry_mid,
        ema50=ema50,
        atr14=atr14,
        atr_pct=atr_pct,
        pivot_low=pivot_low,
        max_risk_pct=7.0,  # sistem disiplini
    )

    # TP1/TP2
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

    # Narrative/scenario (KİLİTLİ mantık + FAZ-1 bilgi satırları)
    trend_text = (
        "güçlü" if (trend_stack_ok and (price_above_ema150 or price_near_ema150))
        else ("zayıf" if close < ema200 else "karışık")
    )
    mom_text = "sağlıklı" if 55 <= rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf/sınır")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_border else "yüksek")

    if status_tag.startswith("🟢"):
        timing_cmd = "ALIM ARANIR"
    elif status_tag.startswith(("🟡", "🔵", "🟠")):
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

    # FAZ-1 line items
    rs_line = (
        f"RS (SPY): {'✅ RS yeni zirve' if rs_new_high else '⚠️ RS yeni zirve değil'}"
        + (f" | RS eğim(20)={rs_slope20:.4f}" if np.isfinite(rs_slope20) else "")
    )
    rel_line = (
        f"RelPerf: 3A={rel_3m:+.1f}% | 6A={rel_6m:+.1f}% | 12A={rel_12m:+.1f}%"
        if (np.isfinite(rel_3m) or np.isfinite(rel_6m) or np.isfinite(rel_12m))
        else "RelPerf: —"
    )
    vol_line = (
        f"Hacim: MA50={_fmt_money(vol_ma50)} | KurumaOranı={(dry_ratio*100):.0f}% → {'✅ kuruma' if dry_ok else '⚠️ kuruma yok'}"
        if np.isfinite(vol_ma50) and np.isfinite(dry_ratio)
        else "Hacim: —"
    )
    br_line = (
        f"Breakout: {'✅' if breakout else '—'} | PivotHigh={pivot_high:.2f} | HacimOranı={(breakout_ratio):.2f} → {'✅' if breakout_ok else '⚠️'}"
        if breakout and np.isfinite(pivot_high)
        else "Breakout: —"
    )
    hi_line = f"52W Zirve Yakınlığı: {hi_band} → {'✅' if hi_ok else '⚠️'}"

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
        f"**Minervini #5:** 52W dip={low_52w:.2f} → {'✅ geçiyor' if m5_ok else '❌ geçmiyor'}  \n"
        f"**FAZ-1:** {hi_line}  \n"
        f"**FAZ-1:** {rs_line}  \n"
        f"**FAZ-1:** {rel_line}  \n"
        f"**FAZ-1:** {vol_line}  \n"
        f"**FAZ-1:** {br_line}  \n\n"
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

    breakdown = ScoreBreakdown(
        trend_stack=trend_pts,
        price_vs_ema150=p_pts,
        momentum_rsi=m_pts,
        volatility_atr=v_pts,
        extension_vs_ema50=e_pts,
        rs_leadership=int(rs_pts),
        volume_confirmation=int(vol_pts),
        high_proximity=int(hi_pts),
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
        "high_proximity_ok": hi_ok,
        "high_proximity_band": hi_band,
        "pivot_low": pivot_low,
        "stop_debug": stop_dbg,
        "stop_structural": stop_structural,
        "stop_noise": stop_noise,
        "targets_debug": targets_dbg,
        "rs_metrics": rs_metrics,
        "vol_metrics": vol_metrics,
        "faz1_pts": {"rs_pts": rs_pts, "vol_pts": vol_pts, "hi_pts": hi_pts},
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

        high_proximity_ok=bool(hi_ok),
        high_proximity_band=str(hi_band),

        rs_new_high=bool(rs_new_high),
        rs_slope20=float(rs_slope20) if np.isfinite(rs_slope20) else float("nan"),
        rel_3m=float(rel_3m) if np.isfinite(rel_3m) else float("nan"),
        rel_6m=float(rel_6m) if np.isfinite(rel_6m) else float("nan"),
        rel_12m=float(rel_12m) if np.isfinite(rel_12m) else float("nan"),

        vol_ma50=float(vol_ma50) if np.isfinite(vol_ma50) else float("nan"),
        dry_ratio=float(dry_ratio) if np.isfinite(dry_ratio) else float("nan"),
        dry_ok=bool(dry_ok),
        pivot_high=float(pivot_high) if np.isfinite(pivot_high) else float("nan"),
        breakout=bool(breakout),
        breakout_ok=bool(breakout_ok),
        breakout_ratio=float(breakout_ratio) if np.isfinite(breakout_ratio) else float("nan"),

        capacity_level=str(cap_level),
        expected_move_pct=float(expected_move_pct) if np.isfinite(expected_move_pct) else float("nan"),
        targets_reason=targets_reason,
        narrative=narrative,
        scenario=scenario,
        debug=debug,
        breakdown=breakdown,
    )


# =========================================================
# PDF EXPORT (Single Stock)
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

    draw_line("MinerWin — Tek Hisse Teknik Analiz Raporu (V5.3 / FAZ-1)", font="Helvetica-Bold", size=15, space=18)
    draw_line(f"Ticker: {ticker}    Zaman: {interval_label}    Bar: {bars}", size=11)
    draw_line(f"Tarih: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", font="Helvetica", size=10)
    draw_line("", size=10, space=10)

    draw_line("Özet", font="Helvetica-Bold", size=13, space=16)
    draw_line(f"Güncel Fiyat: {plan.debug.get('close', float('nan')):.2f}", size=11)
    draw_line(f"Toplam Skor: {plan.total_score}/100    Etiket: {plan.label}", size=11)
    draw_line(f"Setup: {plan.setup_score}/100    Timing: {plan.timing_score}/100", size=11)
    draw_line(f"Durum: {plan.status_tag}", size=11)
    draw_line(
        f"Minervini #5: {'OK' if plan.minervini5_ok else 'FAIL'} | 52W dip={plan.low_52w:.2f} | 52W tepe={plan.high_52w:.2f}",
        size=10,
        space=12,
    )
    draw_line(f"FAZ-1: 52W Zirve Yakınlığı: {plan.high_proximity_band} | RS yeni zirve: {'Evet' if plan.rs_new_high else 'Hayır'} | Kuruma: {'Evet' if plan.dry_ok else 'Hayır'}", size=10, space=12)

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
    show_volume: bool,
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

    # Volume bars (FAZ-1)
    if show_volume and "volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df["time"], y=df["volume"], name="Hacim", yaxis="y2", opacity=0.35)
        )

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

    # secondary axis for volume
    if show_volume:
        fig.update_layout(
            yaxis2=dict(
                title="Hacim",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            )
        )

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
# PORTFOLIO HELPERS (Blue Sky)
# =========================================================
def rolling_52w_levels(df: pd.DataFrame, bars_1day: int = 260) -> tuple[float, float]:
    if df is None or df.empty:
        return (np.nan, np.nan)
    n = min(len(df), int(bars_1day))
    window = df.iloc[-n:]
    hi = float(window["high"].max()) if "high" in window.columns else np.nan
    lo = float(window["low"].min()) if "low" in window.columns else np.nan
    return hi, lo


def is_blue_sky(price: float, high_52w: float, threshold: float = 0.98) -> bool:
    if not (np.isfinite(price) and np.isfinite(high_52w) and high_52w > 0):
        return False
    return price >= (threshold * high_52w)


def trailing_structure_status(price: float, ema20: float, ema50: float) -> tuple[str, str]:
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
        return "STOP YUKARI", "TP1 bölgesi → stopu yukarı çekerek trend takip et."

    if np.isfinite(user_stop) and (price - user_stop) / price < 0.03:
        return "DİKKAT", "Stop çok yakın → oynaklık stoplatabilir. (Gevşetme yok; pozisyon boyunu düşün.)"

    return "TUT", "Koşullar fena değil → plana sadık kal; ekleme için giriş bandı ve timing bekle."


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
    show_volume = st.checkbox("Hacim barlarını göster (FAZ-1)", value=False)
    st.caption("Hepsini kapatırsan çizgi otomatik açılır.")

    st.divider()
    show_quote = st.checkbox("Quote (anlık fiyat) kullan", value=False)
    st.caption("Quote açarsan +1 API çağrısı (ticker başına).")

    st.divider()
    st.subheader("📌 Tek Hisse Test Hafızası (Oturum)")
    if st.session_state.daily_tests:
        df_mem = pd.DataFrame(st.session_state.daily_tests)
        show_cols = [
            "timestamp", "ticker", "timeframe", "price",
            "setup_score", "timing_score", "total_score", "status_tag",
            "minervini5_ok", "stop", "tp1", "tp2",
            "rs_new_high", "dry_ok", "hi_band"
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
                df_daily = None
                df_spy = None
                try:
                    low_52w, high_52w = fetch_daily_52w(ticker)
                    df_daily = fetch_daily_df(ticker, bars=320)
                    df_spy = fetch_daily_df("SPY", bars=320)
                except Exception:
                    pass

                # FAZ-1 RS + Volume (daily based)
                rs_metrics = {}
                vol_metrics = {}
                try:
                    if df_daily is not None and df_spy is not None and not df_daily.empty and not df_spy.empty:
                        rs_metrics = compute_rs_metrics(df_daily, df_spy)
                except Exception:
                    rs_metrics = {}

                # volume module on current interval df (because chart/entry band is that df)
                # NOTE: “kuruma” mantığı entry bandındayken anlamlı; bu yüzden in_entry_band = True/False plan içinde
                # burada sadece metrics'i çıkarıyoruz; plan’a veriyoruz.
                # in_entry_band bilgisi plan içinde hesaplandığı için, kuruma modülünü iki aşamalı yapıyoruz:
                # önce geçici in_entry_band hesaplayıp metrik çıkarıyoruz.
                try:
                    tmp_last = df.iloc[-1]
                    tmp_close = float(tmp_last["close"])
                    tmp_ema20 = float(df.iloc[-1]["ema20"])
                    tmp_ema50 = float(df.iloc[-1]["ema50"])
                    tmp_entry_low = float(min(tmp_ema20, tmp_ema50))
                    tmp_entry_high = float(max(tmp_ema20, tmp_ema50))
                    tmp_in_entry = bool(tmp_entry_low <= tmp_close <= tmp_entry_high)
                    vol_metrics = analyze_volume_profile(df, in_entry_band=tmp_in_entry)
                except Exception:
                    vol_metrics = {}

                plan = build_trade_plan(df, low_52w=low_52w, high_52w=high_52w, rs_metrics=rs_metrics, vol_metrics=vol_metrics)

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

                # Memory record
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
                    "rs_new_high": bool(plan.rs_new_high),
                    "dry_ok": bool(plan.dry_ok),
                    "hi_band": str(plan.high_proximity_band),
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
                    st.caption(
                        f"Yapısal: {plan.debug.get('stop_structural', float('nan')):.2f} | "
                        f"Noise: {plan.debug.get('stop_noise', float('nan')):.2f}"
                    )

                st.markdown(
                    f"<span class='badge'>Minervini #5: {'✅' if plan.minervini5_ok else '❌'} </span> "
                    f"<span class='badge'>52W Zirve: {plan.high_proximity_band}</span> "
                    f"<span class='badge'>RS yeni zirve: {'✅' if plan.rs_new_high else '⚠️'}</span> "
                    f"<span class='badge'>Hacim kuruma: {'✅' if plan.dry_ok else '⚠️'}</span>",
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="card">', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("🧠 Skor Dağılımı (Core + FAZ-1)")
                b = plan.breakdown
                bdf = pd.DataFrame(
                    {
                        "Bileşen": [
                            "Trend",
                            "Fiyat/EMA150",
                            "Momentum (RSI)",
                            "Volatilite (ATR%)",
                            "Uzama (EMA50)",
                            "RS Liderliği (FAZ-1)",
                            "Hacim Onayı (FAZ-1)",
                            "Zirve Yakınlığı (FAZ-1)",
                        ],
                        "Puan": [
                            b.trend_stack, b.price_vs_ema150, b.momentum_rsi, b.volatility_atr, b.extension_vs_ema50,
                            b.rs_leadership, b.volume_confirmation, b.high_proximity
                        ],
                        "Not": [
                            "", "", "", "", "",
                            "RS hattı/performans",
                            "Kuruma + breakout hacmi",
                            "52W high’a yakınlık",
                        ],
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

                # ========= İşlem Yönetimi (FORM) =========
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
                        sug_stop = max(stop0, entry0)  # break-even
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

                # ========= PDF =========
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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Grafik")
        if "__df" not in st.session_state:
            st.info("Soldan ticker girip **Getir & Analiz Et** ile başla.")
        else:
            df = st.session_state["__df"]
            ticker = st.session_state["__ticker"]
            plan = st.session_state["__plan"]
            last_price_line = float(st.session_state.get("__last_price_line", float(df.iloc[-1]["close"])))
            fig = plot_chart(df, ticker, plan, last_price_line, show_candles, show_emas, show_line, show_volume)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 2: PORTFOLIO (same as your V5.2 with BlueSky)
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

                            # FAZ-1 RS for portfolio (daily only)
                            rs_metrics = {}
                            try:
                                df_d = fetch_daily_df(tkr, bars=320)
                                df_s = fetch_daily_df("SPY", bars=320)
                                rs_metrics = compute_rs_metrics(df_d, df_s)
                            except Exception:
                                rs_metrics = {}

                            # volume metrics based on current interval
                            vol_metrics = {}
                            try:
                                tmp_close = float(dfi.iloc[-1]["close"])
                                tmp_ema20 = float(dfi.iloc[-1]["ema20"])
                                tmp_ema50 = float(dfi.iloc[-1]["ema50"])
                                tmp_in_entry = bool(min(tmp_ema20, tmp_ema50) <= tmp_close <= max(tmp_ema20, tmp_ema50))
                                vol_metrics = analyze_volume_profile(dfi, in_entry_band=tmp_in_entry)
                            except Exception:
                                vol_metrics = {}

                            plan = build_trade_plan(dfi, low_52w=low_52w, high_52w=high_52w, rs_metrics=rs_metrics, vol_metrics=vol_metrics)

                            candle_close = float(dfi.iloc[-1]["close"])
                            price = candle_close
                            if show_quote:
                                try:
                                    q = td_quote(tkr)
                                    if "price" in q:
                                        price = float(q["price"])
                                except Exception:
                                    pass

                            # Blue Sky (portfolio only)
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
                                "FAZ-1: Zirve": plan.high_proximity_band,
                                "FAZ-1: RS": "✅" if plan.rs_new_high else "⚠️",
                                "FAZ-1: Kuruma": "✅" if plan.dry_ok else "⚠️",
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
                                "Durum": "HATA",
                                "Aksiyon": "HATA",
                                "Not": f"Veri/analiz hatası: {e}",
                            })

                out = pd.DataFrame(rows)

                st.markdown("### Sonuç Tablosu")
                st.dataframe(out, use_container_width=True, hide_index=True)

                st.markdown("### 🔵 Blue Sky Evresi (Bilgilendirme)")
                st.caption("Sadece kârda olan ve 52W zirve bölgesindeki pozisyonlar için görünür.")

                if not out.empty and "Blue Sky" in out.columns:
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

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Portföy Analiz CSV indir",
                    data=csv_bytes,
                    file_name="portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
