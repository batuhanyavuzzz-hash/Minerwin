import io
import os
import csv
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dataclasses import dataclass
from datetime import datetime

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="Tek Hisse + Portföy Analiz (V5.0)", layout="wide")
st.title("Tek Hisse Teknik Analiz — Twelve Data (V5.0 | Portföy + Minervini + Dinamik Yönetim)")

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
if not API_KEY:
    st.error('TWELVEDATA_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine ekle: TWELVEDATA_API_KEY="..."')
    st.stop()

BASE_URL = "https://api.twelvedata.com"
HISTORY_FILE = "history.csv"
PORTFOLIO_FILE = "portfolio.csv"

INTERVAL_MAP = {
    "Günlük (1day)": "1day",
    "Saatlik (1h)": "1h",
    "15 Dakika (15min)": "15min",
}

# Session memory init
if "daily_tests" not in st.session_state:
    st.session_state.daily_tests = []

if "portfolio" not in st.session_state:
    # default empty portfolio template
    st.session_state.portfolio = pd.DataFrame(
        columns=["ticker", "qty", "avg_cost", "stop", "tp1", "tp2"]
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
        timeout=20,
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
# MINERVINI FILTER (52W LOW +25%)
# =========================================================
@st.cache_data(ttl=6 * 3600)  # 6 hours
def td_52w_low(symbol: str) -> Tuple[float, int]:
    """
    52 haftalık dip için son 252 günlük barın en düşük 'low' değeri.
    TwelveData 1day series ile hesaplar.
    """
    payload = td_time_series(symbol, "1day", 320)  # 252'yi güvenli kapatmak için
    df = parse_ohlcv(payload)
    if df.empty:
        return (float("nan"), 0)

    tail = df.tail(252)
    if tail.empty:
        return (float("nan"), 0)

    low_52w = float(tail["low"].min())
    return (low_52w, int(len(tail)))


def minervini_rule_ok(price: float, low_52w: float) -> bool:
    # Minervini #5: current price at least 25% above 52-week low
    if not (np.isfinite(price) and np.isfinite(low_52w)) or low_52w <= 0:
        return True  # ölçemiyorsak bloklama
    return price >= 1.25 * low_52w


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
# SCORING / PLAN
# =========================================================
@dataclass
class ScoreBreakdown:
    trend_stack: int
    price_vs_ema150: int
    momentum_rsi: int
    volatility_atr: int
    extension_vs_ema50: int
    minervini_52w: int


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
    rr1: float
    rr2: float

    dist_to_entry_pct: float
    watch_level: float

    low_52w: float
    above_52w_low_25pct: bool
    dist_from_52w_low_pct: float

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
    minervini_fail: bool,
) -> str:
    if minervini_fail:
        return "🔴 52W DİBE YAKIN"
    if trend_broken or setup_score < 45:
        return "🔴 TREND BOZULDU"
    if consolidation:
        return "🔵 KONSOLİDASYON"
    if in_entry and timing_score >= 70:
        return "🟢 ALIM BÖLGESİNDE"
    if is_extended and timing_score < 50:
        return "⚫ UZAMIŞ — KOVALAMA"
    return "🟡 PULLBACK BEKLENİYOR"


def _rr(entry: float, stop: float, tp: float) -> float:
    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(tp)):
        return float("nan")
    risk = entry - stop
    if risk <= 0:
        return float("nan")
    return (tp - entry) / risk


def build_trade_plan(df: pd.DataFrame, low_52w: float = float("nan")) -> TradePlan:
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

    # Minervini #5
    above_25 = minervini_rule_ok(close, low_52w)
    dist_from_52w_low_pct = ((close - low_52w) / low_52w * 100) if (np.isfinite(low_52w) and low_52w > 0) else float("nan")
    minervini_fail = (np.isfinite(low_52w) and low_52w > 0 and not above_25)

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

    # Legacy total (0-100)
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

    # Minervini points (soft reward)
    min_pts = 10 if above_25 else 0
    total = min(100, total + min_pts)

    label = label_from_total(int(total))

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

    # Targets:
    # TP1: 2R (kontrol/ilk realize)
    risk = entry_mid - stop
    tp1 = float(entry_mid + 2.0 * risk) if risk > 0 else float(entry_mid * 1.02)

    # TP2: 3.5R (daha gerçekçi swing devamı) — yine “potansiyel” hedef
    tp2 = float(entry_mid + 3.5 * risk) if risk > 0 else float(entry_mid * 1.05)

    rr1 = _rr(entry_mid, stop, tp1)
    rr2 = _rr(entry_mid, stop, tp2)

    # Split scores
    setup_raw = trend_pts + p_pts + m_pts + v_pts + (10 if above_25 else 0)  # max 95
    setup_score = int(round(100 * setup_raw / 95)) if setup_raw > 0 else 0

    dist_entry_pct = _dist_to_entry_pct(close, entry_low, entry_high)
    prox_pts = _proximity_points(dist_entry_pct)       # 0..60
    ext_pts = _extension_points(extended)              # 0..40
    timing_score = int(ext_pts + prox_pts)             # 0..100

    # Hard filter: Minervini fail -> degrade
    if minervini_fail:
        total = min(int(total), 55)
        setup_score = min(int(setup_score), 40)
        label = "UYGUN DEĞİL"

    in_entry = (entry_low <= close <= entry_high)
    consolidation = _detect_consolidation(atr_pct, rsi14)

    status_tag = _status_tag(
        timing_score=int(timing_score),
        setup_score=int(setup_score),
        trend_broken=bool(trend_broken),
        is_extended=bool(extended),
        in_entry=bool(in_entry),
        consolidation=bool(consolidation),
        minervini_fail=bool(minervini_fail),
    )

    watch_level = float(entry_high)

    # Narrative + scenario
    trend_text = (
        "güçlü" if (trend_stack_ok and (price_above_ema150 or price_near_ema150))
        else ("zayıf" if close < ema200 else "karışık")
    )
    mom_text = "sağlıklı" if 55 <= rsi14 <= 75 else ("ısınmış" if rsi14 > 75 else "zayıf/sınır")
    vol_text = "uygun" if vol_ok else ("agresif" if vol_border else "yüksek")

    if status_tag.startswith("🟢"):
        timing_cmd = "ALIM ARANIR"
    elif status_tag.startswith("🟡") or status_tag.startswith("🔵"):
        timing_cmd = "BEKLE / İZLE"
    else:
        timing_cmd = "UZAK DUR / ŞARTLAR OLUŞSUN"

    if status_tag.startswith("🔴 52W"):
        scenario = (
            "Senaryo: Minervini filtresi (52W dip +%25) geçilemedi. Dipten yeni çıkan zayıf hisseler elenir. "
            "Önce 52W dibe göre net güçlenme (dipten +%25 üstü) ve trend metriklerinin toparlanması beklenmeli."
        )
    elif status_tag.startswith("🟢"):
        scenario = (
            "Senaryo: Fiyat giriş bandında (EMA20–EMA50). Bu bölgede satış baskısı zayıflayıp sıkışma görülürse "
            "trend yönünde devam denemesi yapılabilir. Stop altına sarkarsa disiplin gereği çıkış."
        )
    elif status_tag.startswith("🟡"):
        scenario = (
            "Senaryo: Fiyat giriş bandının dışında. EMA20–EMA50 bandına geri çekilme + hacimde düşüş ile "
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
    else:
        scenario = (
            "Senaryo: Trend filtresi bozulmuş. Önce yeniden EMA150/EMA200 üstüne dönüş ve ortalamaların toparlanması gerekir; "
            "aksi halde swing setup yok."
        )

    rule_text = "GEÇTİ ✅" if above_25 else "KALDI ❌"
    low52_txt = f"{low_52w:.2f}" if np.isfinite(low_52w) else "—"
    dist52_txt = f"%{dist_from_52w_low_pct:.2f}" if np.isfinite(dist_from_52w_low_pct) else "—"

    narrative = (
        f"**Toplam Skor:** {int(total)}/100 → **{label}**  \n"
        f"**Setup Kalitesi:** {int(setup_score)}/100  |  **Zamanlama Skoru:** {int(timing_score)}/100  \n"
        f"**Durum:** {status_tag}  \n\n"
        f"**Güncel (Candle) Fiyat:** {close:.2f}  \n"
        f"EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | EMA150: {ema150:.2f} | EMA200: {ema200:.2f}  \n\n"
        f"**Minervini #5 (52W Dip +%25):** {rule_text}  \n"
        f"52W Dip: {low52_txt} | Dipten Uzaklık: {dist52_txt}  \n\n"
        f"**Trend:** {trend_text} (EMA200 eğim={ema200_slope:.4f})  \n"
        f"**Fiyat Konumu:** EMA150 uzaklık %{dist_ema150_pct:.2f}  \n"
        f"**Momentum (RSI14):** {rsi14:.1f} → {mom_text}  \n"
        f"**Volatilite (ATR%):** %{atr_pct:.2f} → {vol_text}  \n"
        f"**Uzama (EMA50 mesafe):** %{dist_ema50_pct:.2f} → {'uzamış' if extended else 'normal'}  \n\n"
        f"**Zamanlama:** **{timing_cmd}**  \n"
        f"**Giriş Bölgesi:** {entry_low:.2f} – {entry_high:.2f}  \n"
        f"**Giriş Bölgesine Mesafe:** {dist_entry_pct:+.2f}%  \n"
        f"**Takip Seviyesi:** {watch_level:.2f}  \n"
        f"**Stop:** {stop:.2f}  \n"
        f"**TP1:** {tp1:.2f} (2R kontrol)  \n"
        f"**TP2:** {tp2:.2f} (trend devam potansiyeli)  \n"
        f"**R/R (TP1):** 1 : {rr1:.2f}   |   **R/R (TP2):** 1 : {rr2:.2f}"
        if np.isfinite(rr1) and np.isfinite(rr2) else
        "**R/R:** —"
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
        "rr1": rr1,
        "rr2": rr2,
        "setup_score": int(setup_score),
        "timing_score": int(timing_score),
        "dist_to_entry_pct": dist_entry_pct,
        "status_tag": status_tag,
        "consolidation": consolidation,
        "low_52w": low_52w,
        "above_52w_low_25pct": above_25,
        "dist_from_52w_low_pct": dist_from_52w_low_pct,
        "minervini_fail": minervini_fail,
    }

    breakdown = ScoreBreakdown(
        trend_stack=trend_pts,
        price_vs_ema150=p_pts,
        momentum_rsi=m_pts,
        volatility_atr=v_pts,
        extension_vs_ema50=e_pts,
        minervini_52w=min_pts,
    )

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
        rr1=float(rr1) if np.isfinite(rr1) else float("nan"),
        rr2=float(rr2) if np.isfinite(rr2) else float("nan"),
        dist_to_entry_pct=float(dist_entry_pct),
        watch_level=float(watch_level),
        low_52w=float(low_52w) if np.isfinite(low_52w) else float("nan"),
        above_52w_low_25pct=bool(above_25),
        dist_from_52w_low_pct=float(dist_from_52w_low_pct) if np.isfinite(dist_from_52w_low_pct) else float("nan"),
        narrative=narrative,
        scenario=scenario,
        debug=debug,
        breakdown=breakdown,
    )


# =========================================================
# DYNAMIC TRADE MANAGEMENT (Stop + TP1/TP2 Update)
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def technical_strength_flags(plan: TradePlan, df: pd.DataFrame) -> Dict[str, bool]:
    """
    Teknik bazlı kararlar için basit bayraklar.
    """
    last = df.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi14 = float(last["rsi14"])

    dist_ema50_pct = ((close - ema50) / ema50) * 100 if ema50 else float("nan")
    extended = bool(np.isfinite(dist_ema50_pct) and dist_ema50_pct > 10.0)  # daha sıkı
    hot = bool(np.isfinite(rsi14) and rsi14 > 75.0)
    strong = bool(plan.setup_score >= 65 and plan.timing_score >= 55 and not plan.status_tag.startswith("🔴"))

    above_ema20 = bool(np.isfinite(close) and np.isfinite(ema20) and close >= ema20)
    above_ema50 = bool(np.isfinite(close) and np.isfinite(ema50) and close >= ema50)

    return {
        "strong": strong,
        "hot": hot,
        "extended": extended,
        "above_ema20": above_ema20,
        "above_ema50": above_ema50,
    }


def propose_trailing_stop(
    df: pd.DataFrame,
    current_stop: float,
    entry_price: float,
    profit_pct: float,
    flags: Dict[str, bool],
) -> float:
    """
    Stop'u teknik temelli yukarı çekme önerisi.
    - Profit >= ~%5 ise: stop'u "entry altı -> risk azalt" veya EMA20 altına yaklaştır.
    - Trend güçlü ise EMA20 tabanlı daha agresif; değilse EMA50 tabanlı daha yumuşak.
    """
    last = df.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    atr14 = float(last["atr14"])

    if not np.isfinite(current_stop):
        current_stop = float("nan")

    # Korumacı taban stoplar
    stop_floor = current_stop if np.isfinite(current_stop) else -np.inf

    # Profit eşiği: %5 üstü -> stop sıkılaştır
    if not np.isfinite(profit_pct):
        return float(stop_floor if np.isfinite(stop_floor) else current_stop)

    if profit_pct < 5.0:
        # erken dönem: sadece teknik bozulmadıkça dokunma (giriş planı geçerli)
        return float(stop_floor)

    # Risk-free / risk azalt mod:
    # 1) entry'nin hafif altı (spread + stop hunt payı)
    rf_stop = entry_price * 0.995 if np.isfinite(entry_price) else float("nan")

    # 2) EMA20 tabanlı trailing (EMA20'nin biraz altı) — güçlü trendte
    ema20_stop = ema20 * 0.995 if np.isfinite(ema20) else float("nan")

    # 3) EMA50 tabanlı trailing — daha gevşek
    ema50_stop = ema50 * 0.995 if np.isfinite(ema50) else float("nan")

    # 4) ATR tabanlı (kırılmayı tolere etmek için)
    atr_stop = close - 1.5 * atr14 if (np.isfinite(close) and np.isfinite(atr14)) else float("nan")

    candidates = []

    # Güçlü ise: EMA20 ve ATR stop'u daha öncelikli
    if flags.get("strong", False) and flags.get("above_ema20", False):
        for v in [rf_stop, ema20_stop, atr_stop]:
            if np.isfinite(v):
                candidates.append(v)
    else:
        for v in [rf_stop, ema50_stop, atr_stop]:
            if np.isfinite(v):
                candidates.append(v)

    # Eğer aşırı ısınmış/uzamışsa: stop'u biraz daha yukarı çek (kârı kilitle)
    if flags.get("hot", False) or flags.get("extended", False):
        # close'un %3-4 altı
        hot_stop = close * 0.965 if np.isfinite(close) else float("nan")
        if np.isfinite(hot_stop):
            candidates.append(hot_stop)

    if not candidates:
        return float(stop_floor)

    proposed = float(max(stop_floor, max(candidates)))
    # Stop mantıksız şekilde fiyatın üstüne çıkmasın (çok nadir)
    if np.isfinite(close) and proposed >= close:
        proposed = float(close * 0.99)

    return float(proposed)


def propose_targets_update(
    current_tp1: float,
    current_tp2: float,
    price: float,
    entry_price: float,
    profit_pct: float,
    flags: Dict[str, bool],
) -> Tuple[float, float, str]:
    """
    TP1/TP2 dinamik güncelleme önerisi:
    - TP1/TP2 girişte "potansiyel kâr noktaları"dır, ama trend güçlü ise yukarı taşınabilir.
    - Profit >= %5 ve teknik güçlü: TP1'i TP1-TP2 bandında yukarı iter; TP2'yi de bir miktar yukarı iter.
    - Aşırı ısınmış/uzamışsa: TP yukarı itmek yerine "kâr al / stop kilitle" yaklaşımı (TP'leri sabit bırakır).
    """
    note = []

    tp1 = current_tp1
    tp2 = current_tp2

    if not (np.isfinite(tp1) and np.isfinite(tp2) and tp2 > tp1):
        return (tp1, tp2, "")

    if not np.isfinite(profit_pct) or profit_pct < 5.0:
        return (tp1, tp2, "")

    # Eğer uzamış/ısınmış: hedefleri kovalamak yerine kârı koru (TP sabit)
    if flags.get("hot", False) or flags.get("extended", False):
        note.append("Isınma/uzama var → TP'leri agresif yukarı taşımak yerine stop'u kârı kilitleyecek şekilde sıkılaştır.")
        return (tp1, tp2, " ".join(note))

    # Trend güçlü ise: TP1'i TP1-TP2 arasına çek (kontrol noktası ileri taşınır)
    if flags.get("strong", False):
        # profit %5 -> bandın %30'u, %10 -> %50, %15 -> %70
        # 5..15 arası lineer
        w = _clamp((profit_pct - 5.0) / 10.0, 0.0, 1.0)  # 0..1
        alpha = 0.30 + 0.40 * w  # 0.30..0.70
        new_tp1 = tp1 + alpha * (tp2 - tp1)

        # TP2'yi de bir miktar yukarı it: mevcut tp2 ile entry+%20 arasında min olarak, ama aşırıya kaçmadan
        if np.isfinite(entry_price) and entry_price > 0:
            tp2_cap = entry_price * 1.35  # güvenli tavan (çok agresif olmasın)
            # tp2 push: profit artınca %3..%8 ek
            push = 0.03 + 0.05 * w
            new_tp2 = tp2 * (1.0 + push)
            new_tp2 = min(new_tp2, tp2_cap)
        else:
            new_tp2 = tp2

        # Güncelleme ancak mantıklıysa
        if new_tp1 > tp1 and new_tp1 < new_tp2:
            note.append(f"Trend güçlü → TP1 ileri taşındı (TP1-TP2 bandında).")
            tp1 = float(new_tp1)

        if new_tp2 > tp2 and new_tp2 > tp1:
            note.append("Trend güçlü → TP2 de yukarı revize edildi.")
            tp2 = float(new_tp2)

    return (tp1, tp2, " ".join(note))


def manage_open_trade(
    df: pd.DataFrame,
    plan: TradePlan,
    price: float,
    entry_price: float,
    user_stop: float,
    user_tp1: float,
    user_tp2: float,
) -> Dict[str, Any]:
    """
    Açık pozisyon yönetimi önerisi:
    - Stop yukarı çekme
    - TP1/TP2 revizyonu (trend güçlü ise)
    - Aksiyon: TUT / KISMİ KAR / KAR AL / AZALT / ÇIKIŞ
    """
    out: Dict[str, Any] = {}
    if not np.isfinite(price):
        return {"action": "DİKKAT", "note": "Fiyat alınamadı."}

    # Trend bozulması / stop altı
    if np.isfinite(user_stop) and price < user_stop:
        return {"action": "STOPTA ÇIK", "note": "Fiyat stop altına indi → disiplin gereği çıkış."}

    flags = technical_strength_flags(plan, df)

    # Profit %
    profit_pct = pct(price, entry_price) if (np.isfinite(entry_price) and entry_price > 0) else float("nan")

    # Stop önerisi
    proposed_stop = propose_trailing_stop(
        df=df,
        current_stop=user_stop,
        entry_price=entry_price,
        profit_pct=profit_pct,
        flags=flags,
    )

    # TP önerisi
    proposed_tp1, proposed_tp2, tp_note = propose_targets_update(
        current_tp1=user_tp1,
        current_tp2=user_tp2,
        price=price,
        entry_price=entry_price,
        profit_pct=profit_pct,
        flags=flags,
    )

    # Aksiyon mantığı
    # 1) Trend bozuksa azalt/çık
    if plan.status_tag.startswith("🔴"):
        action = "AZALT / ÇIKIŞA HAZIR"
        note = "Trend filtresi zayıf → ekleme yok; risk azaltmayı düşün."
    else:
        action = "TUT"
        note = "Trend korunuyor → pozisyon taşınabilir."

    # 2) TP1/TP2 yakınları: teknik sinyale göre kar al / kısmi kar
    # TP1: ilk kontrol/realize
    if np.isfinite(user_tp1) and price >= user_tp1:
        if flags.get("hot", False) or flags.get("extended", False) or plan.timing_score < 45:
            action = "KISMİ KAR AL"
            note = "TP1 görüldü + ısınma/uzama/zayıflama işareti → kısmi kâr mantıklı."
        else:
            action = "TUT / TRAIL"
            note = "TP1 görüldü ama teknik güçlü → stop'u yukarı çekerek taşınabilir."

    # TP2: daha kuvvetli realize alanı
    if np.isfinite(user_tp2) and price >= user_tp2:
        if flags.get("strong", False) and not (flags.get("hot", False) or flags.get("extended", False)):
            action = "TUT (RUNNER)"
            note = "TP2 görüldü, trend hâlâ güçlü → runner bırakıp trailing ile taşıyabilirsin."
        else:
            action = "KAR AL"
            note = "TP2 görüldü + risk/ısınma → realize etmek mantıklı."

    # 3) Profit >=5% ve stop önerisi yükseldiyse: risk-free mode notu
    rf_note = ""
    if np.isfinite(profit_pct) and profit_pct >= 5.0 and np.isfinite(proposed_stop) and np.isfinite(user_stop) and proposed_stop > user_stop:
        rf_note = "Kâr %5+ → stop yukarı çekilerek risk azaltma (risk-free mod) uygulanabilir."

    out.update({
        "profit_pct": float(profit_pct) if np.isfinite(profit_pct) else np.nan,
        "flags": flags,
        "proposed_stop": float(proposed_stop) if np.isfinite(proposed_stop) else np.nan,
        "proposed_tp1": float(proposed_tp1) if np.isfinite(proposed_tp1) else np.nan,
        "proposed_tp2": float(proposed_tp2) if np.isfinite(proposed_tp2) else np.nan,
        "action": action,
        "note": " ".join([note, rf_note, tp_note]).strip(),
    })
    return out


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
    quote: Optional[dict],
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
    draw_line(f"Toplam Skor: {plan.total_score}/100    Etiket: {plan.label}", size=11)
    draw_line(f"Setup Kalitesi: {plan.setup_score}/100    Zamanlama: {plan.timing_score}/100", size=11)
    draw_line(f"Durum: {plan.status_tag}", size=11)

    low52_txt = f"{plan.low_52w:.2f}" if np.isfinite(plan.low_52w) else "—"
    draw_line(f"Minervini #5 (52W Dip +%25): {'GEÇTİ ✅' if plan.above_52w_low_25pct else 'KALDI ❌'}", size=11)
    if np.isfinite(plan.dist_from_52w_low_pct):
        draw_line(f"52W Dip: {low52_txt}  | Dipten Uzaklık: {plan.dist_from_52w_low_pct:.2f}%", size=11)
    else:
        draw_line(f"52W Dip: {low52_txt}", size=11)

    draw_line(f"Giriş: {plan.entry_low:.2f} – {plan.entry_high:.2f} (mid={plan.entry_mid:.2f})", size=11)
    draw_line(f"Stop: {plan.stop:.2f}    TP1: {plan.tp1:.2f}    TP2: {plan.tp2:.2f}", size=11)
    rr1 = f"1:{plan.rr1:.2f}" if np.isfinite(plan.rr1) else "—"
    rr2 = f"1:{plan.rr2:.2f}" if np.isfinite(plan.rr2) else "—"
    draw_line(f"R/R TP1={rr1}   |   R/R TP2={rr2}", size=11)
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
def plot_chart(df: pd.DataFrame, symbol: str, plan: TradePlan, last_price_line: float,
               user_stop: Optional[float] = None, user_tp1: Optional[float] = None, user_tp2: Optional[float] = None):
    fig = go.Figure()

    # ✅ Fiyat (Line) — legend'da diğerleri kapatılsa bile bu trace kalabilir
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["close"],
            name="Fiyat (Line)",
            mode="lines",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC (Mum)",
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

    # User levels prefered (if provided), else plan levels
    stop_level = float(user_stop) if (user_stop is not None and np.isfinite(user_stop)) else float(plan.stop)
    tp1_level = float(user_tp1) if (user_tp1 is not None and np.isfinite(user_tp1)) else float(plan.tp1)
    tp2_level = float(user_tp2) if (user_tp2 is not None and np.isfinite(user_tp2)) else float(plan.tp2)

    fig.add_hline(y=stop_level, line_dash="dash", annotation_text="STOP", annotation_position="bottom left")
    fig.add_hline(y=tp1_level, line_dash="dash", annotation_text="TP1", annotation_position="top left")
    fig.add_hline(y=tp2_level, line_dash="dash", annotation_text="TP2", annotation_position="top left")
    fig.add_hline(y=float(last_price_line), line_dash="dot", annotation_text="GÜNCEL", annotation_position="top right")

    fig.update_layout(
        title=f"{symbol} — Fiyat(Line) + Candlestick + EMA'lar + Trade Levels",
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =========================================================
# SIDEBAR (Global)
# =========================================================
with st.sidebar:
    st.header("Genel Ayarlar")
    default_interval_label = st.selectbox("Varsayılan zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=0)
    bars = st.slider("Bar sayısı", min_value=120, max_value=800, value=300, step=10)

    # Quote = extra API call; default off to save requests.
    show_quote = st.checkbox("Quote (anlık fiyat) kullan", value=False)
    st.caption("Quote açarsan +1 API çağrısı (ticker başına).")

    st.divider()
    st.subheader("📌 Tek Hisse Test Hafızası (Oturum)")
    if st.session_state.daily_tests:
        df_mem = pd.DataFrame(st.session_state.daily_tests)
        show_cols = [
            "timestamp", "ticker", "timeframe", "price",
            "setup_score", "timing_score", "total_score",
            "status_tag", "dist_to_entry_pct",
            "low_52w", "minervini_52w_ok"
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
    left, right = st.columns([0.38, 0.62], vertical_alignment="top")

    with left:
        st.subheader("Hisse")
        ticker = st.text_input("Ticker", placeholder="Örn: NVDA, TSLA, PLTR").strip().upper()
        interval_label = st.selectbox("Zaman çözünürlüğü", list(INTERVAL_MAP.keys()), index=list(INTERVAL_MAP.keys()).index(default_interval_label))
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

                # 52W low: daily ise df içinden, değilse cached daily çek
                low52 = float("nan")
                if interval == "1day" and len(df) >= 252:
                    low52 = float(df.tail(252)["low"].min())
                else:
                    try:
                        low52, _ = td_52w_low(ticker)
                    except Exception:
                        low52 = float("nan")

                plan = build_trade_plan(df, low_52w=low52)

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

                # Default trade inputs (user-editable)
                st.session_state["__entry_price"] = float(plan.entry_mid)  # varsayılan
                st.session_state["__user_stop"] = float(plan.stop)
                st.session_state["__user_tp1"] = float(plan.tp1)
                st.session_state["__user_tp2"] = float(plan.tp2)

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
                    "entry_low": round(float(plan.entry_low), 4),
                    "entry_high": round(float(plan.entry_high), 4),
                    "entry_mid": round(float(plan.entry_mid), 4),
                    "stop_plan": round(float(plan.stop), 4),
                    "tp1_plan": round(float(plan.tp1), 4),
                    "tp2_plan": round(float(plan.tp2), 4),
                    "dist_to_entry_pct": round(float(plan.dist_to_entry_pct), 4),
                    "watch_level": round(float(plan.watch_level), 4),
                    "low_52w": round(float(plan.low_52w), 4) if np.isfinite(plan.low_52w) else "",
                    "minervini_52w_ok": "YES" if plan.above_52w_low_25pct else "NO",
                }
                st.session_state.daily_tests.append(record)
                try:
                    save_to_history(record)
                except Exception as e:
                    st.warning(f"history.csv yazılamadı: {e}")

                st.divider()
                st.subheader("📊 Strateji Özeti")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Güncel Fiyat", f"{float(last_price_line):.2f}")
                c2.metric("Toplam Skor", f"{plan.total_score} / 100")
                c3.metric("Setup Kalitesi", f"{plan.setup_score} / 100")
                c4.metric("Zamanlama", f"{plan.timing_score} / 100")

                st.metric("Durum", plan.status_tag)

                st.subheader("📌 Plan (Giriş / Stop / TP1 / TP2)")
                table = pd.DataFrame(
                    {
                        "Parametre": [
                            "Minervini #5 (52W Dip +%25)",
                            "52W Dip",
                            "Dipten Uzaklık",
                            "Giriş Bölgesi (EMA20–EMA50)",
                            "Giriş Mid",
                            "Giriş Mesafesi",
                            "Takip Seviyesi",
                            "Stop (plan)",
                            "TP1 (plan)",
                            "TP2 (plan)",
                            "R/R (TP1)",
                            "R/R (TP2)",
                        ],
                        "Değer": [
                            "GEÇTİ ✅" if plan.above_52w_low_25pct else "KALDI ❌",
                            f"{plan.low_52w:.2f}" if np.isfinite(plan.low_52w) else "—",
                            f"%{plan.dist_from_52w_low_pct:.2f}" if np.isfinite(plan.dist_from_52w_low_pct) else "—",
                            f"{plan.entry_low:.2f} – {plan.entry_high:.2f}",
                            f"{plan.entry_mid:.2f}",
                            f"{plan.dist_to_entry_pct:+.2f}%",
                            f"{plan.watch_level:.2f}",
                            f"{plan.stop:.2f}",
                            f"{plan.tp1:.2f}",
                            f"{plan.tp2:.2f}",
                            f"1:{plan.rr1:.2f}" if np.isfinite(plan.rr1) else "—",
                            f"1:{plan.rr2:.2f}" if np.isfinite(plan.rr2) else "—",
                        ],
                    }
                )
                st.table(table)

                st.subheader("🧠 Skor Dağılımı")
                b = plan.breakdown
                bdf = pd.DataFrame(
                    {
                        "Bileşen": [
                            "Trend",
                            "Fiyat/EMA150",
                            "Momentum (RSI)",
                            "Volatilite (ATR%)",
                            "Uzama (EMA50)",
                            "Minervini #5 (52W Dip +%25)",
                        ],
                        "Puan": [
                            b.trend_stack,
                            b.price_vs_ema150,
                            b.momentum_rsi,
                            b.volatility_atr,
                            b.extension_vs_ema50,
                            b.minervini_52w,
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

                st.subheader("🎯 İşlem Yönetimi (Dinamik Stop + TP Revizyonu)")
                # User can input their real entry/stop/tps
                colx1, colx2 = st.columns(2)
                with colx1:
                    entry_price = st.number_input("Entry (Gerçek Alış / Planlanan Entry)", value=float(st.session_state.get("__entry_price", plan.entry_mid)), step=0.01, format="%.2f")
                    user_stop = st.number_input("Stop Loss (Senin)", value=float(st.session_state.get("__user_stop", plan.stop)), step=0.01, format="%.2f")
                with colx2:
                    user_tp1 = st.number_input("TP1 (Senin)", value=float(st.session_state.get("__user_tp1", plan.tp1)), step=0.01, format="%.2f")
                    user_tp2 = st.number_input("TP2 (Senin)", value=float(st.session_state.get("__user_tp2", plan.tp2)), step=0.01, format="%.2f")

                st.session_state["__entry_price"] = float(entry_price)
                st.session_state["__user_stop"] = float(user_stop)
                st.session_state["__user_tp1"] = float(user_tp1)
                st.session_state["__user_tp2"] = float(user_tp2)

                mgmt = manage_open_trade(
                    df=df,
                    plan=plan,
                    price=float(last_price_line),
                    entry_price=float(entry_price),
                    user_stop=float(user_stop),
                    user_tp1=float(user_tp1),
                    user_tp2=float(user_tp2),
                )

                # Display management recommendation
                m1, m2, m3, m4 = st.columns(4)
                prof = mgmt.get("profit_pct", np.nan)
                m1.metric("Kâr/Zarar % (Entry → Fiyat)", f"{prof:.2f}%" if np.isfinite(prof) else "—")
                m2.metric("Önerilen Stop", f"{mgmt.get('proposed_stop', np.nan):.2f}" if np.isfinite(mgmt.get("proposed_stop", np.nan)) else "—")
                m3.metric("Önerilen TP1", f"{mgmt.get('proposed_tp1', np.nan):.2f}" if np.isfinite(mgmt.get("proposed_tp1", np.nan)) else "—")
                m4.metric("Önerilen TP2", f"{mgmt.get('proposed_tp2', np.nan):.2f}" if np.isfinite(mgmt.get("proposed_tp2", np.nan)) else "—")

                st.info(f"**Aksiyon:** {mgmt.get('action','—')}  \n{mgmt.get('note','')}")

                # Quick buttons to apply proposal into inputs (session only)
                apply_cols = st.columns(3)
                with apply_cols[0]:
                    if st.button("Önerilen Stop'u Uygula", use_container_width=True, disabled=not np.isfinite(mgmt.get("proposed_stop", np.nan))):
                        st.session_state["__user_stop"] = float(mgmt["proposed_stop"])
                        st.rerun()
                with apply_cols[1]:
                    if st.button("Önerilen TP1'i Uygula", use_container_width=True, disabled=not np.isfinite(mgmt.get("proposed_tp1", np.nan))):
                        st.session_state["__user_tp1"] = float(mgmt["proposed_tp1"])
                        st.rerun()
                with apply_cols[2]:
                    if st.button("Önerilen TP2'yi Uygula", use_container_width=True, disabled=not np.isfinite(mgmt.get("proposed_tp2", np.nan))):
                        st.session_state["__user_tp2"] = float(mgmt["proposed_tp2"])
                        st.rerun()

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

            user_stop = st.session_state.get("__user_stop", None)
            user_tp1 = st.session_state.get("__user_tp1", None)
            user_tp2 = st.session_state.get("__user_tp2", None)

            fig = plot_chart(df, ticker, plan, last_price_line, user_stop=user_stop, user_tp1=user_tp1, user_tp2=user_tp2)
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2: PORTFOLIO
# =========================================================
with tab_portfolio:
    st.subheader("🧳 Portföy Analiz")
    st.caption("Portföy satırlarını gir: ticker, alış ort., stop, TP1, TP2. Sonra analiz et → her hisse için teknik + risk + dinamik yönetim önerisi.")

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

                            # 52W low
                            low52 = float("nan")
                            if interval == "1day" and len(dfi) >= 252:
                                low52 = float(dfi.tail(252)["low"].min())
                            else:
                                try:
                                    low52, _ = td_52w_low(tkr)
                                except Exception:
                                    low52 = float("nan")

                            plan = build_trade_plan(dfi, low_52w=low52)

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

                            pos_value = (qty * price) if np.isfinite(qty) and np.isfinite(price) else np.nan
                            risk_per_share = (price - user_stop) if (np.isfinite(user_stop) and np.isfinite(price)) else np.nan
                            risk_value = (risk_per_share * qty) if (np.isfinite(risk_per_share) and np.isfinite(qty)) else np.nan

                            # If user didn't fill stop/tps, fall back to plan levels
                            eff_stop = user_stop if np.isfinite(user_stop) and user_stop > 0 else plan.stop
                            eff_tp1 = user_tp1 if np.isfinite(user_tp1) and user_tp1 > 0 else plan.tp1
                            eff_tp2 = user_tp2 if np.isfinite(user_tp2) and user_tp2 > 0 else plan.tp2
                            eff_entry = avg_cost if (np.isfinite(avg_cost) and avg_cost > 0) else plan.entry_mid

                            mgmt = manage_open_trade(
                                df=dfi,
                                plan=plan,
                                price=float(price),
                                entry_price=float(eff_entry),
                                user_stop=float(eff_stop),
                                user_tp1=float(eff_tp1),
                                user_tp2=float(eff_tp2),
                            )

                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": round(price, 2),
                                "Entry": round(eff_entry, 2) if np.isfinite(eff_entry) else "",
                                "P&L %": round(pnl_pct, 2) if np.isfinite(pnl_pct) else "",
                                "Stop": round(eff_stop, 2) if np.isfinite(eff_stop) else "",
                                "Stop Mesafe %": round(dist_stop_pct, 2) if np.isfinite(dist_stop_pct) else "",
                                "TP1": round(eff_tp1, 2) if np.isfinite(eff_tp1) else "",
                                "TP1 Mesafe %": round(dist_tp1_pct, 2) if np.isfinite(dist_tp1_pct) else "",
                                "TP2": round(eff_tp2, 2) if np.isfinite(eff_tp2) else "",
                                "TP2 Mesafe %": round(dist_tp2_pct, 2) if np.isfinite(dist_tp2_pct) else "",
                                "52W Dip": round(plan.low_52w, 2) if np.isfinite(plan.low_52w) else "",
                                "Minervini#5": "YES" if plan.above_52w_low_25pct else "NO",
                                "Setup": plan.setup_score,
                                "Timing": plan.timing_score,
                                "Durum": plan.status_tag,
                                "Öneri Aksiyon": mgmt.get("action", ""),
                                "Öneri Stop": round(mgmt.get("proposed_stop", np.nan), 2) if np.isfinite(mgmt.get("proposed_stop", np.nan)) else "",
                                "Öneri TP1": round(mgmt.get("proposed_tp1", np.nan), 2) if np.isfinite(mgmt.get("proposed_tp1", np.nan)) else "",
                                "Öneri TP2": round(mgmt.get("proposed_tp2", np.nan), 2) if np.isfinite(mgmt.get("proposed_tp2", np.nan)) else "",
                                "Poz. Değeri": round(pos_value, 2) if np.isfinite(pos_value) else "",
                                "Risk $": round(risk_value, 2) if np.isfinite(risk_value) else "",
                                "Not": mgmt.get("note", ""),
                            })

                        except Exception as e:
                            rows.append({
                                "Ticker": tkr,
                                "Fiyat": "",
                                "Entry": "",
                                "P&L %": "",
                                "Stop": "",
                                "Stop Mesafe %": "",
                                "TP1": "",
                                "TP1 Mesafe %": "",
                                "TP2": "",
                                "TP2 Mesafe %": "",
                                "52W Dip": "",
                                "Minervini#5": "",
                                "Setup": "",
                                "Timing": "",
                                "Durum": "HATA",
                                "Öneri Aksiyon": "DİKKAT",
                                "Öneri Stop": "",
                                "Öneri TP1": "",
                                "Öneri TP2": "",
                                "Poz. Değeri": "",
                                "Risk $": "",
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

                    colx, coly, colz, colw = st.columns(4)
                    colx.metric("🟢 Alım Bölgesi", len(a))
                    coly.metric("🟡 Pullback", len(b))
                    colz.metric("⚫ Uzamış", len(c))
                    colw.metric("🔴 Kırmızı", len(d))

                # Download results
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Portföy Analiz CSV indir",
                    data=csv_bytes,
                    file_name="portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
