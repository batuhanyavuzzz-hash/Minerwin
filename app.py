# app.py
# MinerWin — Tek Hisse + Portföy Analiz (V7.0) — Twelve Data + Finnhub
#
# V7.2 Değişiklikleri (V7.1 üzerine — KALICI HISTORY):
#  ★ GitHub Gist senkronu: history.csv artık Cloud'un geçici diskinde değil,
#    kullanıcının GitHub hesabındaki gizli bir Gist'te yaşar.
#    - Açılışta Gist'ten çekilir, yerel kayıtlarla birleştirilir (hiçbir şey ezilmez)
#    - Her analizde hem yerel dosyaya hem Gist'e yazılır (hata analizi bloklamaz)
#    - GITHUB_TOKEN yoksa eski düzen (yerel + indir/yükle) aynen çalışır
#  ★ Sanitizer github_pat_ token'larını da maskeler.
#
# V7.1 Değişiklikleri (V7.0 üzerine — OMURGA REFAKTÖRÜ):
#  ★ ANAYASA: (1) Haftalık=KAPI, günlük=TETİK — kapı kapalıyken günlük karar
#    dili hiç konuşmaz (UI+PDF+grafik). (2) Alarm=haftalık bant, her durumda
#    görünür. (3) Günlüğün tek görevi alarm sonrası teyit. (4) Pozisyon YÖNETİMİ
#    kapıya tabi değildir. (5) RET≠BEKLEMEDE: "aday değil" / "aday, fiyat
#    bekleniyor". (6) Program DANIŞMANDIR: emir dili yok, veri saklanmaz.
#  ★ gate alanı (RET/BEKLEMEDE/ACIK) tüm sunumu yönetir; RS<45=RET,
#    RS 45-60 artık veto değil bilgi notu. Emir dili tamamen söküldü
#    (UZAK DUR / kovalamadır / ALINAMAZ / Boyutlama yapılmaz → tarif dili).
#  ★ Bellek önlemi: tüm cache'lere max_entries sınırı (Cloud çökmelerine karşı).
#
# V7.0 Değişiklikleri (V6.3.3 üzerine — ARAYÜZ YENİLEME):
#  ★ SWING MODU (yeni varsayılan görünüm): Kullanıcının gerçek iş akışını
#    tek ekranda yürütür — zaman dilimi seçimi YOK, kafa karışıklığı YOK:
#      Piyasa rejimi → Karar kartı (🟢/🟡/🔴 + gerekçe) → Haftalık bölüm
#      (setup + ALARM BANDI) → Günlük bölüm (timing + stop/TP planı) →
#      Bilanço kontrolü → Günlük/Haftalık geçişli grafik → detaylı PDF
#  ★ GELİŞMİŞ MOD: Eski ekran olduğu gibi korundu (tüm zaman dilimleri,
#    skor dağılımı, debug). Tek hisse sekmesinin üstündeki anahtarla geçilir.
#  ★ Motor koduna DOKUNULMADI — aynı hesaplar, yeni sunum. API maliyeti artmadı.
#  + build_mtf_summary artık plan/df nesnelerini de döndürür (_w_plan, _d_plan,
#    _wdf, _ddf) — Swing Modu grafiği ve planı bunlardan çizer.
#  ★ RİSK YÖNETİMİ: Sidebar'a hesap büyüklüğü + işlem başına risk% girişi.
#    Pozisyon boyutu hesaplayıcı (adet/maliyet/risk $) Swing ve Gelişmiş modda;
#    portföyde Toplam Açık Risk, Açık Risk/Hesap %, En Büyük Pozisyon % KPI'ları.
#  ★ RS RATING KARARA BAĞLANDI: RS < 45 → 🔴 veto; RS < 60 → 🟢 verilmez (🟡'ya
#    düşürülür). Minervini prensibi: lider olmayan hisse aday bile değildir.
#  ★ DAĞITIM GÜNÜ SAYIMI: SPY'da son 25 seansta fiyat ↓ + hacim ↑ günleri sayılır.
#    ≥6 dağıtım günü → rejim 🟢'den 🟡'ya düşürülür (kurumsal satış erken uyarısı).
#  ★ SWING KARAR RÖTUŞLARI (saha geri bildirimi):
#    - Haftalık ⚫ UZAMIŞ ise karar 🟢 olamaz → bilgilendirici 🟡 (devam girişi
#      kovalamadır; kullanıcının stratejisi haftalık banda pullback beklemektir)
#    - 🟡 mesajları aktif takip planı içerir (alarm bandı + "alarmın kurulu kalsın")
#    - Pozisyon boyutu: "hesaplanamadı" yerine gerçek sebep ("1 adet bile hedef
#      riski aşıyor — bu hisse mevcut risk kuralınla alınamaz")
#    - Teyit bandı bekleme durumlarında bağlamlı nota dönüşür (bugünkü değer,
#      giriş gününde geçerli olmayacak uyarısıyla); sadece 🟢'de metrik kalır
#  ★ EVRE GÖSTERİMİ (saha geri bildirimi — "bantlar tutmuyor" karışıklığı):
#    - Günlük grafiğe haftalık ALARM bandı turuncu gölge olarak çizilir
#      (haftalık grafiğe de günlük bant) — iki bandın konumu tek bakışta
#    - Karar kartına 📍 Evre satırı: geometri sözle anlatılır ("Uzamış — alarm
#      %13 aşağıda", "ALARM BÖLGESİNDE — günlük 🟢 teyidi bekle" vb.)
#  ★ HÜKÜM KARTI + TUTARLILIK (saha geri bildirimi — "çıktılar çelişmesin"):
#    İlke: BİR HİSSE, BİR KARAR, HER YERDE AYNI SES.
#    - Swing en üstte net hüküm: KARAR + Neden + Alarm. Alarm dili düzeltildi:
#      alım seviyesi DEĞİL, yeniden analiz tetiği (filtre giriş ANINDA geçilir)
#    - Pozisyon fişi hükme tabi: karar 🟢 değilse adet verilmez (ekran + PDF)
#    - PDF senaryosu hükme hizalı; İşlem Planı "bugünkü referans" etiketli
#    - Gelişmiş modun en üstünde aynı Swing hükmü — mod farkı ses farkı değil
#  ★ PDF PROFESYONELLEŞTİRME (Seviye 1 — veri bütünlüğü):
#    - Tek hisse: Pozisyon Boyutu satırı, MTF tablosuna RS Rating,
#      KPI'lara Dağıtım Günü sayısı
#    - Portföy: Açık Risk/Hesap %, En Büyük Pozisyon %, tabloya Aksiyon kolonu
#
# V6.3.3 Değişiklikleri (V6.3.2 üzerine):
#  + PDF çıktıları V6.3 özellikleriyle senkronize edildi:
#    - Tek hisse PDF: Piyasa rejimi + sonraki bilanço KPI satırı,
#      yaklaşan bilanço uyarı kutusu (≤14 gün), MTF Özet tablosu
#      (haftalık setup/günlük timing/karar/alarm bandı)
#    - Portföy PDF: Piyasa rejimi KPI'ı + "Bilanço" kolonu
#
# V6.3.2 Değişiklikleri (V6.3.1 üzerine):
#  + Finnhub yedek kaynağı: Bilanço tarihleri için Twelve Data /earnings
#    başarısız olursa (403 = planda yok) otomatik Finnhub'a düşülür.
#    Secrets'a FINNHUB_API_KEY eklenmesi yeterli — yoksa eski davranış korunur.
#  + Sanitizer artık Finnhub'ın token= parametresini de maskeler.
#  + UI'da bilanço kaynağı gösterilir (TwelveData / Finnhub).
#
# V6.3.1 Değişiklikleri (V6.3 üzerine — GÜVENLİK düzeltmesi):
#  !! Hata mesajlarında API anahtarı sızıntısı kapatıldı. requests'in HTTPError
#     mesajı tam URL'yi (apikey dahil) içeriyordu ve UI'da gösteriliyordu.
#     Artık tüm hata mesajlarından apikey maskeleniyor (_sanitize_err).
#  !! 403 (plan desteklemiyor) hatası kullanıcı dostu mesaja çevrildi.
#  !! Earnings 403 alınca oturum boyunca tekrar denenmez (kredi israfı önlenir).
#
# V6.3 Değişiklikleri (V6.2.1 üzerine — yeni özellikler):
#  A. Piyasa Sağlığı Paneli: SPY bazlı rejim göstergesi (🟢 RİSK AÇIK / 🟡 TEMKİNLİ /
#     🔴 RİSK KAPALI). Sekmelerin üstünde butonla, analizlerde otomatik gösterilir.
#     Rejim kırmızıysa alım uyarısı verilir.
#  B. Bilanço (Earnings) Uyarısı: Yaklaşan bilanço 14 gün içindeyse gap riski uyarısı.
#     Tek hissede banner, portföyde "Bilanço" kolonu. Sidebar'dan kapatılabilir.
#     Not: Twelve Data free planda earnings endpoint'i desteklenmeyebilir — bu durumda
#     uygulama kırılmaz, bilgi notu gösterilir.
#  C. MTF Özet (Haftalık + Günlük): Hangi timeframe'de analiz yaparsan yap, haftalık
#     setup skoru + günlük timing skoru yan yana gösterilir; birleşik karar
#     (🟢 SİNYAL / 🟡 İZLE / 🔴 UZAK DUR) ve haftalık EMA20–EMA50 alarm bandı verilir.
#
# V6.2.1 Değişiklikleri (V6.2 üzerine — kod incelemesi düzeltmeleri):
#  1. FIX: max_loss_stop artık sadece gerçek zarar üreten bacakları topluyor
#     (break-even üstü stoplar "maks zarar"ı yanlış şişiriyordu)
#  2. FIX: RSI — hiç düşüş olmayan pencerede NaN yerine 100 üretir
#  3. FIX: history.csv — mevcut dosyanın header'ına hizalanarak yazılır (şema kayması önlendi)
#  4. FIX: Dar baz tespiti sabit referans pencere (120 bar) kullanır — bar slider'ından bağımsız
#  5. FIX: Kırılım hacim teyidi shift(1) ile — bugünün hacmi kendi ortalamasını şişirmez
#  6. FIX: Twelve Data rate limit (429) yakalanır, bekleyip 2 kez yeniden dener
#  7. FIX: TP2 zemin garantisi cap'i deldiğinde işaretlenir ve UI'da uyarı gösterilir
#  8. FIX: check_weekly_trend / quote hataları sessizce yutulmaz, UI'da caption gösterilir
#  9. FIX: datetime.utcnow() (deprecated) → datetime.now(timezone.utc);
#     rapor tarihleri Europe/Istanbul saat dilimiyle yazılır
# 10. FIX: st.data_editor ayrı widget key ("pf_editor") ile kullanılır (rerun kayıp riskine karşı)
# 11. FIX: import re dosya ortasından üste taşındı; ws_sum[f"A13"] → ws_sum["A13"]
# 12. Portföy dosyası bölümüne ortak/geçici disk uyarısı eklendi (Streamlit Cloud)

import io
import os
import csv
import html
import time
import base64
import re as _re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, Tuple

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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

# NEW (V6.3): Bilanço uyarısı için gün eşiği
EARNINGS_WARN_DAYS = 14

# FIX (V6.2.1): Rapor tarihleri için Türkiye saat dilimi
TR_TZ = ZoneInfo("Europe/Istanbul")

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
<div class="sub-title">Minervini-Based Technical Trading Engine — V7.2</div>
""",
    unsafe_allow_html=True,
)
st.divider()

API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
# FIX (V6.3.1): Secrets'a yapıştırırken kalan boşluk/tırnak 401'e yol açabiliyor
if isinstance(API_KEY, str):
    API_KEY = API_KEY.strip().strip('"').strip("'")
if not API_KEY:
    st.error('TWELVEDATA_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets içine ekle: TWELVEDATA_API_KEY="..."')
    st.stop()

BASE_URL = "https://api.twelvedata.com"

# NEW (V6.3.2): Finnhub — bilanço tarihleri için opsiyonel yedek kaynak.
# Tanımlı değilse uygulama aynen çalışır, sadece TD 403 verirse bilanço özelliği susar.
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
if isinstance(FINNHUB_API_KEY, str):
    FINNHUB_API_KEY = FINNHUB_API_KEY.strip().strip('"').strip("'")

# NEW (V7.2): GitHub Gist — history için kalıcı bulut depolama (opsiyonel).
# Token yoksa uygulama eski düzende (yerel dosya + indir/yükle) çalışır.
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
if isinstance(GITHUB_TOKEN, str):
    GITHUB_TOKEN = GITHUB_TOKEN.strip().strip('"').strip("'")
# NEW (V7.2): Kural seti sürümü — history kayıtlarına yazılır. Filtre
# eşiklerinden biri (RS, setup, uzamış %8, dağıtım ≥6...) değiştirildiğinde
# BU SAYI ELLE ARTIRILIR ki karne "hangi kural dönemine ait karar" bilsin.
RULE_VER = "v1"

GIST_DESC = "minerwin-history (otomatik — MinerWin uygulamasi)"
GIST_FILENAME = "history.csv"
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


# NEW (V7.0): Pozisyon boyutu hesaplayıcı
def position_size_calc(account_size: float, risk_pct: float, entry: float, stop: float) -> Dict[str, Any]:
    """shares = (hesap × risk%) ÷ (giriş − stop). Maliyet hesabı aşarsa
    kaldıraçsız üst sınıra çekilir ve 'capped' işaretlenir."""
    out = {"shares": np.nan, "cost": np.nan, "risk_amt": np.nan, "capped": False}
    if not (np.isfinite(account_size) and account_size > 0
            and np.isfinite(risk_pct) and risk_pct > 0
            and np.isfinite(entry) and np.isfinite(stop)
            and entry > stop > 0):
        return out
    risk_amt = account_size * (risk_pct / 100.0)
    per_share_risk = entry - stop
    out["per_share_risk"] = float(per_share_risk)
    shares = int(risk_amt // per_share_risk)
    if shares <= 0:
        # NEW (V7.0): 1 adet bile hedef riski aşıyor — sessiz NaN yerine sebep döndür
        out["reason"] = "risk_exceeds"
        return out
    cost = shares * entry
    if cost > account_size:
        shares = int(account_size // entry)
        if shares <= 0:
            return out
        cost = shares * entry
        out["capped"] = True
    out.update({
        "shares": float(shares),
        "cost": float(cost),
        "risk_amt": float(shares * per_share_risk),
    })
    return out


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
    # FIX (V6.2.1): Hiç düşüş olmayan pencerede avg_loss=0 → RSI NaN kalıyordu
    # ve bfill() sondaki NaN'ları dolduramıyordu. Doğru değerler atanır:
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
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
# FIX (V6.3.1): Hata mesajlarından API anahtarını maskeler.
# requests'in HTTPError string'i tam URL'yi (apikey dahil) içerir —
# bu mesaj UI'da gösterildiğinde anahtar sızıyordu.
# NEW (V6.3.2): Finnhub'ın token= parametresi de maskelenir.
_APIKEY_RE = _re.compile(r"(apikey|token)=[A-Za-z0-9]+")


_GHPAT_RE = _re.compile(r"github_pat_[A-Za-z0-9_]+")


def _sanitize_err(msg) -> str:
    s = _APIKEY_RE.sub(r"\1=***", str(msg))
    return _GHPAT_RE.sub("github_pat_***", s)


def _td_get(endpoint: str, params: dict, timeout: int = 25, max_retries: int = 2) -> dict:
    """
    FIX (V6.2.1): Twelve Data GET — rate limit (429) durumunda bekleyip yeniden dener.
    Free planda dakikada 8 kredi vardır; portföy analizi limiti kolayca aşabilir.
    429 hem HTTP status hem JSON body içindeki "code" alanı olarak gelebilir.
    FIX (V6.3.1): HTTP hataları apikey maskelenerek fırlatılır; 403 için
    kullanıcı dostu "plan desteklemiyor" mesajı verilir.
    """
    last_msg = "rate limit"
    for attempt in range(max_retries + 1):
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
        if r.status_code == 429:
            last_msg = "HTTP 429 — dakikalık kredi doldu"
            if attempt < max_retries:
                time.sleep(15)
                continue
            raise RuntimeError(f"TwelveData rate limit: {last_msg}. Biraz bekleyip tekrar dene.")
        if r.status_code == 401:
            raise RuntimeError(
                "TwelveData: 401 Unauthorized — API anahtarı geçersiz veya iptal edilmiş. "
                "Streamlit Cloud → Settings → Secrets içindeki TWELVEDATA_API_KEY değerini "
                "yeni anahtarla güncelle ve uygulamayı yeniden başlat (Reboot)."
            )
        if r.status_code == 403:
            raise RuntimeError(
                f"TwelveData /{endpoint}: 403 Forbidden — bu endpoint mevcut API planında desteklenmiyor."
            )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as he:
            raise RuntimeError(_sanitize_err(he)) from None
        data = r.json()
        if isinstance(data, dict) and str(data.get("code")) == "429":
            last_msg = str(data.get("message", "rate limit"))
            if attempt < max_retries:
                time.sleep(15)
                continue
            raise RuntimeError(f"TwelveData rate limit: {_sanitize_err(last_msg)}")
        return data
    raise RuntimeError(f"TwelveData rate limit: {_sanitize_err(last_msg)}")


@st.cache_data(ttl=120, max_entries=64)
def td_time_series(symbol: str, interval: str, outputsize: int) -> dict:
    return _td_get(
        "time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": API_KEY,
            "format": "JSON",
        },
        timeout=25,
    )


@st.cache_data(ttl=120, max_entries=64)
def td_quote(symbol: str) -> dict:
    return _td_get(
        "quote",
        params={"symbol": symbol, "apikey": API_KEY, "format": "JSON"},
        timeout=20,
    )


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
@st.cache_data(ttl=300, max_entries=32)
def _fetch_daily_df(symbol: str, outputsize: int = 320) -> pd.DataFrame:
    payload = td_time_series(symbol, "1day", int(outputsize))
    return parse_ohlcv(payload)


@st.cache_data(ttl=600, max_entries=32)
def _fetch_weekly_df(symbol: str, outputsize: int = 60) -> pd.DataFrame:
    """Weekly veri çeker — weekly trend kontrolü için kullanılır."""
    payload = td_time_series(symbol, "1week", int(outputsize))
    return parse_ohlcv(payload)


def check_weekly_trend(symbol: str) -> Dict[str, Any]:
    result = {"weekly_trend_ok": None, "warning": "", "weekly_close": float("nan"), "weekly_ma10": float("nan")}
    try:
        wdf = _fetch_weekly_df(symbol, 60)
        if wdf is None or len(wdf) < 12:
            return result
        wdf["ma10"] = wdf["close"].rolling(10).mean()
        last = wdf.iloc[-1]
        weekly_close = float(last["close"])
        weekly_ma10 = float(last["ma10"])
        if not (np.isfinite(weekly_close) and np.isfinite(weekly_ma10)):
            return result
        ma10_slope = slope(wdf["ma10"], lookback=4)
        trend_ok = (weekly_close > weekly_ma10) and (np.isfinite(ma10_slope) and ma10_slope > 0)
        result["weekly_trend_ok"] = trend_ok
        result["weekly_close"] = weekly_close
        result["weekly_ma10"] = weekly_ma10
        result["weekly_ma10_slope"] = float(ma10_slope) if np.isfinite(ma10_slope) else float("nan")
        if not trend_ok:
            result["warning"] = "⚠️ Weekly trend zayıf — büyük trend teyitsiz"
    except Exception as e:
        # FIX (V6.2.1): Hata sessizce yutulmuyor — UI'da gösterilmek üzere kaydedilir
        result["error"] = _sanitize_err(e)
    return result


# =========================================================
# BİLANÇO (EARNINGS) — NEW V6.3
# =========================================================
@st.cache_data(ttl=3600, max_entries=64)
def td_earnings(symbol: str) -> dict:
    """Twelve Data /earnings — sembolün geçmiş + yaklaşan bilanço tarihleri.
    Not: Free planda bu endpoint desteklenmeyebilir; çağıran taraf hatayı
    yakalayıp bilgi notu gösterir, uygulama kırılmaz."""
    return _td_get(
        "earnings",
        params={"symbol": symbol, "outputsize": 8, "apikey": API_KEY, "format": "JSON"},
        timeout=20,
    )


@st.cache_data(ttl=3600, max_entries=64)
def finnhub_earnings(symbol: str) -> dict:
    """NEW (V6.3.2): Finnhub earnings calendar — Twelve Data /earnings planda
    yoksa yedek kaynak. Bugünden +120 güne kadarki bilanço tarihlerini çeker."""
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY tanımlı değil (Streamlit Secrets).")
    today = datetime.now(TR_TZ).date()
    r = requests.get(
        "https://finnhub.io/api/v1/calendar/earnings",
        params={
            "from": today.isoformat(),
            "to": (today + timedelta(days=120)).isoformat(),
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
        timeout=20,
    )
    if r.status_code == 401:
        raise RuntimeError("Finnhub: 401 — API anahtarı geçersiz (Secrets'taki FINNHUB_API_KEY'i kontrol et).")
    if r.status_code == 429:
        raise RuntimeError("Finnhub: 429 — dakikalık limit doldu, biraz sonra tekrar dene.")
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as he:
        raise RuntimeError(_sanitize_err(he)) from None
    return r.json()


def _parse_dates(items, key: str = "date") -> list:
    """Sözlük listesinden geçerli YYYY-MM-DD tarihlerini ayrıştırır."""
    dates = []
    for e in items:
        if not isinstance(e, dict):
            continue
        try:
            dates.append(datetime.strptime(str(e.get(key, "")), "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def next_earnings_info(symbol: str) -> Dict[str, Any]:
    """En yakın gelecek bilanço tarihini ve kaç gün kaldığını döndürür.
    FIX (V6.3.1): TD 403 (plan desteklemiyor) alındıysa oturum boyunca tekrar
    denenmez — portföyde ticker başına boşa kredi harcanmaz.
    NEW (V6.3.2): TD başarısız olursa Finnhub'a düşülür (anahtar tanımlıysa)."""
    out = {"date": None, "days": None, "error": "", "source": ""}
    today = datetime.now(TR_TZ).date()

    def _pick(dates) -> bool:
        future = [d for d in dates if d >= today]
        if future:
            nd = min(future)
            out["date"] = nd.isoformat()
            out["days"] = int((nd - today).days)
            return True
        return False

    # --- 1) Twelve Data (plan destekliyorsa) ---
    if not st.session_state.get("__earnings_unsupported"):
        try:
            data = td_earnings(symbol)
            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(_sanitize_err(data.get("message", "earnings hatası")))
            vals = (data.get("earnings") or data.get("values") or []) if isinstance(data, dict) else []
            _pick(_parse_dates(vals))
            out["source"] = "TwelveData"
            return out
        except Exception as ex:
            out["error"] = _sanitize_err(ex)
            if "403" in out["error"] or "desteklenmiyor" in out["error"]:
                st.session_state["__earnings_unsupported"] = True
            # Finnhub başarılı olursa bu hata aşağıda temizlenir

    # --- 2) Finnhub fallback ---
    if FINNHUB_API_KEY:
        try:
            data = finnhub_earnings(symbol)
            cal = data.get("earningsCalendar", []) if isinstance(data, dict) else []
            _pick(_parse_dates(cal))
            out["source"] = "Finnhub"
            out["error"] = ""
            return out
        except Exception as ex:
            fh_err = _sanitize_err(ex)
            out["error"] = (out["error"] + " | " if out["error"] else "") + fh_err
            return out

    if not out["error"]:
        out["error"] = "Earnings kaynağı yok (Twelve Data planı desteklemiyor, FINNHUB_API_KEY tanımlı değil)."
    return out


# =========================================================
# PİYASA SAĞLIĞI (SPY REJİM) — NEW V6.3
# =========================================================
def market_health_pack(spy_df: pd.DataFrame) -> Dict[str, Any]:
    """SPY üzerinden piyasa rejimini belirler (Minervini'nin M harfi).
    🟢 RİSK AÇIK: close > EMA50 > EMA200 ve EMA200 eğimi pozitif
    🔴 RİSK KAPALI: close < EMA200 veya (close < EMA50 ve EMA50 eğimi negatif)
    🟡 TEMKİNLİ: aradaki her durum"""
    out = {
        "regime": "—", "detail": "", "swing_ok": None, "error": "",
        "close": float("nan"), "ema50": float("nan"), "ema200": float("nan"),
        "dist_ema50_pct": float("nan"), "ema200_slope": float("nan"),
    }
    try:
        if spy_df is None or spy_df.empty or len(spy_df) < 210:
            out["error"] = "SPY verisi yetersiz (min 210 bar gerekli)"
            return out
        d = spy_df.copy()
        d["ema50"] = ema(d["close"], 50)
        d["ema200"] = ema(d["close"], 200)
        close = float(d["close"].iloc[-1])
        e50 = float(d["ema50"].iloc[-1])
        e200 = float(d["ema200"].iloc[-1])
        s200 = slope(d["ema200"], lookback=20)
        s50 = slope(d["ema50"], lookback=10)
        out.update({
            "close": close, "ema50": e50, "ema200": e200,
            "dist_ema50_pct": pct(close, e50),
            "ema200_slope": float(s200) if np.isfinite(s200) else float("nan"),
        })

        # NEW (V7.0): Dağıtım günü sayımı (son 25 seans) — erken uyarı.
        # Dağıtım günü: fiyat ≥%0.2 düşer + hacim önceki günden yüksektir
        # (kurumsal satış izi). EMA'lar gecikmeli; bu sayaç tepeyi erken yakalar.
        # FIX (V7.1): (a) Önceki gün hacmi 0/boşsa gün sayılmaz — veri boşluğu
        # sayacı şişirip yüksek değerde "yapıştırabiliyordu". (b) Sayılan günler
        # tarih tarih dökülür (dist_detail) — sayaç artık denetlenebilir.
        dist_days = 0
        dist_detail = []
        try:
            if "volume" in d.columns and len(d) >= 30:
                vv = d["volume"].astype(float).fillna(0.0)
                cc = d["close"].astype(float)
                down = cc < cc.shift(1) * 0.998
                vol_up = (vv > vv.shift(1)) & (vv.shift(1) > 0) & (vv > 0)
                mask = (down & vol_up).tail(25)
                dist_days = int(mask.sum())
                for i in mask[mask].index:
                    if i - 1 in cc.index and cc.loc[i - 1] > 0:
                        dist_detail.append({
                            "Tarih": str(d.loc[i, "time"].date()) if "time" in d.columns else str(i),
                            "Değişim %": round(float(cc.loc[i] / cc.loc[i - 1] - 1) * 100.0, 2),
                            "Hacim ×önceki": round(float(vv.loc[i] / vv.loc[i - 1]), 2) if vv.loc[i - 1] > 0 else float("nan"),
                        })
        except Exception:
            dist_days = 0
            dist_detail = []
        out["dist_days"] = dist_days
        out["dist_detail"] = dist_detail
        out["dist_last"] = dist_detail[-1]["Tarih"] if dist_detail else "—"

        if close > e50 and e50 > e200 and np.isfinite(s200) and s200 > 0:
            out["regime"] = "🟢 RİSK AÇIK"
            out["detail"] = "SPY > EMA50 > EMA200 ve uzun trend pozitif — swing alımları için ortam uygun."
            out["swing_ok"] = True
        elif close < e200 or (close < e50 and np.isfinite(s50) and s50 < 0):
            out["regime"] = "🔴 RİSK KAPALI"
            out["detail"] = "SPY zayıf (EMA200 altı veya EMA50 altı + negatif eğim) — yeni swing alımı için koşullar uygun değil."
            out["swing_ok"] = False
        else:
            out["regime"] = "🟡 TEMKİNLİ"
            out["detail"] = "SPY karışık bölgede — pozisyon boyunu küçült, sadece en güçlü setuplara odaklan."
            out["swing_ok"] = None

        # NEW (V7.0): ≥6 dağıtım günü EMA'lar yeşilken bile rejimi düşürür
        if dist_days >= 6 and out["swing_ok"] is True:
            out["regime"] = "🟡 TEMKİNLİ"
            out["swing_ok"] = None
            out["detail"] = (
                f"EMA dizilimi pozitif AMA son 25 seansta {dist_days} dağıtım günü — "
                f"kurumsal satış birikiyor; pozisyon boyunu küçült, agresif alım yapma."
            )
        elif dist_days >= 4:
            out["detail"] += f" (Dağıtım günü: {dist_days}/25 — izlemede.)"
    except Exception as ex:
        out["error"] = _sanitize_err(ex)
    return out


def render_market_health(mh: Dict[str, Any]):
    if mh.get("error"):
        st.caption(f"ℹ️ Piyasa sağlığı hesaplanamadı: {mh['error']}")
        return
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Piyasa Rejimi (SPY)", mh.get("regime", "—"))
    m2.metric("SPY Kapanış", f"{mh.get('close', float('nan')):.2f}" if np.isfinite(mh.get("close", np.nan)) else "—")
    m3.metric("EMA50 Mesafe", f"{mh.get('dist_ema50_pct', float('nan')):+.2f}%" if np.isfinite(mh.get("dist_ema50_pct", np.nan)) else "—")
    m4.metric("EMA200 Eğim", f"{mh.get('ema200_slope', float('nan')):.3f}" if np.isfinite(mh.get("ema200_slope", np.nan)) else "—")
    m5.metric(
        "Dağıtım Günü (25g)", f"{mh.get('dist_days', 0)}",
        help="Fiyat ≥%0.2 düşüp hacmin arttığı günler. ≥6 kurumsal satış uyarısıdır ve rejimi düşürür.",
    )
    st.caption(mh.get("detail", ""))
    # FIX (V7.1): Sayaç denetlenebilir — hangi günleri saydığı tarih tarih görünür.
    # "Sayı hep aynı mı takılı, canlı mı?" sorusunun cevabı: son tarih ilerliyorsa canlı.
    if mh.get("dist_detail"):
        with st.expander(f"📋 Dağıtım günleri dökümü ({mh.get('dist_days', 0)} gün — son: {mh.get('dist_last', '—')})"):
            st.dataframe(pd.DataFrame(mh["dist_detail"]), hide_index=True, use_container_width=True)
            st.caption("Kontrol: TradingView'da SPY günlük grafikte bu tarihlerin kırmızı + yüksek hacimli olduğunu doğrulayabilirsin.")


# =========================================================
# MTF ÖZET (HAFTALIK + GÜNLÜK) — NEW V6.3
# =========================================================
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema150"] = ema(df["close"], 150)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    return df


def build_mtf_summary(symbol: str, low_52w: float, high_52w: float) -> Dict[str, Any]:
    """Haftalık setup + günlük timing'i tek pakette döndürür.
    Kullanıcının iş akışı: haftalık giriş bandına alarm kur → alarm çalınca
    günlük ile teyit et. Bu özet iki adımı tek ekranda birleştirir."""
    out = {"error": ""}
    try:
        wdf = _add_indicators(_fetch_weekly_df(symbol, 260))
        w_plan = build_trade_plan(wdf, low_52w=low_52w, high_52w=high_52w)

        ddf = _add_indicators(_fetch_daily_df(symbol, 320))
        d_plan = build_trade_plan(ddf, low_52w=low_52w, high_52w=high_52w)

        # NEW (V7.0): RS Rating hesaplanır ve KARARA BAĞLANIR.
        # Minervini prensibi: endekse karşı zayıf hisse lider değildir —
        # teknik görünüm ne olursa olsun aday bile olamaz.
        rs_rating = float("nan")
        try:
            spy_df = _fetch_spy_daily(320)
            rs = analyze_relative_strength(ddf, spy_df)
            rs_rating = float(rs.get("rs_rating", float("nan")))
        except Exception:
            pass

        weekly_ok = (w_plan.setup_score >= 60) and (not w_plan.status_tag.startswith(("🔴", "🟣")))
        daily_green = d_plan.status_tag.startswith("🟢")
        w_extended = w_plan.status_tag.startswith("⚫")
        rs_weak = np.isfinite(rs_rating) and rs_rating < 60
        rs_very_weak = np.isfinite(rs_rating) and rs_rating < 45

        # ================= OMURGA (V7.1) =================
        # 1) Haftalık = kapı, günlük = tetik. Kapı kapalıyken günlük karar
        #    dili HİÇ konuşmaz (UI ve PDF bu 'gate' alanına göre gizler).
        # 2) Alarm = haftalık bant; her durumda gösterilir.
        # 5) RET ≠ BEKLEMEDE: retde "aday değil", uzamışta "aday, fiyat bekleniyor".
        # 6) Program danışmandır: emir dili yok, durum anlatılır.
        # RS<45 kalite kriteridir (RET); RS 45-60 sadece bilgi notudur.
        _wlo, _whi = w_plan.entry_low, w_plan.entry_high
        rs_note = f" · Not: RS {rs_rating:.0f} — endekse görece zayıf" if rs_weak and not rs_very_weak else ""

        if rs_very_weak:
            gate = "RET"
            verdict = (f"Aday değil — RS Rating {rs_rating:.0f}: hisse endekse karşı belirgin zayıf. "
                       f"Kalite kriteri sağlanmıyor.")
            verdict_kind = "error"
        elif not weekly_ok:
            gate = "RET"
            verdict = (f"Aday değil — haftalık kriterler sağlanmıyor "
                       f"(setup {w_plan.setup_score}/100, durum: {w_plan.status_tag}).")
            verdict_kind = "error"
        elif w_extended:
            gate = "BEKLEMEDE"
            verdict = (f"Aday — haftalık yapı kriterlerden geçiyor (setup {w_plan.setup_score}/100). "
                       f"Fiyat haftalık bandın üstünde; giriş bölgesinde değil. "
                       f"Bant: {_wlo:.2f} – {_whi:.2f}.{rs_note}")
            verdict_kind = "warning"
        elif daily_green:
            gate = "ACIK"
            verdict = (f"Giriş koşulları oluşmuş — haftalık yapı uygun, günlük teyit mevcut.{rs_note}")
            verdict_kind = "success"
        else:
            gate = "ACIK"
            # NEW (V7.2): Fiyat banda SERT düşüşle geldiyse (günlük yapı fiyatın
            # üstünde kırık) hüküm bunu adıyla söyler — koşul yapısal-ikili,
            # şiddet sürekli sayı; yeni eşik icat edilmedi.
            _cls = float(d_plan.debug.get("close", float("nan")))
            _dlo = float(d_plan.entry_low) if np.isfinite(d_plan.entry_low) else float("nan")
            if np.isfinite(_cls) and np.isfinite(_dlo) and _dlo > _cls > 0:
                _gap = (_dlo - _cls) / _cls * 100.0
                _apc = float(d_plan.debug.get("atr_pct", float("nan")))
                _amul = (_gap / _apc) if (np.isfinite(_apc) and _apc > 0) else float("nan")
                _sev = f" (≈{_amul:.1f}×ATR)" if np.isfinite(_amul) else ""
                verdict = (f"Aday — haftalık uygun; günlük yapı fiyatın "
                           f"%{_gap:.1f}{_sev} üstünde — teyit için günlük onarım gerekli. "
                           f"Bant: {_wlo:.2f} – {_whi:.2f}.{rs_note}")
            else:
                verdict = (f"Aday — haftalık uygun; günlük teyit henüz oluşmadı. "
                           f"Bant: {_wlo:.2f} – {_whi:.2f}.{rs_note}")
            verdict_kind = "warning"

        out.update({
            "w_setup": w_plan.setup_score, "w_status": w_plan.status_tag,
            "w_entry_low": w_plan.entry_low, "w_entry_high": w_plan.entry_high,
            "d_timing": d_plan.timing_score, "d_status": d_plan.status_tag,
            "d_entry_low": d_plan.entry_low, "d_entry_high": d_plan.entry_high,
            "verdict": verdict, "verdict_kind": verdict_kind,
            "gate": gate,
            "weekly_ok": weekly_ok, "daily_green": daily_green,
            "rs_rating": rs_rating,
            "w_extended": w_extended,
            # NEW (V7.0): Swing Modu bu nesnelerden grafik ve plan çizer
            "_w_plan": w_plan, "_d_plan": d_plan,
            "_wdf": wdf, "_ddf": ddf,
        })
    except Exception as ex:
        out["error"] = _sanitize_err(ex)
    return out


def compute_52w_levels(df: pd.DataFrame, bars_1day: int = 260) -> Tuple[float, float]:
    if df is None or df.empty:
        return float("nan"), float("nan")
    n = min(len(df), int(bars_1day))
    window = df.iloc[-n:]
    low_52w = float(window["low"].min()) if "low" in window.columns else float("nan")
    high_52w = float(window["high"].max()) if "high" in window.columns else float("nan")
    return low_52w, high_52w


def get_daily_52w_levels(symbol: str, interval: str, current_df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
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
    """
    FIX (V6.2.1→V7.2): Şema evrimi güvenli hale getirildi. Eski yöntem yeni
    alanları ATLIYORDU (mevcut header'a hizala); yeni yöntem BİRLEŞİK şema
    kullanır: eski kolonlar korunur, yeni kolonlar eklenir, eski satırlarda
    yeni alanlar boş kalır. CSV asla bozulmaz.
    """
    new_df = pd.DataFrame([row])
    if os.path.isfile(HISTORY_FILE):
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
            merged = pd.concat(
                [old_df.reindex(columns=all_cols), new_df.reindex(columns=all_cols)],
                ignore_index=True,
            )
        except Exception:
            merged = new_df
    else:
        merged = new_df
    merged.to_csv(HISTORY_FILE, index=False)
    # NEW (V7.2): yerel yazımdan sonra Gist'e it — kalıcı bulut kopyası
    _gist_push_history()


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
# GIST SENKRON — NEW V7.2 (kalıcı history)
# =========================================================
def _gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gist_find() -> str:
    """MinerWin gist'inin id'sini bulur; yoksa boş string döner."""
    r = requests.get("https://api.github.com/gists",
                     headers=_gh_headers(), params={"per_page": 100}, timeout=15)
    r.raise_for_status()
    for g in r.json():
        if g.get("description") == GIST_DESC and GIST_FILENAME in (g.get("files") or {}):
            return str(g["id"])
    return ""


def _gist_create(content: str) -> str:
    r = requests.post("https://api.github.com/gists", headers=_gh_headers(),
                      json={"description": GIST_DESC, "public": False,
                            "files": {GIST_FILENAME: {"content": content or "timestamp,ticker\n"}}},
                      timeout=15)
    r.raise_for_status()
    return str(r.json()["id"])


def _gist_read(gid: str) -> str:
    r = requests.get(f"https://api.github.com/gists/{gid}", headers=_gh_headers(), timeout=15)
    r.raise_for_status()
    f = (r.json().get("files") or {}).get(GIST_FILENAME) or {}
    if f.get("truncated") and f.get("raw_url"):
        rr = requests.get(f["raw_url"], headers=_gh_headers(), timeout=20)
        rr.raise_for_status()
        return rr.text
    return f.get("content", "") or ""


def _gist_write(gid: str, content: str):
    r = requests.patch(f"https://api.github.com/gists/{gid}", headers=_gh_headers(),
                       json={"files": {GIST_FILENAME: {"content": content}}}, timeout=20)
    r.raise_for_status()


def _merge_history(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """İki history çerçevesini birleştirir: hiçbir kayıt ezilmez, mükerrerler
    (aynı timestamp+ticker) ayıklanır, tarihe göre sıralanır."""
    frames = [d for d in (df_a, df_b) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    subset = [c for c in ("timestamp", "ticker") if c in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="first")
        if "timestamp" in merged.columns:
            merged = merged.sort_values("timestamp")
    return merged.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _gist_boot() -> dict:
    """Süreç başına BİR KEZ çalışır: Gist'i bulur/oluşturur, uzak kayıtları
    yerelle birleştirip diske yazar. Yereldeki fazla kayıt varsa Gist'e iter.
    Başarısızlık uygulamayı asla durdurmaz — durum sözlükle raporlanır."""
    out = {"gid": "", "status": "kapalı (GITHUB_TOKEN tanımlı değil)"}
    if not GITHUB_TOKEN:
        return out
    try:
        local_df = read_history_df()
        gid = _gist_find()
        if not gid:
            content = local_df.to_csv(index=False) if not local_df.empty else "timestamp,ticker\n"
            gid = _gist_create(content)
            out.update(gid=gid, status=f"aktif — yeni gist oluşturuldu ({len(local_df)} kayıt taşındı)")
            return out
        remote_txt = _gist_read(gid)
        try:
            remote_df = pd.read_csv(io.StringIO(remote_txt)) if remote_txt.strip() else pd.DataFrame()
        except Exception:
            remote_df = pd.DataFrame()
        merged = _merge_history(local_df, remote_df)
        if not merged.empty:
            merged.to_csv(HISTORY_FILE, index=False)
        if len(merged) > len(remote_df):
            _gist_write(gid, merged.to_csv(index=False))
        out.update(gid=gid, status=f"aktif — {len(merged)} kayıt senkronda")
        return out
    except Exception as e:
        out["status"] = f"hata: {_sanitize_err(e)}"
        return out


def _gist_push_history():
    """Yerel history'nin tamamını Gist'e iter. Sessizce başarısız olabilir —
    analiz akışını asla bloklamaz; bir sonraki kayıtta/açılışta arayı kapatır."""
    try:
        boot = _gist_boot()
        if boot.get("gid"):
            df_all = read_history_df()
            if not df_all.empty:
                _gist_write(boot["gid"], df_all.to_csv(index=False))
    except Exception:
        pass


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

# FIX (V6.2.1): Dar baz tespitinde referans pencere sabitlendi.
# Eskiden çekilen tüm pencere (bar slider'ı 120–800) referans alınıyordu;
# aynı hissede slider değişince baz tespiti de değişiyordu.
BASE_REF_WINDOW = 120

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

    # FIX (V6.2.1): Referans olarak son BASE_REF_WINDOW bar kullanılır (sabit pencere)
    ref = df.tail(BASE_REF_WINDOW)
    atr_full = float(ref["atr14"].dropna().mean())
    atr_base = float(df["atr14"].iloc[-BAZ_LOOKBACK:].mean())
    atr_contracted = (
        np.isfinite(atr_full) and np.isfinite(atr_base)
        and atr_full > 0
        and atr_base <= atr_full * ATR_CONTRACT_RATIO
    )

    vol = df["volume"].astype(float).fillna(0.0)
    # FIX (V6.2.1): Hacim referansı da sabit pencereden
    vol_ref = vol.tail(BASE_REF_WINDOW)
    vol_full_mean = float(vol_ref.mean())
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
        # FIX (V6.2.1): shift(1) — bugünün dev hacmi kendi ortalamasını şişirip
        # kırılım eşiğini yükseltmesin diye ortalama bir önceki bara kadar alınır
        vol_50mean = float(vol.rolling(50).mean().shift(1).iloc[-1])

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

    if np.isfinite(high_52w) and high_52w > 0:
        allow_looser_cap = breakout_detected or (
            np.isfinite(dist_to_52w_high_pct) and dist_to_52w_high_pct <= 1.0
        )
        if not allow_looser_cap and close < high_52w * 0.99:
            tp2 = min(tp2, high_52w * 0.98)

    if tp2 <= tp1:
        tp2 = tp1 * 1.06

    # FIX (V6.2.1): 3.5R zemin garantisi cap'leri (momentum/52W) delebiliyor.
    # Bu bilinçli bir tasarım tercihi ama artık işaretleniyor — UI'da uyarı gösterilir
    # ki kullanıcı TP2'nin tarihsel cap'in üzerine zorlandığını bilsin.
    tp2_before_floor = tp2
    tp2 = max(tp2, tp2_floor)
    tp2_floor_override = bool(tp2 > tp2_before_floor + 1e-9)

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
        "tp2_floor_override": tp2_floor_override,
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
    if slope_val > 0.3:
        return "Yükseliyor ↑"
    if slope_val < -0.3:
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
        if rsi_slope_val > 0.3:
            rsi_dir_pts = 5
        elif rsi_slope_val < -0.3:
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
        timing_cmd = "GİRİŞ KOŞULLARI OLUŞMADI"

    if status_tag.startswith("🟢"):
        scenario = (
            "Fiyat giriş bandında (EMA20–EMA50). Bu bölgede satış baskısı zayıflayıp küçük gövdeli mumlar + "
            "hacim düşüşü ile sıkışma görülürse, trend yönünde devam denemesi yapılabilir. Stop altına sarkarsa iptal."
        )
    elif status_tag.startswith("🟡"):
        scenario = (
            "Fiyat şu an giriş bandının dışında. EMA20–EMA50 bandına geri çekilme + hacimde düşüş ile "
            "konsolidasyon beklenir. Bu gerçekleşmeden yapılan alım kovalamaya girer."
        )
    elif status_tag.startswith("🔵"):
        scenario = (
            "Düşük volatilite ile yatay sıkışma var. Kırılımı takip et: güçlü kapanış + hacim artışı gelirse "
            "setup aktifleşir; aksi halde zaman kaybı."
        )
    elif status_tag.startswith("⚫"):
        scenario = (
            "Fiyat EMA50'ye göre uzamış. Pullback gelmeden giriş riskli. En iyi plan: giriş bandına yaklaşmasını "
            "bekle ve orada güç işareti (higher low / güçlü kapanış) ara."
        )
    elif status_tag.startswith("🟣"):
        scenario = (
            "Minervini #5 filtresi geçmiyor (fiyat 52W dip +%25 üstünde değil). Dipten yeni çıkan zayıf yapı olabilir. "
            "Önce güç kanıtı (trend + fiyat aksiyonu) gelmeden swing setup yok."
        )
    else:
        scenario = (
            "Trend filtresi bozulmuş. Önce yeniden EMA150/EMA200 üstüne dönüş ve ortalamaların toparlanması gerekir; "
            "aksi halde swing setup yok."
        )

    targets_reason = (
        f"Targets: kapasite={cap_level}, beklenen taşıma ≈ %{expected_move_pct:.1f} "
        f"(ATR/impuls/52W tavanı ile sınırlandı). "
        f"TP tavan: TP1≤%{targets_dbg.get('tp1_cap_pct', 0):.0f} / TP2≤%{targets_dbg.get('tp2_cap_pct', 0):.0f}"
    )
    # FIX (V6.2.1): TP2 zemin garantisi cap'i deldiyse bunu açıkça belirt
    if targets_dbg.get("tp2_floor_override"):
        targets_reason += " · ⚠️ TP2, 3.5R zemin garantisiyle tavanın üzerine yükseltildi — hedefi temkinli değerlendir."

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
# PDF EXPORT — ORTAK ARAÇLAR  (V6.3 — Profesyonel Tasarım)
# =========================================================

# Renk paleti
_C_DARK      = colors.HexColor("#0F172A")
_C_ACCENT    = colors.HexColor("#2563EB")
_C_ACCENT_LT = colors.HexColor("#DBEAFE")
_C_LIGHT     = colors.HexColor("#F8FAFC")
_C_BORDER    = colors.HexColor("#CBD5E1")
_C_GREEN     = colors.HexColor("#166534")
_C_GREEN_BG  = colors.HexColor("#DCFCE7")
_C_RED       = colors.HexColor("#991B1B")
_C_RED_BG    = colors.HexColor("#FEE2E2")
_C_MID       = colors.HexColor("#64748B")
_C_AMBER     = colors.HexColor("#92400E")
_C_AMBER_BG  = colors.HexColor("#FEF3C7")
_C_PURPLE    = colors.HexColor("#6D28D9")
_C_PURPLE_BG = colors.HexColor("#EDE9FE")
_C_WHITE     = colors.white
_C_ZEBRA     = colors.HexColor("#F1F5F9")


# FIX (V6.2.1): "import re as _re" dosya ortasından üstteki import bloğuna taşındı.

# Emoji'leri silerken Türkçe karakterleri koruyan yardımcı
_EMOJI_RE = _re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010FFFF"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\uFE0F"
    "\u23cf\u23e9-\u23f3\u23f8-\u23fa"
    "\u26A0\u26AA\u26AB"
    "\U0001F7E0-\U0001F7EB"  # colored circles
    "]+", flags=_re.UNICODE
)

def _strip_emoji(text: str) -> str:
    """Emoji'leri ve ok işaretlerini siler, Türkçe karakterleri korur."""
    cleaned = _EMOJI_RE.sub("", text)
    # Ok karakterlerini de kaldır (PDF fontunda kutu olarak görünüyor)
    cleaned = cleaned.replace("↑", "").replace("↓", "").replace("→", "").replace("←", "")
    return cleaned.strip()


def _setup_pdf_fonts() -> tuple[str, str]:
    import reportlab as _rl
    system_candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "MW", "MW-Bold"),
    ]
    for reg, bold, fn, fn_b in system_candidates:
        try:
            if os.path.isfile(reg) and os.path.isfile(bold):
                pdfmetrics.registerFont(TTFont(fn,   reg))
                pdfmetrics.registerFont(TTFont(fn_b, bold))
                return fn, fn_b
        except Exception:
            pass
    rl_fonts = os.path.join(os.path.dirname(_rl.__file__), "fonts")
    try:
        pdfmetrics.registerFont(TTFont("MW",      os.path.join(rl_fonts, "Vera.ttf")))
        pdfmetrics.registerFont(TTFont("MW-Bold", os.path.join(rl_fonts, "VeraBd.ttf")))
        return "MW", "MW-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _pdf_styles(fn: str, fn_bold: str) -> dict:
    base = getSampleStyleSheet()["Normal"]
    def S(name, **kw):
        kw.setdefault("fontName", fn)
        return ParagraphStyle(name, parent=base, **kw)
    return {
        "h1":       S("h1",    fontName=fn_bold, fontSize=20, leading=24, textColor=_C_DARK, spaceAfter=2),
        "h2":       S("h2",    fontName=fn_bold, fontSize=12, leading=16, textColor=_C_ACCENT, spaceAfter=2),
        "h3":       S("h3",    fontName=fn_bold, fontSize=10, leading=13, textColor=_C_DARK, spaceAfter=1),
        "label":    S("label", fontName=fn,      fontSize=7.5, leading=10, textColor=_C_MID),
        "value":    S("value", fontName=fn_bold, fontSize=12, leading=15, textColor=_C_DARK),
        "value_sm": S("value_sm", fontName=fn_bold, fontSize=10, leading=13, textColor=_C_DARK),
        "body":     S("body",  fontSize=8.5, leading=12, textColor=_C_DARK),
        "small":    S("small", fontSize=7.5, leading=10, textColor=_C_MID),
        "warn":     S("warn",  fontName=fn_bold, fontSize=8.5, leading=12, textColor=_C_AMBER),
        "footer":   S("footer", fontSize=7, leading=9, textColor=colors.HexColor("#94A3B8")),
    }


def _section_header(text: str, st_styles: dict, page_w: float) -> list:
    return [
        Spacer(1, 10),
        Paragraph(text, st_styles["h2"]),
        HRFlowable(width="100%", thickness=1.2, color=_C_ACCENT, spaceAfter=6),
    ]


def _pdf_header_story(logo_b64: str, title: str, subtitle: str, st_styles: dict, page_w: float) -> list:
    from reportlab.platypus import Image as RLImage
    story = []
    title_style = ParagraphStyle(
        "banner_title", parent=st_styles["h1"],
        textColor=_C_WHITE, fontSize=18, leading=22,
    )
    sub_style = ParagraphStyle(
        "banner_sub", parent=st_styles["small"],
        textColor=colors.HexColor("#CBD5E1"), fontSize=7.5, leading=10,
    )
    if logo_b64:
        try:
            logo_bytes = base64.b64decode(logo_b64)
            logo_buf   = io.BytesIO(logo_bytes)
            # Logo banner yüksekliğine orantılı — genişlik/yükseklik oranı korunur
            logo_img   = RLImage(logo_buf, width=2.8*cm, height=0.95*cm)
            logo_img.hAlign = "LEFT"
            banner_data = [
                [logo_img, Paragraph(title, title_style)],
                ["",       Paragraph(subtitle, sub_style)],
            ]
            banner_tbl = Table(banner_data, colWidths=[3.2*cm, page_w - 3.2*cm])
        except Exception:
            banner_data = [
                [Paragraph(title, title_style)],
                [Paragraph(subtitle, sub_style)],
            ]
            banner_tbl = Table(banner_data, colWidths=[page_w])
    else:
        banner_data = [
            [Paragraph(title, title_style)],
            [Paragraph(subtitle, sub_style)],
        ]
        banner_tbl = Table(banner_data, colWidths=[page_w])

    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), _C_DARK),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
        ("TOPPADDING",   (0,0), (0,0),   10),
        ("BOTTOMPADDING",(0,-1),(-1,-1),  8),
        ("TOPPADDING",   (0,1), (-1,1),   0),
        ("BOTTOMPADDING",(0,0), (-1,0),   2),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 10))
    return story


def _status_badge(status_tag: str, st_styles: dict, page_w: float, label: str = "DURUM",
                  kind: str = "") -> Table:
    # FIX (V7.2): Renk önceliği verdict_kind'dedir — danışman-dili (emojisiz)
    # cümlelerde nötr/olumlu durumlar varsayılan kırmızıya düşmesin.
    tag_lower = status_tag.lower()
    if kind == "success":
        bg = _C_GREEN
    elif kind == "warning":
        bg = _C_AMBER
    elif kind == "error":
        bg = _C_RED
    elif "alim" in tag_lower or status_tag.startswith("🟢"):
        bg = _C_GREEN
    elif "pullback" in tag_lower or status_tag.startswith("🟡"):
        bg = _C_AMBER
    elif "konsolidasyon" in tag_lower or status_tag.startswith("🔵"):
        bg = _C_ACCENT
    elif "52w" in tag_lower or status_tag.startswith("🟣"):
        bg = _C_PURPLE
    elif "uzam" in tag_lower or status_tag.startswith("⚫"):
        bg = colors.HexColor("#374151")
    else:
        bg = _C_RED

    status_clean = _strip_emoji(status_tag)
    badge_style = ParagraphStyle(
        "badge", parent=st_styles["body"],
        fontName=st_styles["h2"].fontName,
        fontSize=10, leading=14, textColor=_C_WHITE,
    )
    tbl = Table(
        [[Paragraph(f"<b>{html.escape(label)}: {html.escape(status_clean)}</b>", badge_style)]],
        colWidths=[page_w],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    return tbl


def _kpi_card(label: str, value: str, st_styles: dict, width: float,
              accent_color=None) -> Table:
    ac = accent_color or _C_ACCENT
    data = [
        [Paragraph(label, st_styles["label"])],
        [Paragraph(value, st_styles["value_sm"])],
    ]
    tbl = Table(data, colWidths=[width])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _C_LIGHT),
        ("LINEABOVE",     (0,0), (-1,0),  2.5, ac),
        ("BOX",           (0,0), (-1,-1), 0.4, _C_BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (0,0),   6),
        ("BOTTOMPADDING", (0,-1),(-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return tbl


def _kpi_row(items: list[tuple[str, str]], st_styles: dict, page_w: float,
             gap: float = 6, accent_colors: list = None) -> Table:
    n = len(items)
    card_w = (page_w - gap * (n - 1)) / n
    cards = []
    for i, (lbl, val) in enumerate(items):
        ac = accent_colors[i] if accent_colors and i < len(accent_colors) else _C_ACCENT
        cards.append(_kpi_card(lbl, val, st_styles, card_w, accent_color=ac))
    col_ws = []
    spaced_row = []
    for i, card in enumerate(cards):
        spaced_row.append(card)
        col_ws.append(card_w)
        if i < n - 1:
            spaced_row.append("")
            col_ws.append(gap)
    tbl = Table([spaced_row], colWidths=col_ws)
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    return tbl


def _data_table(headers: list[str], body_rows: list[list], st_styles: dict, col_widths: list,
                highlight_col: int = -1, font_size: float = 8) -> Table:
    fn   = st_styles["body"].fontName
    fn_b = st_styles["h2"].fontName

    def safe(v):
        s = str(v) if v is not None else "—"
        return html.escape(s)

    ld = font_size + 3
    hdr_style = ParagraphStyle("tbl_hdr", parent=st_styles["body"],
                                fontName=fn_b, fontSize=font_size, textColor=_C_WHITE, leading=ld)
    cell_style = ParagraphStyle("tbl_cell", parent=st_styles["body"],
                                 fontSize=font_size, leading=ld)

    data = [[Paragraph(html.escape(h), hdr_style) for h in headers]]
    for row in body_rows:
        data.append([Paragraph(safe(v), cell_style) for v in row])

    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0), _C_DARK),
        ("LINEBELOW",    (0,0), (-1,0), 1.5, _C_ACCENT),
        ("GRID",         (0,1), (-1,-1), 0.3, _C_BORDER),
        ("BOX",          (0,0), (-1,-1), 0.6, _C_BORDER),
        ("FONT",         (0,0), (-1,-1), fn),
        ("FONTSIZE",     (0,0), (-1,-1), font_size),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [_C_WHITE, _C_ZEBRA]),
    ]

    if highlight_col >= 0 and len(body_rows) > 0:
        for ri, row in enumerate(body_rows):
            try:
                val = float(row[highlight_col])
                if val > 0:
                    style_cmds.append(("TEXTCOLOR", (highlight_col, ri+1), (highlight_col, ri+1), _C_GREEN))
                elif val < 0:
                    style_cmds.append(("TEXTCOLOR", (highlight_col, ri+1), (highlight_col, ri+1), _C_RED))
            except (ValueError, TypeError, IndexError):
                pass

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _score_bar_table(items: list[tuple[str, int, int]], st_styles: dict, page_w: float) -> Table:
    fn   = st_styles["body"].fontName
    fn_b = st_styles["h2"].fontName

    hdr_style = ParagraphStyle("sb_hdr", parent=st_styles["body"],
                                fontName=fn_b, fontSize=8, textColor=_C_WHITE, leading=11)
    cell_style = ParagraphStyle("sb_cell", parent=st_styles["body"], fontSize=8, leading=11)

    name_w = page_w * 0.32
    bar_w  = page_w * 0.50
    num_w  = page_w * 0.18

    data = [[
        Paragraph("Bileşen", hdr_style),
        Paragraph("", hdr_style),
        Paragraph("Puan / Maks", hdr_style),
    ]]

    for name, pts, mx in items:
        pct_fill = abs(pts / mx * 100) if mx > 0 else 0
        pct_fill = min(100, pct_fill)

        bar_fill_w = bar_w * 0.9 * (pct_fill / 100.0)
        bar_bg_w   = bar_w * 0.9 - bar_fill_w

        if pts > 0:
            bar_color = _C_ACCENT
        elif pts < 0:
            bar_color = colors.HexColor("#EF4444")  # kırmızı bar negatif puan
        else:
            bar_color = colors.HexColor("#E2E8F0")  # gri bar sıfır puan

        bar_cells = []
        bar_widths = []
        if bar_fill_w > 0:
            bar_cells.append("")
            bar_widths.append(bar_fill_w)
        if bar_bg_w > 0:
            bar_cells.append("")
            bar_widths.append(bar_bg_w)

        if bar_cells:
            inner = Table([bar_cells], colWidths=bar_widths, rowHeights=[10])
            inner_style = [
                ("TOPPADDING",   (0,0), (-1,-1), 0),
                ("BOTTOMPADDING",(0,0), (-1,-1), 0),
                ("LEFTPADDING",  (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ]
            if bar_fill_w > 0:
                inner_style.append(("BACKGROUND", (0,0), (0,0), bar_color))
            if bar_bg_w > 0:
                idx = 1 if bar_fill_w > 0 else 0
                inner_style.append(("BACKGROUND", (idx,0), (idx,0), colors.HexColor("#F1F5F9")))
            inner.setStyle(TableStyle(inner_style))
        else:
            inner = ""

        pts_label = f"{pts}"
        data.append([
            Paragraph(html.escape(name), cell_style),
            inner,
            Paragraph(f"<b>{pts_label}</b> / {mx}", cell_style),
        ])

    tbl = Table(data, colWidths=[name_w, bar_w, num_w], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), _C_DARK),
        ("GRID",         (0,1), (-1,-1), 0.25, _C_BORDER),
        ("BOX",          (0,0), (-1,-1), 0.5, _C_BORDER),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [_C_WHITE, _C_ZEBRA]),
    ]))
    return tbl


def _footer_block(st_styles: dict) -> list:
    return [
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.5, color=_C_BORDER, spaceBefore=2),
        Spacer(1, 3),
        Paragraph(
            "MinerWin — Bu rapor otomatik teknik analiz amacıyla üretilmiştir. "
            "Yatırım tavsiyesi niteliği taşımaz. Yatırım kararları tamamen kullanıcının "
            "kendi sorumluluğundadır.",
            st_styles["footer"],
        ),
    ]


# =========================================================
# PDF EXPORT (Tek Hisse)
# =========================================================
def build_pdf_bytes_single(
    ticker: str,
    interval_label: str,
    bars: int,
    plan: TradePlan,
    quote: dict | None,
    logo_b64_str: str = "",
    earn: dict | None = None,
    mh: dict | None = None,
    mtf: dict | None = None,
    ps: dict | None = None,
    risk_pct: float = float("nan"),
):
    fn, fn_bold = _setup_pdf_fonts()
    sty = _pdf_styles(fn, fn_bold)

    buf = io.BytesIO()
    page_w = A4[0] - 3.2*cm
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.2*cm,  bottomMargin=1.2*cm,
        title=f"MinerWin — {ticker} Analiz Raporu",
        author="MinerWin",
    )

    story = []

    # FIX (V6.2.1): datetime.utcnow() deprecated → datetime.now(timezone.utc)
    subtitle = (f"Ticker: {ticker}  |  Zaman: {interval_label}  |  "
                f"Bar: {bars}  |  "
                f"Tarih: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    story += _pdf_header_story(logo_b64_str, "MinerWin — Teknik Analiz Raporu", subtitle, sty, page_w)

    # OMURGA (V7.1): Manşet = tek anlatım. Kapı kapalıyken günlük durum
    # alt satırda dahi görünmez (günlük karar dili taşır).
    _gate = mtf.get("gate") if mtf else None
    _closed = _gate in ("RET", "BEKLEMEDE")
    # OMURGA #1 (PDF): Kapı kapalıyken rapor HAFTALIK plan üzerinden anlatılır —
    # günlük timing/skor/plan PDF'te de konuşmaz (Timing 100/100 şizofrenisi biter).
    if _closed and mtf and mtf.get("_w_plan") is not None:
        plan = mtf["_w_plan"]
    if mtf and mtf.get("verdict"):
        story.append(_status_badge(str(mtf["verdict"]), sty, page_w, label="DURUM",
                                   kind=str(mtf.get("verdict_kind", ""))))
        story.append(Spacer(1, 4))
        if _gate == "ACIK":
            sub_line = (f"Haftalık: {html.escape(_strip_emoji(str(mtf.get('w_status', '—'))))}  |  "
                        f"Günlük: {html.escape(_strip_emoji(plan.status_tag))}")
        else:
            sub_line = f"Haftalık: {html.escape(_strip_emoji(str(mtf.get('w_status', '—'))))}"
        story.append(Paragraph(sub_line, sty["small"]))
    else:
        story.append(_status_badge(plan.status_tag, sty, page_w))
    story.append(Spacer(1, 8))

    close_val = plan.debug.get("close", float("nan"))
    price_str = f"${close_val:.2f}" if np.isfinite(close_val) else "—"
    min5_str  = "GEÇTİ" if plan.minervini5_ok else "GEÇMEDİ"
    min5_clr  = _C_GREEN if plan.minervini5_ok else _C_RED

    cap_tr = {"HIGH": "YÜKSEK", "MID": "ORTA", "LOW": "DÜŞÜK"}.get(plan.capacity_level, plan.capacity_level)

    row1_items = [
        ("GÜNCEL FİYAT",  price_str),
        ("TOPLAM SKOR",   f"{plan.total_score} / 100"),
        ("KAPASİTE",      cap_tr),
    ]
    story.append(_kpi_row(row1_items, sty, page_w, accent_colors=[_C_ACCENT, _C_ACCENT, _C_ACCENT]))
    story.append(Spacer(1, 5))

    if _closed:
        row2_items = [
            ("HAFTALIK SETUP", f"{plan.setup_score} / 100"),
            ("KAPI", "KAPALI — RET" if _gate == "RET" else "KAPALI — BEKLEMEDE"),
            ("MİNERVİNİ #5",  min5_str),
        ]
        _row2_clr = [_C_ACCENT, (_C_RED if _gate == "RET" else _C_AMBER), min5_clr]
    else:
        row2_items = [
            ("SETUP (GÜNLÜK)",   f"{plan.setup_score} / 100"),
            ("TIMING (GÜNLÜK)",  f"{plan.timing_score} / 100"),
            ("MİNERVİNİ #5",  min5_str),
        ]
        _row2_clr = [_C_ACCENT, _C_ACCENT, min5_clr]
    story.append(_kpi_row(row2_items, sty, page_w, accent_colors=_row2_clr))
    story.append(Spacer(1, 5))

    # NEW (V6.3.3): Piyasa rejimi + sonraki bilanço KPI satırı
    regime_txt = _strip_emoji(str(mh.get("regime", "—"))) if mh else "—"
    if mh and mh.get("swing_ok") is True:
        regime_clr = _C_GREEN
    elif mh and mh.get("swing_ok") is False:
        regime_clr = _C_RED
    else:
        regime_clr = _C_AMBER
    if earn and earn.get("date"):
        earn_txt = f"{earn['date']} ({earn['days']} gün)"
        earn_clr = _C_RED if (earn.get("days") is not None and earn["days"] <= EARNINGS_WARN_DAYS) else _C_ACCENT
    else:
        earn_txt = "—"
        earn_clr = _C_MID
    # NEW (V7.0): Dağıtım günü sayısı — rejim kararının 'neden'i raporda görünür
    dist_val = mh.get("dist_days") if mh else None
    dist_txt = f"{dist_val} / 25" if dist_val is not None else "—"
    dist_clr = _C_RED if (isinstance(dist_val, int) and dist_val >= 6) else (_C_AMBER if (isinstance(dist_val, int) and dist_val >= 4) else _C_MID)
    row3_items = [
        ("PİYASA REJİMİ (SPY)", regime_txt),
        ("DAĞITIM GÜNÜ (25G)", dist_txt),
        ("SONRAKİ BİLANÇO", earn_txt),
    ]
    story.append(_kpi_row(row3_items, sty, page_w, accent_colors=[regime_clr, dist_clr, earn_clr]))
    story.append(Spacer(1, 6))

    if plan.high_vol_warning:
        warn_tbl = Table(
            [[Paragraph("UYARI: Yüksek volatilite — stop cap devrede, pozisyon boyutunu küçült.", sty["warn"])]],
            colWidths=[page_w],
        )
        warn_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _C_AMBER_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, _C_AMBER),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(warn_tbl)
        story.append(Spacer(1, 6))

    # FIX (V6.2.1): TP2 zemin garantisi cap'i deldiyse PDF'te de uyarı göster
    if plan.debug.get("targets_debug", {}).get("tp2_floor_override"):
        tp2_warn_tbl = Table(
            [[Paragraph("NOT: TP2, 3.5R zemin garantisiyle tarihsel tavanın üzerine yükseltildi — hedefi temkinli değerlendir.", sty["warn"])]],
            colWidths=[page_w],
        )
        tp2_warn_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _C_AMBER_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, _C_AMBER),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tp2_warn_tbl)
        story.append(Spacer(1, 6))

    # NEW (V6.3.3): Yaklaşan bilanço uyarı kutusu (≤14 gün)
    if earn and earn.get("days") is not None and earn["days"] <= EARNINGS_WARN_DAYS:
        earn_warn_tbl = Table(
            [[Paragraph(
                f"UYARI: Yaklaşan bilanço {earn['date']} ({earn['days']} gün sonra) — "
                f"gece açılan gap stop koruması tanımaz. Swing girişini buna göre planla.",
                sty["warn"],
            )]],
            colWidths=[page_w],
        )
        earn_warn_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), _C_AMBER_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, _C_AMBER),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(earn_warn_tbl)
        story.append(Spacer(1, 6))

    # V7.0: Hüküm giriş vermiyorsa plan "bugünkü referans"tır — emir değil kayıt
    _hold = bool(mtf) and mtf.get("verdict_kind") in ("warning", "error")
    # OMURGA #1 (PDF): Kapı kapalıyken İşlem Planı basılmaz — günlük giriş/stop/TP
    # karar dilidir. Yerine yalnız referans seviyeler tablosu gelir.
    if _closed:
        story += _section_header("Referans Seviyeler", sty, page_w)
        _ref_rows = [
            ["Haftalık Bant — alarm buraya kurulur",
             f"{mtf.get('w_entry_low', float('nan')):.2f} – {mtf.get('w_entry_high', float('nan')):.2f}"],
            ["52W Dip",           f"{plan.low_52w:.2f}" if np.isfinite(plan.low_52w) else "—"],
            ["52W Zirve Uzaklık", f"%{plan.dist_to_52w_high_pct:.1f}" if np.isfinite(plan.dist_to_52w_high_pct) else "—"],
        ]
        story.append(_data_table(["Parametre", "Değer"], _ref_rows, sty,
                                 [page_w*0.40, page_w*0.60]))
    else:
        story += _section_header(
            "İşlem Planı (bugünkü referans — giriş günü yeniden hesaplanır)" if _hold else "İşlem Planı",
            sty, page_w,
        )
        rr1 = f"1:{plan.rr_tp1:.2f}" if np.isfinite(plan.rr_tp1) else "—"
        rr2 = f"1:{plan.rr_tp2:.2f}" if np.isfinite(plan.rr_tp2) else "—"

        plan_left = [
            ["Giriş Bölgesi",      f"{plan.entry_low:.2f} — {plan.entry_high:.2f}"],
            ["Stop",               f"{plan.stop:.2f}"],
            ["TP1  (R/R)",         f"{plan.tp1:.2f}  ({rr1})"],
            ["TP2  (R/R)",         f"{plan.tp2:.2f}  ({rr2})"],
        ]
        # NEW (V7.0): Pozisyon boyutu — raporun cevaplamadığı son eyleme dönük soru
        if ps is not None:
            if ps.get("suppressed"):
                plan_left.append([
                    "Pozisyon Boyutu",
                    "Günlük giriş planı kapalı (haftalık kapı) — giriş günü güncel değerlerle hesaplanır",
                ])
            elif np.isfinite(ps.get("shares", float("nan"))):
                _rp = f" (hedef %{risk_pct:.2f})" if np.isfinite(risk_pct) else ""
                _teyit_note = " — teyit sonrası güncel değerlerle yenile" if _hold else ""
                plan_left.append([
                    "Pozisyon Boyutu",
                    f"{int(ps['shares'])} adet ≈ ${ps['cost']:,.0f} | risk ${ps['risk_amt']:,.0f}{_rp}{_teyit_note}",
                ])
            elif ps.get("reason") == "risk_exceeds":
                plan_left.append(["Pozisyon Boyutu",
                                  "1 adet dahi hedef risk bütçesinin üstünde (adet başına risk hedefi aşıyor)"])
        plan_right = [
            ["52W Dip",            f"{plan.low_52w:.2f}" if np.isfinite(plan.low_52w) else "—"],
            ["52W Zirve Uzaklık",  f"%{plan.dist_to_52w_high_pct:.1f}" if np.isfinite(plan.dist_to_52w_high_pct) else "—"],
            ["Dar Baz",            "Var" if plan.base_detected else "Yok"],
            ["Pivot Kırılımı",     "Var" if plan.breakout_detected else "Yok"],
        ]

        half_w = page_w * 0.48
        gap_w  = page_w * 0.04
        tbl_left  = _data_table(["Parametre", "Değer"], plan_left,  sty, [half_w*0.48, half_w*0.52])
        tbl_right = _data_table(["Parametre", "Değer"], plan_right, sty, [half_w*0.48, half_w*0.52])
        side_by_side = Table([[tbl_left, "", tbl_right]], colWidths=[half_w, gap_w, half_w])
        side_by_side.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING",   (0,0), (-1,-1), 0),
            ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ]))
        story.append(side_by_side)

    # NEW (V6.3.3): MTF Özet tablosu (haftalık + günlük)
    if mtf and not mtf.get("error") and "w_setup" in mtf:
        story += _section_header("MTF Özet (Haftalık + Günlük)", sty, page_w)
        # OMURGA #1: günlük satırlar (timing, durum, teyit bandı, Evre'nin
        # günlük referansı) yalnızca kapı AÇIKken tabloya girer.
        mtf_rows = [
            ["Haftalık Setup", f"{mtf['w_setup']} / 100"],
            ["Haftalık Durum", _strip_emoji(str(mtf["w_status"]))],
        ]
        if not _closed:
            mtf_rows += [
                ["Günlük Timing", f"{mtf['d_timing']} / 100"],
                ["Günlük Durum", _strip_emoji(str(mtf["d_status"]))],
            ]
        mtf_rows.append(["Durum", _strip_emoji(str(mtf["verdict"]))])
        if not _closed:
            mtf_rows.append(["Evre", _strip_emoji(_swing_phase(
                plan.debug.get("close", float("nan")),
                mtf.get("w_entry_low", float("nan")), mtf.get("w_entry_high", float("nan")),
                mtf.get("d_entry_low", float("nan")), mtf.get("d_entry_high", float("nan")),
                atr_pct=float(plan.debug.get("atr_pct", float("nan"))),
            )) or "—"])
        mtf_rows += [
            ["RS Rating", f"{mtf.get('rs_rating', float('nan')):.0f}" if np.isfinite(mtf.get("rs_rating", float("nan"))) else "—"],
            ["Haftalık Bant — alarm (EMA20–EMA50)", f"{mtf['w_entry_low']:.2f} – {mtf['w_entry_high']:.2f}"],
        ]
        if not _closed:
            mtf_rows.append(["Günlük Teyit Bandı", f"{mtf['d_entry_low']:.2f} – {mtf['d_entry_high']:.2f}"])
        story.append(_data_table(["Parametre", "Değer"], mtf_rows, sty,
                                 [page_w*0.42, page_w*0.58]))

    story += _section_header("Skor Dağılımı", sty, page_w)
    b = plan.breakdown
    score_items = [
        ("Trend",              b.trend_stack,        30),
        ("Fiyat / EMA150",     b.price_vs_ema150,    20),
        ("Momentum (RSI)",     b.momentum_rsi,       20),
        ("Volatilite (ATR%)",  b.volatility_atr,     15),
        ("Uzama (EMA50)",      b.extension_vs_ema50, 15),
        ("52W Zirve",          b.near_52w_high,      10),
        ("RSI Yönü",           b.rsi_direction,       5),
        ("Dar Baz (bonus)",    b.base_bonus,          7),
        ("Kırılım (bonus)",    b.breakout_bonus,      8),
    ]
    story.append(_score_bar_table(score_items, sty, page_w))

    # V7.0: Senaryo metni hükümle çelişemez — karar giriş vermiyorsa günlük
    # grafiğin "girilebilir" iması yerine hükme hizalı plan yazılır.
    if _hold:
        _whi = mtf.get("w_entry_high", float("nan")) if mtf else float("nan")
        if mtf.get("verdict_kind") == "warning":
            if np.isfinite(_whi):
                scenario_src = (
                    f"Fiyat giriş bölgesinde değil; haftalık bant referansı {_whi:.2f}. "
                    f"Fiyat banda geldiğinde analiz yenilenir: o gün haftalık kriterler korunuyor "
                    f"ve günlük teyit oluşuyorsa giriş değerlendirilir; pullback yapıyı bozmuşsa "
                    f"hisse adaylıktan çıkar."
                )
            else:
                scenario_src = "Fiyat giriş bölgesinde değil. Koşullar oluştuğunda analiz yenilenir."
        else:
            scenario_src = (
                "Hisse şu an kriterlerden geçmiyor (haftalık trend / göreli güç). "
                "Haftalık bant raporda referans olarak yer alır; yapı düzeldiğinde yeniden değerlendirilir."
            )
        story += _section_header("Plan", sty, page_w)
    else:
        scenario_src = plan.scenario
        story += _section_header("Senaryo", sty, page_w)
    scenario_clean = scenario_src.replace("**", "")
    scen_tbl = Table(
        [[Paragraph(html.escape(scenario_clean), sty["body"])]],
        colWidths=[page_w],
    )
    scen_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _C_LIGHT),
        ("BOX",           (0,0), (-1,-1), 0.4, _C_BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(scen_tbl)

    rsi_dir_clean = _strip_emoji(plan.rsi_direction_label)
    rsi_slope_str = f"{plan.rsi_slope_val:.2f}" if np.isfinite(plan.rsi_slope_val) else "—"
    story.append(Spacer(1, 6))
    story.append(_kpi_row([("RSI YÖNÜ", rsi_dir_clean), ("RSI EĞİM", rsi_slope_str)], sty, page_w))

    if quote and isinstance(quote, dict):
        story += _section_header("Quote (Anlik Fiyat)", sty, page_w)
        q_keys = ["name", "exchange", "currency", "price", "change", "percent_change", "previous_close"]
        q_body = [[k, str(quote[k])] for k in q_keys if k in quote]
        if q_body:
            story.append(_data_table(["Alan", "Değer"], q_body, sty,
                                     [page_w*0.35, page_w*0.65]))

    story += _footer_block(sty)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
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
    alarm_band: tuple | None = None,
    alarm_label: str = "ALARM (haftalık)",
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
    # NEW (V7.0): Haftalık alarm bandı ikinci gölge olarak çizilir —
    # iki bandın birbirine göre konumu (evre) tek bakışta görünür.
    if alarm_band is not None:
        try:
            ab_lo, ab_hi = float(alarm_band[0]), float(alarm_band[1])
            if np.isfinite(ab_lo) and np.isfinite(ab_hi) and ab_hi > ab_lo:
                fig.add_hrect(
                    y0=ab_lo, y1=ab_hi,
                    opacity=0.10, line_width=0,
                    fillcolor="orange",
                    annotation_text=alarm_label,
                    annotation_position="bottom left",
                )
        except Exception:
            pass
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
@st.cache_data(ttl=300, max_entries=32)
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
    # FIX (V6.2.1): Kırılım hacim teyidi için bugünü içermeyen ortalama kullanılır
    vol_ma50_prev = float(v.rolling(50).mean().shift(1).iloc[-1])
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
        # FIX (V6.2.1): vol_ma50 yerine vol_ma50_prev (bugünün hacmi eşiği şişirmesin)
        if np.isfinite(pivot) and close_ > pivot and np.isfinite(vol_ma50_prev) and vol_ma50_prev > 0:
            out["breakout_ok"] = bool(volume_today >= BREAKOUT_VOL_MULTIPLIER * vol_ma50_prev)

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
            # FIX (V6.2.1): Sadece gerçekten zarar üreten bacaklar toplanır.
            # Eskiden abs(toplam) alınıyordu: stop'u maliyet üstüne çekilmiş
            # pozisyonlarda (break-even+ stop) stop-avg_cost pozitif çıkıyor,
            # bu da "maks zarar"ı yanlış şişiriyor/azaltıyordu.
            stop_pnl = stv["Qty_n"] * (stv["Stop_n"] - stv["Avg_n"])
            loss_legs = stop_pnl[stop_pnl < 0]
            k["max_loss_stop"] = abs(float(loss_legs.sum())) if not loss_legs.empty else 0.0

    # NEW (V7.0): Portföy seviyesinde risk KPI'ları
    # Toplam açık risk: sadece gerçek risk taşıyan bacaklar (Risk $ > 0)
    if "Risk $" in df.columns:
        rk = df["Risk $"].apply(to_num)
        pos_risk = rk[np.isfinite(rk) & (rk > 0)]
        k["total_open_risk"] = float(pos_risk.sum()) if not pos_risk.empty else 0.0
    else:
        k["total_open_risk"] = np.nan
    # En büyük pozisyonun portföy değerine oranı (konsantrasyon)
    if np.isfinite(k["portfolio_value"]) and k["portfolio_value"] > 0:
        k["max_pos_pct"] = float(valid["pos_value"].max() / k["portfolio_value"] * 100.0)
    else:
        k["max_pos_pct"] = np.nan

    return k


# =========================================================
# PDF EXPORT (Portföy)
# =========================================================
def build_portfolio_pdf_bytes(
    title: str,
    out: pd.DataFrame,
    kpis: Dict[str, float],
    interval_label: str,
    bars: int,
    logo_b64_str: str = "",
    mh: Dict[str, Any] | None = None,
    account_size: float = 0.0,
) -> bytes:
    fn, fn_bold = _setup_pdf_fonts()
    st_styles = _pdf_styles(fn, fn_bold)

    buf = io.BytesIO()
    # Portföy tablosu geniş — landscape A4 kullan
    page_size = (A4[1], A4[0])  # landscape
    page_w = page_size[0] - 3.2*cm
    doc = SimpleDocTemplate(
        buf, pagesize=page_size,
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm,
        title=title, author="MinerWin",
    )
    story = []

    # FIX (V6.2.1): Rapor tarihi Türkiye saatiyle
    subtitle = (f"Zaman dilimi: {interval_label}  |  Bar: {bars}  |  "
                f"Olusturma: {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M')}")
    story += _pdf_header_story(logo_b64_str, title, subtitle, st_styles, page_w)

    pv   = kpis.get("portfolio_value", np.nan)
    pnlv = kpis.get("pnl_value", np.nan)
    pnlp = kpis.get("pnl_pct", np.nan)
    mxp  = kpis.get("max_profit_tp1", np.nan)
    mxl  = kpis.get("max_loss_stop", np.nan)

    pnl_color = _C_GREEN if (np.isfinite(pnlv) and pnlv >= 0) else _C_RED

    row1 = [
        ("PORTFÖY DEĞERİ",    fmt_money(pv)),
        ("ANLIK P&amp;L ($)", fmt_money(pnlv)),
        ("ANLIK P&amp;L (%)", fmt_pct(pnlp)),
    ]
    story.append(_kpi_row(row1, st_styles, page_w, accent_colors=[_C_ACCENT, pnl_color, pnl_color]))
    story.append(Spacer(1, 5))

    row2 = [
        ("MAKS KAR (TP1)",     fmt_money(mxp)),
        ("MAKS ZARAR (STOP)",  fmt_money(mxl)),
        ("TOPLAM AÇIK RİSK",   fmt_money(kpis.get("total_open_risk", np.nan))),
    ]
    story.append(_kpi_row(row2, st_styles, page_w, accent_colors=[_C_GREEN, _C_RED, _C_AMBER]))
    story.append(Spacer(1, 5))

    # NEW (V6.3.3 + V7.0): Piyasa rejimi + risk yüzdeleri KPI satırı
    tor = kpis.get("total_open_risk", np.nan)
    orp_txt = f"%{tor/account_size*100.0:.2f}" if (np.isfinite(tor) and account_size > 0) else "—"
    mpp = kpis.get("max_pos_pct", np.nan)
    mpp_txt = f"%{mpp:.1f}" if np.isfinite(mpp) else "—"
    if mh and mh.get("regime") and mh.get("regime") != "—":
        regime_txt = _strip_emoji(str(mh["regime"]))
        if mh.get("swing_ok") is True:
            regime_clr = _C_GREEN
        elif mh.get("swing_ok") is False:
            regime_clr = _C_RED
        else:
            regime_clr = _C_AMBER
        story.append(_kpi_row(
            [("PİYASA REJİMİ (SPY)", regime_txt),
             ("SPY KAPANIŞ", f"{mh.get('close', float('nan')):.2f}" if np.isfinite(mh.get("close", np.nan)) else "—"),
             ("AÇIK RİSK / HESAP", orp_txt),
             ("EN BÜYÜK POZİSYON", mpp_txt)],
            st_styles, page_w, accent_colors=[regime_clr, _C_ACCENT, _C_AMBER, _C_ACCENT],
        ))
    else:
        story.append(_kpi_row(
            [("AÇIK RİSK / HESAP", orp_txt), ("EN BÜYÜK POZİSYON", mpp_txt)],
            st_styles, page_w, accent_colors=[_C_AMBER, _C_ACCENT],
        ))
    story.append(Spacer(1, 6))

    story += _section_header("Pozisyonlar", st_styles, page_w)

    if out is None or out.empty:
        story.append(Paragraph("Tablo boş.", st_styles["body"]))
    else:
        preferred_cols = [
            "Ticker", "Fiyat", "Qty", "Alış Ort.", "P&L %",
            "Stop", "TP1", "TP2", "Setup", "Timing",
            "Durum", "Bilanço", "Aksiyon", "Liderlik", "RS Rating",
            "52W Zirve Uzaklık %", "Blue Sky", "RSI Yönü",
        ]
        col_map = {
            "Alış Ort.":           "Alış Ort.",
            "52W Zirve Uzaklık %": "52W Zirve Uzaklık %",
            "RSI Yönü":            "RSI Yönü",
            "Hacim Kuruması":      "Hacim Kuruması",
        }
        dfp = out.rename(columns=col_map).copy()
        cols = [c for c in preferred_cols if c in dfp.columns]
        dfp = dfp[cols]

        def cell(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "—"
            return _strip_emoji(str(v)) or str(v)

        body_rows = [[cell(row[c]) for c in dfp.columns] for _, row in dfp.iterrows()]

        pnl_col_idx = -1
        try:
            pnl_col_idx = list(dfp.columns).index("P&L %")
        except ValueError:
            pass

        w_map = {
            "Ticker": 0.07, "Fiyat": 0.07, "Qty": 0.05, "Alış Ort.": 0.08,
            "P&L %": 0.06, "Stop": 0.07, "TP1": 0.07, "TP2": 0.07,
            "Setup": 0.05, "Timing": 0.05, "Durum": 0.13, "Bilanço": 0.08, "Aksiyon": 0.08, "Liderlik": 0.06,
            "RS Rating": 0.06, "52W Zirve Uzaklık %": 0.08,
            "Blue Sky": 0.05, "RSI Yönü": 0.06,
        }
        total_ratio = sum(w_map.get(c, 0.07) for c in dfp.columns)
        col_widths  = [page_w * w_map.get(c, 0.07) / total_ratio for c in dfp.columns]

        story.append(_data_table(list(dfp.columns), body_rows, st_styles, col_widths,
                                  highlight_col=pnl_col_idx, font_size=6.5))

    story += _footer_block(st_styles)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================================================
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
    ws_sum.title = "Ozet"
    ws_pos = wb.create_sheet("Pozisyonlar")

    FONT_TITLE  = Font(name="Arial", size=16, bold=True, color="0F172A")
    FONT_SUB    = Font(name="Arial", size=10, color="64748B")
    FONT_HDR    = Font(name="Arial", size=10, bold=True, color="0F172A")
    FONT_BODY   = Font(name="Arial", size=10, color="111827")
    FONT_KPI_L  = Font(name="Arial", size=9,  bold=True, color="64748B")
    FONT_KPI_V  = Font(name="Arial", size=14, bold=True, color="0F172A")
    FONT_FOOT   = Font(name="Arial", size=8,  color="94A3B8", italic=True)

    FILL_HDR    = PatternFill("solid", fgColor="EFF6FF")
    FILL_CARD   = PatternFill("solid", fgColor="FFFFFF")
    FILL_ALT    = PatternFill("solid", fgColor="F8FAFC")

    thin  = Side(style="thin",   color="E2E8F0")
    thick = Side(style="medium", color="3B82F6")
    BORDER_CARD = Border(left=thin, right=thin, top=thin, bottom=thin)
    BORDER_HDR  = Border(left=thin, right=thin, top=thick, bottom=thick)

    ALN_C  = Alignment(horizontal="center", vertical="center")
    ALN_L  = Alignment(horizontal="left",   vertical="center")
    ALN_R  = Alignment(horizontal="right",  vertical="center")
    ALN_WL = Alignment(horizontal="left",   vertical="top", wrap_text=True)

    ws_sum["A1"] = title
    ws_sum["A1"].font = FONT_TITLE
    ws_sum.merge_cells("A1:K1")
    ws_sum["A1"].alignment = ALN_L

    # FIX (V6.2.1): Rapor tarihi Türkiye saatiyle
    ws_sum["A2"] = f"Zaman: {interval_label}  |  Bar: {bars}  |  Tarih: {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M')}"
    ws_sum["A2"].font = FONT_SUB
    ws_sum.merge_cells("A2:K2")

    ws_sum.row_dimensions[1].height = 28
    ws_sum.row_dimensions[2].height = 16
    ws_sum.row_dimensions[3].height = 10

    kpi_cards = [
        ("Portfoy Degeri ($)",    kpis.get("portfolio_value", np.nan),  "money"),
        ("Anlik P&L ($)",         kpis.get("pnl_value", np.nan),        "money"),
        ("Anlik P&L (%)",         kpis.get("pnl_pct", np.nan),          "pct"),
        ("Max Kar — TP1 ($)",     kpis.get("max_profit_tp1", np.nan),   "money"),
        ("Max Zarar — Stop ($)",  kpis.get("max_loss_stop", np.nan),    "money"),
        ("Toplam Acik Risk ($)",  kpis.get("total_open_risk", np.nan), "money"),
    ]
    card_positions = [
        ("A", "C", 4, 7), ("E", "G", 4, 7), ("I", "K", 4, 7),
        ("A", "C", 8,11), ("E", "G", 8,11), ("I", "K", 8,11),
    ]
    for (lbl, val, kind), (c1, c2, r1, r2) in zip(kpi_cards, card_positions):
        for r in range(r1, r2+1):
            for c in range(ord(c1), ord(c2)+1):
                cell = ws_sum[f"{chr(c)}{r}"]
                cell.fill   = FILL_CARD
                cell.border = BORDER_CARD

        lbl_cell = ws_sum[f"{c1}{r1}"]
        lbl_cell.value = lbl
        lbl_cell.font  = FONT_KPI_L
        lbl_cell.alignment = Alignment(horizontal="left", vertical="top")
        ws_sum.merge_cells(f"{c1}{r1}:{c2}{r1}")

        val_cell = ws_sum[f"{c1}{r1+1}"]
        if kind == "money":
            val_cell.value          = float(val) if np.isfinite(val) else ""
            val_cell.number_format  = '#,##0.00'
        elif kind == "pct":
            val_cell.value          = float(val)/100.0 if np.isfinite(val) else ""
            val_cell.number_format  = '0.00%'
        else:
            val_cell.value = str(val)
            val_cell.font  = FONT_SUB
            val_cell.alignment = ALN_WL
        if kind in ("money", "pct"):
            val_cell.font      = FONT_KPI_V
            val_cell.alignment = ALN_L
        ws_sum.merge_cells(f"{c1}{r1+1}:{c2}{r2}")

    for col, w in [("A",22),("B",16),("C",16),("D",4),
                   ("E",22),("F",16),("G",16),("H",4),
                   ("I",22),("J",16),("K",16)]:
        ws_sum.column_dimensions[col].width = w

    # FIX (V6.2.1): gereksiz f-string kaldırıldı (ws_sum[f"A13"] → ws_sum["A13"])
    ws_sum["A13"] = "MinerWin V7.0 — Otomatik teknik analiz, yatirim tavsiyesi degildir."
    ws_sum["A13"].font = FONT_FOOT
    ws_sum.merge_cells("A13:K13")

    if out is None or out.empty:
        ws_pos["A1"] = "Pozisyon tablosu bos."
        ws_pos["A1"].font = FONT_BODY
    else:
        df = out.copy()
        preferred_cols = [
            "Ticker", "Fiyat", "Qty", "Alış Ort.", "P&L %",
            "Stop", "Stop Mesafe %", "TP1", "TP1 Mesafe %", "TP2", "TP2 Mesafe %",
            "R (TP1/Stop)", "R (TP2/Stop)",
            "Setup", "Timing", "Durum", "Minervini #5", "Bilanço",
            "Liderlik", "RS Rating", "RS Yeni Zirve",
            "Endekse Üstünlük 3A", "Hacim Kuruması", "Kuruma Oranı",
            "52W Zirve Uzaklık %", "Blue Sky", "İz Süren Yapı",
            "RSI Yönü", "Yüksek Vol Uyarı",
            "Poz. Değeri", "Risk $", "Aksiyon", "Not",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        df = df[cols].copy()

        for ci, col_name in enumerate(df.columns, start=1):
            cell = ws_pos.cell(row=1, column=ci, value=col_name)
            cell.font      = FONT_HDR
            cell.fill      = FILL_HDR
            cell.border    = BORDER_HDR
            cell.alignment = ALN_C

        NUM_MONEY = {"Fiyat","Alış Ort.","Stop","TP1","TP2","Poz. Değeri","Risk $"}
        NUM_PCT   = {"P&L %","Stop Mesafe %","TP1 Mesafe %","TP2 Mesafe %"}
        NUM_RR    = {"R (TP1/Stop)","R (TP2/Stop)"}
        NUM_INT   = {"Setup","Timing"}
        WRAP_COLS = {"Not","Durum","İz Süren Yapı","Aksiyon","RSI Yönü"}
        CTR_COLS  = {"Ticker","Blue Sky","Minervini #5","RS Yeni Zirve","Yüksek Vol Uyarı"}

        for ri in range(df.shape[0]):
            row_fill = FILL_ALT if ri % 2 == 1 else PatternFill("solid", fgColor="FFFFFF")
            for ci, col_name in enumerate(df.columns, start=1):
                v    = df.iloc[ri, ci-1]
                cell = ws_pos.cell(row=2+ri, column=ci)
                cell.font   = FONT_BODY
                cell.border = BORDER_CARD
                cell.fill   = row_fill

                if col_name in NUM_MONEY:
                    try:
                        cell.value          = float(v) if v != "" else ""
                        cell.number_format  = '#,##0.00'
                        cell.alignment      = ALN_R
                    except Exception:
                        cell.value = v; cell.alignment = ALN_R
                elif col_name in NUM_PCT:
                    try:
                        vv = float(v) if v != "" else ""
                        cell.value         = vv/100.0 if vv != "" else ""
                        cell.number_format = '0.00%'
                        cell.alignment     = ALN_R
                    except Exception:
                        cell.value = v; cell.alignment = ALN_R
                elif col_name in NUM_RR:
                    try:
                        cell.value         = float(v) if v != "" else ""
                        cell.number_format = '0.00'
                        cell.alignment     = ALN_R
                    except Exception:
                        cell.value = v; cell.alignment = ALN_R
                elif col_name in NUM_INT:
                    try:
                        cell.value         = int(v) if v != "" else ""
                        cell.number_format = '0'
                        cell.alignment     = ALN_R
                    except Exception:
                        cell.value = v; cell.alignment = ALN_R
                elif col_name in WRAP_COLS:
                    cell.value = str(v) if v is not None else ""
                    cell.alignment = ALN_WL
                elif col_name in CTR_COLS:
                    cell.value = str(v) if v is not None else ""
                    cell.alignment = ALN_C
                else:
                    cell.value = str(v) if v is not None else ""
                    cell.alignment = ALN_L

        if "P&L %" in df.columns:
            pnl_ci = df.columns.tolist().index("P&L %") + 1
            col_l  = get_column_letter(pnl_ci)
            rng    = f"{col_l}2:{col_l}{df.shape[0]+1}"
            ws_pos.conditional_formatting.add(
                rng, CellIsRule(operator="greaterThan", formula=["0"],
                                font=Font(color="166534", bold=True, name="Arial", size=10)))
            ws_pos.conditional_formatting.add(
                rng, CellIsRule(operator="lessThan", formula=["0"],
                                font=Font(color="991B1B", bold=True, name="Arial", size=10)))

        ws_pos.freeze_panes = "B2"
        ws_pos.auto_filter.ref = f"A1:{get_column_letter(df.shape[1])}{df.shape[0]+1}"
        tab = XLTable(displayName="Pozisyonlar",
                      ref=ws_pos.auto_filter.ref)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False, showLastColumn=False,
            showRowStripes=False, showColumnStripes=False)
        ws_pos.add_table(tab)

        col_w_map = {
            "Ticker":18, "Fiyat":12, "Qty":10, "Alış Ort.":13,
            "P&L %":10, "Stop":12, "Stop Mesafe %":13,
            "TP1":12, "TP1 Mesafe %":13, "TP2":12, "TP2 Mesafe %":13,
            "R (TP1/Stop)":12, "R (TP2/Stop)":12,
            "Setup":10, "Timing":10,
            "Durum":26, "Minervini #5":13, "Bilanço":16, "Liderlik":14,
            "RS Rating":12, "RS Yeni Zirve":14,
            "Endekse Üstünlük 3A":18, "Hacim Kuruması":16, "Kuruma Oranı":14,
            "52W Zirve Uzaklık %":18, "Blue Sky":10, "İz Süren Yapı":28,
            "RSI Yönü":20, "Yüksek Vol Uyarı":16,
            "Poz. Değeri":14, "Risk $":12,
            "Aksiyon":14, "Not":44,
        }
        for ci, col_name in enumerate(df.columns, start=1):
            ws_pos.column_dimensions[get_column_letter(ci)].width = col_w_map.get(col_name, 14)

        ws_pos.row_dimensions[1].height = 22
        for r in range(2, df.shape[0]+2):
            ws_pos.row_dimensions[r].height = 18

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

    # NEW (V6.3): MTF ve bilanço kontrol seçenekleri
    show_mtf = st.checkbox("MTF özet (Haftalık + Günlük)", value=True)
    st.caption("Tek hisse analizinde haftalık setup + günlük timing özeti. +1-2 API çağrısı (cache'li).")
    check_earnings = st.checkbox("Bilanço (earnings) kontrolü", value=True)
    st.caption("Yaklaşan bilanço uyarısı. Önce Twelve Data, olmazsa Finnhub denenir (1 saat cache'li).")

    # NEW (V7.0): Risk Yönetimi
    st.divider()
    st.subheader("💰 Risk Yönetimi")
    account_size = st.number_input(
        "Hesap büyüklüğü ($)", min_value=0.0, value=10000.0, step=500.0, key="acct_size",
    )
    risk_pct_per_trade = st.number_input(
        "İşlem başına risk (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.25, key="risk_pct_trade",
    )
    st.caption("Pozisyon boyutu = (hesap × risk%) ÷ (giriş − stop). Hesabı 0 yaparsan hesaplayıcı kapanır.")

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

    # NEW (V7.2): Gist boot — süreçte ilk kez burada tetiklenir (cache_resource
    # sayesinde sonraki rerun'larda maliyetsiz). Durum kullanıcıya gösterilir.
    _gb = _gist_boot()
    st.caption(f"☁️ Kalıcı yedek (GitHub Gist): {_gb.get('status', '—')}")

    with st.expander("📚 Geçmiş (history.csv)"):
        hist_df = read_history_df()
        if hist_df.empty:
            st.info("history.csv yok veya boş.")
        else:
            st.dataframe(hist_df.tail(200), use_container_width=True, hide_index=True)

    # NEW (V7.1): history.csv GERİ YÜKLEME — Cloud diski sıfırlanınca
    # bilgisayarındaki yedeği geri basmak için. İndirme rutininin eşi:
    # "indir" = yedek al, burası = yedekten dön. Mükerrer satırlar ayıklanır,
    # sunucudaki mevcut yeni kayıtlarla birleştirilir (hiçbir şey ezilmez).
    up_hist = st.file_uploader(
        "🔄 history.csv geri yükle (yedekten)",
        type=["csv"],
        key="hist_restore",
        help="Daha önce indirdiğin history.csv yedeğini seç — mevcut kayıtlarla birleştirilir.",
    )
    if up_hist is not None:
        _sig = f"{up_hist.name}:{up_hist.size}"
        if st.session_state.get("__hist_restored_sig") != _sig:
            try:
                df_up = pd.read_csv(up_hist)
                if "timestamp" not in df_up.columns or "ticker" not in df_up.columns:
                    st.error("Bu dosya MinerWin history.csv'sine benzemiyor (timestamp/ticker kolonları yok).")
                else:
                    df_cur = read_history_df()
                    merged = pd.concat([df_cur, df_up], ignore_index=True) if not df_cur.empty else df_up.copy()
                    merged = merged.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
                    merged = merged.sort_values("timestamp").reset_index(drop=True)
                    merged.to_csv(HISTORY_FILE, index=False)
                    _gist_push_history()  # NEW (V7.2): geri yüklenen birikim buluta da gitsin
                    st.session_state["__hist_restored_sig"] = _sig
                    st.success(
                        f"✅ Geri yüklendi: dosyadan {len(df_up)} satır alındı, "
                        f"birleşik toplam {len(merged)} satır (mükerrerler ayıklandı)."
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Geri yükleme başarısız: {_sanitize_err(e)}")


# =========================================================
# SWING MODU — NEW V7.0
# =========================================================
def _swing_phase(price: float, w_low: float, w_high: float,
                 d_low: float, d_high: float, atr_pct: float = float("nan")) -> str:
    """NEW (V7.0): İki bandın geometrisinden trend evresini türetir.
    Dört sayının zihinde birleştirilmesi işini kullanıcıdan alır."""
    vals = [price, w_low, w_high, d_low, d_high]
    if not all(np.isfinite(v) for v in vals) or w_high <= 0 or d_high <= 0:
        return ""
    if price < w_low:
        drop = (w_low - price) / w_low * 100.0
        return f"Haftalık bandın %{drop:.1f} ALTINDA — trend hasarlı, alarm konusu değil"
    if w_low <= price <= w_high:
        # NEW (V7.2): Bant içindeyken günlük yapı fiyatın ÜSTÜNDE kırıksa
        # (sert düşüşle geliş), tarif bunu adıyla söyler. Koşul yapısal-ikilidir
        # (eşik icat edilmedi); şiddet sürekli sayıyla verilir (% + ATR-normalize).
        if np.isfinite(d_low) and d_low > price:
            gap = (d_low - price) / price * 100.0
            atr_mult = (gap / atr_pct) if (np.isfinite(atr_pct) and atr_pct > 0) else float("nan")
            atr_txt = f" (≈{atr_mult:.1f}×ATR)" if np.isfinite(atr_mult) else ""
            return (
                f"🎯 ALARM BÖLGESİNDE — günlük yapı fiyatın %{gap:.1f}{atr_txt} üstünde; "
                f"teyit için günlük taban oluşumu izlenmeli. "
                f"Okuma: ≈3×ATR altı kopma hissenin kendi dilinde ılımlıdır; üstü genelde haftalar süren onarım ister."
            )
        return "🎯 ALARM BÖLGESİNDE — şimdi günlük teyidin oluşmasını bekle"
    if d_low <= price <= d_high:
        gap = (price - w_high) / w_high * 100.0
        return f"Sığ pullback — fiyat günlük bantta tutundu (derin alarm bölgesi %{gap:.1f} aşağıda)"
    if price < d_low:
        gap = (price - w_high) / w_high * 100.0
        return f"Derin pullback SÜRÜYOR — günlük bant kırıldı, alarm bölgesine %{gap:.1f} kaldı"
    gap = (price - w_high) / w_high * 100.0
    return f"Uzamış — alarm bölgesi %{gap:.1f} aşağıda; sabır evresi"


def render_swing_mode(bars_n: int, use_quote: bool, use_earnings: bool,
                      acct_size: float = 0.0, risk_pct: float = 1.0):
    """OMURGA (V7.1):
    1) Haftalık = kapı, günlük = tetik — kapı kapalıyken günlük karar dili yok.
    2) Alarm = haftalık bant; her durumda gösterilir.
    3) Günlüğün tek görevi: alarm sonrası teyit.
    5) RET ≠ BEKLEMEDE: farklı anlatım, ikisinde de günlük kapalı.
    6) Program danışmandır: durum anlatılır, karar kullanıcıya bırakılır."""
    st.subheader("🎯 Swing Analiz")
    st.caption("Haftalık kapı → (geçtiyse) günlük tetik → bilanço notu.")

    c_in1, c_in2 = st.columns([0.68, 0.32], vertical_alignment="bottom")
    with c_in1:
        sw_ticker = st.text_input("Ticker", placeholder="Örn: NVDA, ASTS, PLTR", key="sw_ticker").strip().upper()
    with c_in2:
        sw_run = st.button("Analiz Et", type="primary", use_container_width=True, key="sw_run")

    if sw_run:
        if not sw_ticker:
            st.warning("Ticker gir.")
        else:
            with st.spinner("Haftalık + günlük analiz yapılıyor..."):
                try:
                    ddf_tmp = _fetch_daily_df(sw_ticker, 320)
                    low_52w, high_52w = compute_52w_levels(ddf_tmp, 260)
                    mtf = build_mtf_summary(sw_ticker, low_52w, high_52w)
                    if mtf.get("error"):
                        raise RuntimeError(mtf["error"])
                    try:
                        mh = market_health_pack(_fetch_spy_daily(320))
                        st.session_state["__mh"] = mh
                    except Exception:
                        mh = {}
                    earn = next_earnings_info(sw_ticker) if use_earnings else {}
                    price = float(mtf["_ddf"].iloc[-1]["close"])
                    if use_quote:
                        try:
                            q = td_quote(sw_ticker)
                            if "price" in q:
                                price = float(q["price"])
                        except Exception:
                            pass
                    st.session_state["__sw"] = {
                        "ticker": sw_ticker, "mtf": mtf, "mh": mh,
                        "earn": earn, "price": price,
                        "ts": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M"),
                    }
                    # Eski PDF'ler session'da birikmesin (bellek hijyeni)
                    for _k in [k for k in list(st.session_state.keys()) if str(k).startswith("__sw_pdf::")]:
                        st.session_state.pop(_k, None)
                    d_plan = mtf["_d_plan"]
                    record = {
                        "timestamp": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": sw_ticker,
                        "timeframe": "swing",
                        "price": round(price, 4),
                        "rule_ver": RULE_VER,
                        "gate": mtf.get("gate", ""),
                        "earnings_days": (earn.get("days", "") if (earn and not earn.get("error")) else ""),
                        "rs_rating": round(float(mtf["rs_rating"]), 1) if np.isfinite(mtf.get("rs_rating", float("nan"))) else "",
                        "regime": _strip_emoji(str(mh.get("regime", ""))) if mh else "",
                        "dist_days": mh.get("dist_days", "") if mh else "",
                        "setup_score": int(mtf["w_setup"]),
                        "timing_score": int(mtf["d_timing"]),
                        "total_score": int(d_plan.total_score),
                        "status_tag": f"[{mtf.get('gate','?')}] {mtf['verdict']}",
                        "minervini5_ok": bool(d_plan.minervini5_ok),
                        "rsi_direction": d_plan.rsi_direction_label,
                        "dist_to_52w_high_pct": round(float(d_plan.dist_to_52w_high_pct), 2) if np.isfinite(d_plan.dist_to_52w_high_pct) else "",
                        "high_vol_warning": d_plan.high_vol_warning,
                        "entry_low": round(float(mtf["w_entry_low"]), 4),
                        "entry_high": round(float(mtf["w_entry_high"]), 4),
                        "stop": round(float(d_plan.stop), 4),
                        "tp1": round(float(d_plan.tp1), 4),
                        "tp2": round(float(d_plan.tp2), 4),
                        "rr_tp1": round(float(d_plan.rr_tp1), 4) if np.isfinite(d_plan.rr_tp1) else "",
                        "rr_tp2": round(float(d_plan.rr_tp2), 4) if np.isfinite(d_plan.rr_tp2) else "",
                        "capacity": d_plan.capacity_level,
                    }
                    try:
                        save_to_history(record)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Analiz başarısız: {_sanitize_err(e)}")
                    st.session_state.pop("__sw", None)

    sw = st.session_state.get("__sw")
    if not sw:
        st.info("Ticker girip **Analiz Et** ile başla.")
        return

    t = sw["ticker"]
    mtf, mh, earn, price = sw["mtf"], sw["mh"], sw["earn"], sw["price"]
    w_plan, d_plan = mtf["_w_plan"], mtf["_d_plan"]
    gate = mtf.get("gate", "ACIK")

    st.divider()

    # ---- 0) Piyasa (bilgi) ----
    if mh:
        render_market_health(mh)
        st.divider()

    # ---- 1) DURUM — tek anlatım ----
    hc1, hc2 = st.columns([0.28, 0.72])
    hc1.metric(t, f"{price:.2f}")
    with hc2:
        box = st.success if mtf["verdict_kind"] == "success" else (
            st.warning if mtf["verdict_kind"] == "warning" else st.error)
        box(f"**DURUM:** {mtf['verdict']}")
        _wlo, _whi = mtf.get("w_entry_low", float("nan")), mtf.get("w_entry_high", float("nan"))
        if np.isfinite(_wlo) and np.isfinite(_whi) and _whi > 0:
            if price > _whi:
                st.caption(f"Fiyat haftalık bandın %{(price - _whi) / _whi * 100:.1f} üstünde.")
            elif price < _wlo:
                st.caption(f"Fiyat haftalık bandın %{(_wlo - price) / _wlo * 100:.1f} altında.")
            else:
                st.caption("Fiyat haftalık bandın içinde.")

    # ---- 2) HAFTALIK — her zaman (OMURGA #2: alarm haftalığın malı) ----
    st.markdown("#### Haftalık")
    w1, w2, w3 = st.columns(3)
    w1.metric("Setup", f"{mtf['w_setup']} / 100")
    w2.metric("Durum", mtf["w_status"])
    rsr = mtf.get("rs_rating", float("nan"))
    w3.metric("RS Rating", f"{rsr:.0f}" if np.isfinite(rsr) else "—",
              help="Endekse göre göreli güç. <45 kalite kriteri dışıdır; 45-60 bilgi notudur.")
    st.info(
        f"📐 **Haftalık bant (EMA20–EMA50): {mtf['w_entry_low']:.2f} – {mtf['w_entry_high']:.2f}** — "
        f"alarm bu banda kurulur; çaldığında analiz yenilenir."
    )
    st.caption("Bant her hafta kayar; alarmlar hafta kapanışında yenilenir.")

    # ---- 3) GÜNLÜK — yalnızca kapı AÇIKken (OMURGA #1) ----
    if gate == "ACIK":
        st.markdown("#### Günlük")
        d1, d2, d3 = st.columns(3)
        d1.metric("Timing", f"{mtf['d_timing']} / 100")
        d2.metric("Durum", mtf["d_status"])
        d3.metric("Banda Mesafe", f"{d_plan.dist_to_entry_pct:+.1f}%")
        rr1 = f"1:{d_plan.rr_tp1:.2f}" if np.isfinite(d_plan.rr_tp1) else "—"
        rr2 = f"1:{d_plan.rr_tp2:.2f}" if np.isfinite(d_plan.rr_tp2) else "—"
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Günlük Bant", f"{mtf['d_entry_low']:.2f}–{mtf['d_entry_high']:.2f}")
        p2.metric("Stop", f"{d_plan.stop:.2f}")
        p3.metric("TP1", f"{d_plan.tp1:.2f}", help=f"R/R {rr1}")
        p4.metric("TP2", f"{d_plan.tp2:.2f}", help=f"R/R {rr2}")
        if not mtf.get("daily_green"):
            st.caption("Seviyeler bugünkü değerlerdir; teyit gününde analiz yenilenip güncellenir.")
        if d_plan.high_vol_warning:
            st.caption("Not: ATR% yüksek — stop dinamik tavana dayandı; volatilite ortalamanın üstünde.")
        if d_plan.debug.get("targets_debug", {}).get("tp2_floor_override"):
            st.caption("Not: TP2, 3.5R zemini nedeniyle tarihsel tavanın üzerinde.")

        # Pozisyon boyutu — her zaman hesaplanır, bilgi dilinde (OMURGA #6)
        ps = position_size_calc(acct_size, risk_pct, d_plan.entry_mid, d_plan.stop)
        if np.isfinite(ps.get("shares", float("nan"))):
            cap_note = " · maliyet hesap sınırına çekildi" if ps.get("capped") else ""
            st.info(
                f"💰 %{risk_pct:.2f} risk kuralına göre: **{int(ps['shares'])} adet** ≈ "
                f"${ps['cost']:,.0f} | risk ${ps['risk_amt']:,.0f}{cap_note}"
            )
        elif ps.get("reason") == "risk_exceeds" and acct_size > 0:
            _pr = ps.get("per_share_risk", float("nan"))
            st.info(
                f"💰 Adet başına risk ${_pr:,.0f} = hesabın %{_pr / acct_size * 100:.2f}'i — "
                f"%{risk_pct:.2f} kuralının üstünde; 1 adet dahi kuralı aşar. Karar senin."
            )
    else:
        ps = None
        st.caption(
            "Günlük katman kapalı — haftalık kapı geçilmeden günlük tetik anlatılmaz. "
            "Alarm gününde yenilenen analizde açılır."
        )

    # ---- 4) BİLANÇO (bilgi notu) ----
    if earn.get("days") is not None and earn["days"] <= EARNINGS_WARN_DAYS:
        st.warning(f"📅 Bilanço: {earn['date']} ({earn['days']} gün) — gece gap'i stop tanımaz.")
    elif earn.get("date"):
        st.caption(f"📅 Sonraki bilanço: {earn['date']} ({earn['days']} gün, {earn.get('source', '?')}).")
    elif use_earnings and earn.get("error"):
        st.caption(f"ℹ️ Bilanço verisi alınamadı: {earn['error']}")

    st.divider()

    # ---- 5) GRAFİK — kapı kapalıyken sadece haftalık ----
    st.markdown("#### 📈 Grafik")
    if gate == "ACIK":
        sw_tf = st.radio("Grafik", ["Günlük", "Haftalık"], horizontal=True,
                         key="sw_chart_tf", label_visibility="collapsed")
        if sw_tf == "Günlük":
            fig = plot_chart(mtf["_ddf"], t, d_plan, price, show_candles, show_emas, show_line,
                             alarm_band=(mtf["w_entry_low"], mtf["w_entry_high"]),
                             alarm_label="Haftalık bant")
        else:
            w_last = float(mtf["_wdf"].iloc[-1]["close"])
            fig = plot_chart(mtf["_wdf"], t, w_plan, w_last, show_candles, show_emas, show_line)
    else:
        w_last = float(mtf["_wdf"].iloc[-1]["close"])
        fig = plot_chart(mtf["_wdf"], t, w_plan, w_last, show_candles, show_emas, show_line)
        st.caption("Kapı kapalıyken günlük grafik gösterilmez (günlük seviyeler karar dili taşır).")
    st.plotly_chart(fig, use_container_width=True)

    # ---- 6) PDF — istek üzerine üretilir (her rerun'da değil; bellek/CPU tasarrufu) ----
    _pdf_key = f"__sw_pdf::{t}::{sw.get('ts', '')}"
    if st.button("📄 Swing Raporu Hazırla (PDF)", use_container_width=True, key="sw_pdf_make"):
        st.session_state[_pdf_key] = build_pdf_bytes_single(
            ticker=t, interval_label="Swing (Haftalık kapı + Günlük tetik)", bars=bars_n,
            plan=d_plan, quote=None, logo_b64_str=logo_b64,
            earn=(earn if use_earnings else None), mh=mh, mtf=mtf,
            ps=ps, risk_pct=risk_pct,
        )
    if st.session_state.get(_pdf_key):
        st.download_button(
            "⬇️ PDF İndir", data=st.session_state[_pdf_key],
            file_name=f"{t}_swing_rapor.pdf", mime="application/pdf",
            use_container_width=True, key="sw_pdf",
        )


# =========================================================
# PİYASA SAĞLIĞI PANELİ — NEW V6.3
# =========================================================
mh_c1, mh_c2 = st.columns([0.28, 0.72])
with mh_c1:
    if st.button("📡 Piyasa Sağlığını Getir (SPY)", use_container_width=True):
        try:
            st.session_state["__mh"] = market_health_pack(_fetch_spy_daily(320))
        except Exception as e:
            st.session_state["__mh"] = {"error": str(e)}
if st.session_state.get("__mh"):
    render_market_health(st.session_state["__mh"])
st.divider()


# =========================================================
# ANA SEKMELER
# =========================================================
tab_single, tab_portfolio = st.tabs(["📈 Tek Hisse Analiz", "🧳 Portföy Analiz"])


# =========================================================
# SEKME 1: TEK HİSSE
# =========================================================
with tab_single:
    # NEW (V7.0): Görünüm modu — Swing (senin iş akışın, varsayılan) / Gelişmiş (eski ekran)
    ui_mode = st.radio(
        "Görünüm modu",
        ["🎯 Swing Modu", "🔬 Gelişmiş Mod"],
        horizontal=True,
        label_visibility="collapsed",
        key="ui_mode",
    )
    st.divider()
    if ui_mode.startswith("🎯"):
        render_swing_mode(bars, show_quote, check_earnings, account_size, risk_pct_per_trade)
    else:
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
                            st.error(f"Veri alınamadı: {_sanitize_err(e)}")

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
                            st.error(f"Daily veri / 52W hesap hatası: {_sanitize_err(e)}")
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

                        weekly_info = {}
                        try:
                            weekly_info = check_weekly_trend(ticker)
                        except Exception:
                            pass

                        # NEW (V6.3): Bilanço kontrolü
                        earn = next_earnings_info(ticker) if check_earnings else {}

                        # NEW (V6.3): Piyasa sağlığı (SPY leadership_pack ile zaten cache'te)
                        try:
                            mh = market_health_pack(_fetch_spy_daily(320))
                            st.session_state["__mh"] = mh
                        except Exception:
                            mh = {}

                        # NEW (V6.3): MTF özet (haftalık setup + günlük timing)
                        mtf = build_mtf_summary(ticker, low_52w, high_52w) if show_mtf else {}

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
                            except Exception as e:
                                # FIX (V6.2.1): Quote hatası sessizce yutulmuyor
                                q = {}
                                st.caption(f"ℹ️ Quote alınamadı, mum kapanışı kullanılıyor: {_sanitize_err(e)}")

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

                        # FIX (V6.2.1): timestamp Türkiye saatiyle
                        record = {
                            "rule_ver": RULE_VER,
                            "timestamp": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
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
                        # V7.0: TEK SES — Swing hükmü Gelişmiş modda da en üsttedir;
                        # altındaki her şey bu hükmün ham verisi olarak okunur.
                        if mtf and mtf.get("verdict"):
                            _vk_adv = mtf.get("verdict_kind")
                            if _vk_adv == "success":
                                st.success(f"**SWING KARARI:** {mtf['verdict']}")
                            elif _vk_adv == "warning":
                                st.warning(f"**SWING KARARI:** {mtf['verdict']}")
                            else:
                                st.error(f"**SWING KARARI:** {mtf['verdict']}")
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

                        # FIX (V6.2.1): TP2 zemin garantisi cap'i deldiyse UI'da uyarı
                        if plan.debug.get("targets_debug", {}).get("tp2_floor_override"):
                            st.info(
                                "ℹ️ **TP2 Notu:** TP2, 3.5R zemin garantisi nedeniyle tarihsel "
                                "cap'in (momentum/52W tavanı) üzerine yükseltildi. Bu hedefi "
                                "temkinli değerlendir — R/R oranı bu durumda kendi kendini doğrular."
                            )

                        if weekly_info.get("warning"):
                            st.warning(weekly_info["warning"])
                        # FIX (V6.2.1): Weekly kontrol hatası artık görünür
                        if weekly_info.get("error"):
                            st.caption(f"ℹ️ Weekly trend kontrolü yapılamadı: {weekly_info['error']}")

                        # NEW (V6.3): Piyasa rejimi uyarısı
                        if mh.get("swing_ok") is False:
                            st.error(f"🔴 **Piyasa Rejimi (SPY):** {mh.get('detail', '')}")
                        elif str(mh.get("regime", "")).startswith("🟡"):
                            st.warning(f"🟡 **Piyasa Rejimi (SPY):** {mh.get('detail', '')}")

                        # NEW (V6.3): Bilanço uyarısı — gece gap'i stop'a saygı duymaz
                        if earn.get("days") is not None and earn["days"] <= EARNINGS_WARN_DAYS:
                            st.error(
                                f"📅 **Yaklaşan Bilanço:** {earn['date']} ({earn['days']} gün sonra) — "
                                f"gece açılan gap stop koruması tanımaz. Swing girişini buna göre planla "
                                f"veya bilanço sonrasını bekle."
                            )
                        elif earn.get("date"):
                            st.caption(f"📅 Sonraki bilanço: {earn['date']} ({earn['days']} gün sonra) — kaynak: {earn.get('source', '?')}")
                        elif check_earnings and earn.get("error"):
                            st.caption(f"ℹ️ Bilanço verisi alınamadı (mevcut API planında olmayabilir): {earn['error']}")

                        col_baz, col_kir = st.columns(2)
                        _intraday_note = "" if interval_label == "Günlük (1day)" else " · Aktif timeframe bazlı"
                        with col_baz:
                            st.metric(
                                "Dar Baz",
                                "✅ Tespit Edildi" if plan.base_detected else "— Yok",
                                help=f"Son 20 barda ATR daralması + hacim kuruması birlikte varsa baz oluşmuştur (referans: son {BASE_REF_WINDOW} bar).{_intraday_note}"
                            )
                        with col_kir:
                            st.metric(
                                "Pivot Kırılımı",
                                "✅ Kırıldı + Hacim" if plan.breakout_detected else "— Yok",
                                help=f"Son 20 barın zirvesi kırıldı + hacim 50g ortalamasının %140 üstünde.{_intraday_note}"
                            )
                        if interval_label != "Günlük (1day)":
                            st.caption("ℹ️ Dar baz ve pivot kırılımı aktif timeframe'e göre hesaplanır — günlük değil.")

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

                        # NEW (V6.3): MTF Özet — haftalık setup + günlük timing tek ekranda
                        if show_mtf:
                            st.subheader("🧭 MTF Özet (Haftalık + Günlük)")
                            if mtf.get("error"):
                                st.caption(f"ℹ️ MTF hesaplanamadı: {mtf['error']}")
                            elif mtf:
                                cM1, cM2, cM3, cM4 = st.columns(4)
                                cM1.metric("Haftalık Setup", f"{mtf['w_setup']} / 100")
                                cM2.metric("Haftalık Durum", mtf["w_status"])
                                cM3.metric("Günlük Timing", f"{mtf['d_timing']} / 100")
                                cM4.metric("Günlük Durum", mtf["d_status"])
                                if mtf["verdict_kind"] == "success":
                                    st.success(mtf["verdict"])
                                elif mtf["verdict_kind"] == "warning":
                                    st.warning(mtf["verdict"])
                                else:
                                    st.error(mtf["verdict"])
                                st.info(
                                    f"🔔 **Bu haftanın alarm bandı (haftalık EMA20–EMA50):** "
                                    f"{mtf['w_entry_low']:.2f} – {mtf['w_entry_high']:.2f}  |  "
                                    f"**Günlük teyit bandı:** {mtf['d_entry_low']:.2f} – {mtf['d_entry_high']:.2f}"
                                )
                                st.caption(
                                    "İş akışı: haftalık alarm bandına fiyat alarmı kur → alarm çalınca "
                                    "günlük durum 🟢 ise teyitli giriş ara. Haftalık bant her hafta kayar — "
                                    "alarmları hafta kapanışında yenile."
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

                        # OMURGA (V7.1): Pozisyon her zaman hesaplanır — kapı kapalıysa
                        # gizlenmez, bilgi notuyla sunulur (madde 6: veri saklanmaz).
                        ps_adv = position_size_calc(account_size, risk_pct_per_trade, plan.entry_mid, plan.stop)
                        _gate_adv = mtf.get("gate") if (show_mtf and mtf and not mtf.get("error")) else None
                        if _gate_adv and _gate_adv != "ACIK":
                            st.caption("ℹ️ Swing kapısı kapalı — aşağıdaki hesap bilgi amaçlıdır; giriş günü güncel değerlerle yenilenir.")
                        if np.isfinite(ps_adv.get("shares", np.nan)):
                            cap_note = " · ⚠️ hesap sınırına takıldı" if ps_adv["capped"] else ""
                            st.info(
                                f"💰 **Pozisyon Boyutu:** {int(ps_adv['shares'])} adet ≈ ${ps_adv['cost']:,.0f}  |  "
                                f"Risk: ${ps_adv['risk_amt']:,.0f} (hedef %{risk_pct_per_trade:.2f}){cap_note}"
                            )
                        elif ps_adv.get("reason") == "risk_exceeds" and account_size > 0:
                            _pr = ps_adv.get("per_share_risk", float("nan"))
                            st.warning(
                                f"💰 **Alınamaz:** 1 adet bile hedef riski aşıyor — adet başına risk "
                                f"${_pr:,.0f} = hesabın %{_pr/account_size*100:.2f}'i (hedef %{risk_pct_per_trade:.2f})."
                            )

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
                        # V7.0 TEK SES: Senaryo günlük grafiğin okumasıdır — hükümle çelişemez
                        if show_mtf and mtf and not mtf.get("error") and mtf.get("verdict_kind") in ("warning", "error"):
                            st.caption(
                                "ℹ️ Not: Bu senaryo GÜNLÜK grafiğin kendi okumasıdır — "
                                "nihai hüküm yukarıdaki SWING KARARI'dır."
                            )

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
                        pdf_bytes = build_pdf_bytes_single(ticker=ticker, interval_label=interval_label, bars=bars, plan=plan, quote=(q if show_quote else None), logo_b64_str=logo_b64, earn=(earn if check_earnings else None), mh=mh, mtf=(mtf if show_mtf else None), ps=ps_adv, risk_pct=risk_pct_per_trade)
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

        # FIX (V6.2.1): Streamlit Cloud disk uyarısı — ortak ve geçici depolama
        st.caption(
            "⚠️ **Önemli:** portfolio.csv ve history.csv sunucu diskine yazılır. "
            "Streamlit Cloud'da bu dosyalar **uygulamayı açan herkesle ortaktır** ve "
            "her yeniden başlatma/deploy'da **silinir**. Verini düzenli olarak indir."
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
        # FIX (V6.2.1): data_editor ayrı widget key ile kullanılır ve sonucu
        # state'e atanır — session_state key'i ile widget key'inin çakışmasından
        # doğan "düzenleme kayboluyor" rerun sorunlarına karşı daha sağlam pattern.
        edited_pf = st.data_editor(
            st.session_state.portfolio,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="pf_editor",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "qty": st.column_config.NumberColumn("Adet", min_value=0.0, step=1.0),
                "avg_cost": st.column_config.NumberColumn("Alış Ort.", min_value=0.0, step=0.01, format="%.2f"),
                "stop": st.column_config.NumberColumn("Stop", min_value=0.0, step=0.01, format="%.2f"),
                "tp1": st.column_config.NumberColumn("TP1", min_value=0.0, step=0.01, format="%.2f"),
                "tp2": st.column_config.NumberColumn("TP2", min_value=0.0, step=0.01, format="%.2f"),
            },
        )
        st.session_state.portfolio = edited_pf

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

                with st.spinner("Portföy verileri çekiliyor... (rate limit'e takılırsa otomatik bekler)"):
                    spy_df_shared = pd.DataFrame()
                    try:
                        spy_df_shared = _fetch_spy_daily(320)
                    except Exception:
                        pass

                    # NEW (V6.3): Piyasa sağlığı — SPY zaten elimizde, ekstra çağrı yok
                    mh_pf = market_health_pack(spy_df_shared) if not spy_df_shared.empty else {}
                    if mh_pf:
                        st.session_state["__mh"] = mh_pf

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

                            # NEW (V6.3): Bilanço kontrolü (+1 API çağrısı/ticker, 1 saat cache'li)
                            earn_str = ""
                            if check_earnings:
                                _earn = next_earnings_info(tkr)
                                if _earn.get("days") is not None and _earn["days"] <= EARNINGS_WARN_DAYS:
                                    earn_str = f"⚠️ {_earn['date']} ({_earn['days']}g)"
                                elif _earn.get("date"):
                                    earn_str = str(_earn["date"])

                            pos_value = (qty * price) if np.isfinite(qty) and np.isfinite(price) else np.nan

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
                                "Bilanço": earn_str,
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
                                "Aksiyon": "HATA", "Not": f"Veri/analiz hatası: {_sanitize_err(e)}",
                                "52W High": "", "Blue Sky": "", "İz Süren Yapı": "",
                            })

                out = pd.DataFrame(rows)

                # NEW (V6.3): Piyasa rejimi — sonuçların en üstünde
                if mh_pf:
                    st.markdown("### 📡 Piyasa Rejimi")
                    render_market_health(mh_pf)
                    if mh_pf.get("swing_ok") is False:
                        st.error(
                            "🔴 Piyasa rejimi zayıf — bu ortamda yeni swing alımı önerilmez; "
                            "mevcut pozisyonlarda stop disiplini öncelik."
                        )

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

                # NEW (V7.0): Risk yönetimi KPI satırı
                r2c1, r2c2, r2c3 = st.columns(3)
                tor = kpis.get("total_open_risk", np.nan)
                r2c1.metric("Toplam Açık Risk", fmt_money(tor),
                            help="Sadece gerçek risk taşıyan bacaklar: maliyet altındaki stoplardan gelen olası zarar toplamı.")
                if np.isfinite(tor) and account_size > 0:
                    _orp = tor / account_size * 100.0
                    r2c2.metric("Açık Risk / Hesap", f"%{_orp:.2f}",
                                help=f"Hesap büyüklüğü (${account_size:,.0f}) sidebar'dan.")
                    if _orp > 6.0:
                        st.warning(f"⚠️ Toplam açık risk hesabın %{_orp:.1f}'i — tüm stoplar aynı günde çalışırsa kayıp bu. %6 üstü agresif; pozisyon boylarını gözden geçir.")
                else:
                    r2c2.metric("Açık Risk / Hesap", "—", help="Sidebar'dan hesap büyüklüğü gir.")
                mpp = kpis.get("max_pos_pct", np.nan)
                r2c3.metric("En Büyük Pozisyon", f"%{mpp:.1f}" if np.isfinite(mpp) else "—",
                            help="Portföy değerine oranı — konsantrasyon göstergesi.")
                # FIX (V6.2.1): max_loss_stop artık sadece zarar üreten bacakları içerir
                st.caption(
                    "ℹ️ 'Stop Hepsi Olursa' yalnızca maliyet altındaki stoplardan gelen "
                    "gerçek zararı gösterir; break-even üstüne çekilmiş stoplar dahil edilmez."
                )

                st.markdown("### ⬇️ İndir")
                title = "MinerWin – Portföy Analizi V7.0"
                pdf_bytes = build_portfolio_pdf_bytes(title=title, out=out, kpis=kpis, interval_label=interval_label_pf, bars=bars, logo_b64_str=logo_b64, mh=mh_pf, account_size=account_size)
                xls_bytes = build_portfolio_excel_bytes(title=title, out=out, kpis=kpis, interval_label=interval_label_pf, bars=bars)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button("📄 Portföy Raporu (PDF) indir", data=pdf_bytes, file_name=f"MinerWin_Portfoy_{datetime.now(TR_TZ).strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)
                with d2:
                    st.download_button("📊 Portföy Raporu (Excel) indir", data=xls_bytes, file_name=f"MinerWin_Portfoy_{datetime.now(TR_TZ).strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

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

                # NEW (V6.3): Yaklaşan bilanço toplu uyarısı
                if check_earnings and not out.empty and "Bilanço" in out.columns:
                    earn_warn_tickers = out[out["Bilanço"].astype(str).str.startswith("⚠️")]["Ticker"].tolist()
                    if earn_warn_tickers:
                        st.error(
                            f"📅 Yaklaşan bilanço ({EARNINGS_WARN_DAYS} gün içinde): "
                            f"**{', '.join(earn_warn_tickers)}** — gece gap riski, stop koruma sağlamaz!"
                        )
