# app.py
# MinerWin — Tek Hisse + Portföy Analiz (V7.0) — Twelve Data + Finnhub
#
# V7.3 Değişiklikleri (TEYİT REFORMU — RULE_VER v2):
#  ★ Günlük teyit tanımı değişti: "fiyat bant içinde" (v1) → "kapanış EMA20
#    üstü + RSI14>50 + hacim ×1.2" (v3). Gerekçe: 30 hisse / 3 yıl / 2.284
#    bağımsız işlemlik retro-test — v1 dört yılın dördünde negatif medyan,
#    beklenti +2.49; v3 beklenti +5.99, isabet %54, K/Z 1.87, her rejimde pozitif.
#  ★ Eski tanım artık GÖLGE olarak kaydedilir (teyit_v1_eski) — karnede
#    "eski kural ne derdi?" karşılaştırması yapılabilsin.
#  ★ Kapı / RS vetosu / setup eşiği DEĞİŞMEDİ: retro-test edge'in seçim
#    katmanında olduğunu gösterdi (nötr hisselerde tüm tanımlar negatif).
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
import json
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
from typing import Any, Dict, List, Optional, Tuple

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
<div class="sub-title">Minervini-Based Technical Trading Engine — V7.3</div>
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
STOP_TIGHTEN = 0.75      # kalibrasyon: stop mesafesi ×0.75 (125 hisse/3 yıl simülasyonu)
MAX_HOLD_DAYS = 40       # kalibrasyon: pozisyon en fazla 40 işlem günü taşınır
# Minervini ölçüsü: tek işlemde kayıp %10'u geçmemeli (ideali %5-8).
# Retro defterinde risk medyanı %8.5-9.7, uçlar %16'ya kadar çıkıyordu —
# sınır kırılım anında uygulanmadığı için. Artık hem taban hem kırılımda geçerli.
MAX_BASE_RISK = 0.10   # taban dibi girişten en fazla %12 uzakta (VCP darlığı)
def _daily_chase_check(ddf: pd.DataFrame, d_plan) -> Dict[str, Any]:
    """Günlük tarafta kovalama freni. Döner: ok (teyit geçerli mi) + gerekçe."""
    out = {"ok": True, "sebep": "", "risk_pct": float("nan"),
           "bant_uzaklik": float("nan"), "limit": float("nan")}
    try:
        px = float(ddf["close"].iloc[-1])
        # (1) Risk tavanı
        stop = float(d_plan.stop)
        if px > 0 and np.isfinite(stop) and stop > 0:
            out["risk_pct"] = (px - stop) / px * 100.0
            if out["risk_pct"] > MAX_ENTRY_RISK_PCT:
                out["ok"] = False
                out["sebep"] = (f"giriş-stop mesafesi %{out['risk_pct']:.1f} — "
                                f"üst sınır %{MAX_ENTRY_RISK_PCT:.0f}; bu noktadan alım "
                                f"riski taşınamaz hale getirir")
                return out
        # (2) Bant mesafesi
        hi = float(max(d_plan.entry_low, d_plan.entry_high))
        if np.isfinite(hi) and hi > 0 and px > hi:
            out["bant_uzaklik"] = (px / hi - 1.0) * 100.0
            _tr = pd.concat([ddf["high"] - ddf["low"],
                             (ddf["high"] - ddf["close"].shift(1)).abs(),
                             (ddf["low"] - ddf["close"].shift(1)).abs()], axis=1).max(axis=1)
            _atrp = float((_tr.rolling(14).mean() / ddf["close"]).iloc[-1] * 100.0)
            limit = min(DAILY_CHASE_CAP_PCT, 2.0 * _atrp) if np.isfinite(_atrp) and _atrp > 0 \
                else DAILY_CHASE_CAP_PCT
            out["limit"] = limit
            if out["bant_uzaklik"] > limit:
                out["ok"] = False
                out["sebep"] = (f"fiyat günlük bandın %{out['bant_uzaklik']:.1f} üstünde — "
                                f"bu hissede kovalama sınırı %{limit:.1f}")
        else:
            out["bant_uzaklik"] = 0.0
    except Exception:
        pass
    return out


def rs_sirali_puan(ham_guc: Dict[str, float]) -> Dict[str, int]:
    """Taranan EVREN İÇİNDE yüzdelik RS sırası (IBD mantığı).

    Mevcut rs_rating hissenin yalnızca SPY'a göre farkını ölçer ve kelepçe
    yüzünden 20-80 arasında sıkışır — "piyasanın en iyi %20'si" anlamı taşımaz.
    Bu fonksiyon taramada biriken ham güç değerlerini sıralayıp 1-99 arası
    yüzdelik verir: 99 = evrenin en güçlüsü. Minervini'nin eşikleri (80+)
    ancak bu cetvelde anlamlıdır. KARAR ETKİLEMEZ — bilgi katmanıdır.
    """
    gecerli = {k: float(v) for k, v in ham_guc.items()
               if v is not None and np.isfinite(float(v))}
    if len(gecerli) < 5:
        return {}
    s = pd.Series(gecerli).rank(pct=True) * 98 + 1
    return {k: int(round(v)) for k, v in s.items()}


def _gun_ifadesi(n) -> str:
    """0 → 'bugün', 1 → 'dün', 2+ → 'N gün önce'. Rapor dilinde sayı değil, insan dili."""
    try:
        n = int(n)
    except Exception:
        return "yakın zamanda"
    return "bugün" if n <= 0 else ("dün" if n == 1 else f"{n} gün önce")


ARMED_DAYS = 14          # kapı açıldıktan sonra setup kaç gün "kurulu" kalır
# Kovalama toleransı ARTIK SABİT DEĞİL: hissenin kendi volatilitesine göre.
# Gerekçe: %8, günlük ATR'si %1 olan KO için çok geniş (8 günlük hareket),
# ATR'si %4 olan CRWV için çok dar (2 günlük hareket). Ölçü: 2×ATR%,
# %4 ile %12 arasına sıkıştırılır.
# GÜNLÜK KOVALAMA FRENİ (V7.5) — Minervini mantığı:
#   (1) RİSK TAVANI: giriş-stop mesafesi %MAX_ENTRY_RISK'i geçerse işlem geçersiz.
#       Kovalayan giriş stopu uzatır → risk şişer → sistem kendiliğinden reddeder.
#       Volatiliteye göre otomatik ayarlanır (stop zaten ATR'den türüyor).
#   (2) BANT MESAFESİ: kapanış, günlük EMA20-50 bandının üstünden en fazla
#       min(2×ATR%, %5) uzakta. Minervini'nin "pivottan %5'ten fazla yukarıda alma"
#       kuralının bant karşılığı; aşırı volatil hissede sert tavan görevi görür.
MAX_ENTRY_RISK_PCT = 10.0
DAILY_CHASE_CAP_PCT = 5.0

BAND_TOL_ATR_CARPAN = 2.0
BAND_TOL_MIN = 4.0
BAND_TOL_MAX = 12.0


def _band_tolerance_pct(ddf: pd.DataFrame) -> float:
    """Hisseye özel kovalama toleransı (%). ATR yoksa %8'e düşer."""
    try:
        _tr = pd.concat([ddf["high"] - ddf["low"],
                         (ddf["high"] - ddf["close"].shift(1)).abs(),
                         (ddf["low"] - ddf["close"].shift(1)).abs()], axis=1).max(axis=1)
        _atrp = float((_tr.rolling(14).mean() / ddf["close"]).iloc[-1] * 100.0)
        if not np.isfinite(_atrp) or _atrp <= 0:
            return 8.0
        return float(min(BAND_TOL_MAX, max(BAND_TOL_MIN, BAND_TOL_ATR_CARPAN * _atrp)))
    except Exception:
        return 8.0
RULE_VER = "v5"   # v5: ANA TETİK = VCP tabanı + pivot kırılımı (retro: PF 3.35 vs 1.34)

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
@st.cache_data(ttl=7200, max_entries=64)
def _fetch_daily_df(symbol: str, outputsize: int = 320) -> pd.DataFrame:
    payload = td_time_series(symbol, "1day", int(outputsize))
    return parse_ohlcv(payload)


# =========================================================
# HAZIR TARAMA EVRENLERİ (V7.6)
# =========================================================
EVREN_MINERVINI = ",".join([
    # Momentum / lider adayları — sistemin asıl av sahası
    "NVDA","AVGO","AMD","MU","MRVL","CRDO","ALAB","NVMI","ONTO","ACLS","ACMR","ARM","TSM","ASML","AMAT",
    "LRCX","KLAC","TER","ENTG","MPWR","SWKS","QRVO","NXPI","MCHP","ON","SMCI","ANET","PSTG","STX","SNDK",
    "WDC","LITE","FN","CIEN","COHR","AEIS","VRT","ETN","PWR","GEV","VST","TLN","CEG","NRG","OKLO","SMR",
    "LEU","CCJ","UEC","DNN","UUUU","NNE","BE","PLUG","FCEL","EOSE","ENPH","RUN","SHLS","ARRY","FSLR",
    "PLTR","CRWD","PANW","ZS","S","OKTA","NET","DDOG","SNOW","MDB","ESTC","GTLB","TEAM","NOW","CRM",
    "APP","TTD","ROKU","SPOT","NFLX","RBLX","U","DKNG","FLUT","SHOP","SQ","PYPL","AFRM","SOFI","UPST",
    "HOOD","COIN","NU","TOST","LMND","CVNA","IREN","RIOT","MARA","CLSK","HUT","BITF","CIFR","WULF","APLD",
    "CORZ","IONQ","RGTI","QBTS","BBAI","SOUN","AI","PATH","TEM","RXRX","CRSP","NTLA","BEAM","SRPT","ALNY",
    "IONS","VKTX","EXAS","GH","NTRA","HIMS","OSCR","DOCS","ISRG","VRTX","REGN","BSX","AXON","KTOS","AVAV",
    "RKLB","ASTS","LUNR","PL","ACHR","JOBY","EH","BLDE","ONDS","UMAC","BWXT","HWM","TDG","LDOS","PSN",
    "CACI","SNPS","CDNS","ADBE","ORCL","MSFT","GOOGL","AMZN","META","AAPL","UBER","ABNB","DASH","BKNG",
    "CELH","DUOL","ANF","DECK","ONON","BIRK","CAVA","WING","SG","CMG","SBUX","NKE","LULU","TJX","ROST",
    "COST","WMT","TGT","HD","LOW","DELL","HPE","NTAP","JBL","FLEX","SANM","ZBRA","KEYS","TDY","GRMN",
    "GLW","APH","TEL","ADI","TXN","QCOM","INTC","ARQQ","LAES","GLXY","SEZL","DAVE","OS","BMNR","CRCL",
    "SPCE","RDW","NVAX","MRNA","APLS","VERV","TERN","ALT","CLOV","ALHC","HUMA","PGNY","AMPL","FROG","DOCN",
    "TWLO","ZM","DOCU","HUBS","VEEV","WDAY","SPGI","ICE","CME","MCO","KKR","APO","ARES","BX","TW","NDAQ",
    "MSCI","FICO","VRSK","INFO","CPRT","ODFL","SAIA","XPO","URI","LII","HUBB","EMR","PH","ITW","ROK",
])

EVREN_GENIS = ",".join([
    # Geniş piyasa — mega-cap ve savunmacı sektörler (kıyas evreni)
    "JPM","BAC","GS","MS","WFC","C","SCHW","BLK","AXP","V","MA","COF","CB","PGR","TRV","ALL","MET",
    "PRU","AIG","AFL","USB","PNC","TFC","FITB","KEY","RF","CFG","HBAN","MTB","STT","BK","NTRS","AMP",
    "LLY","UNH","JNJ","ABBV","MRK","PFE","TMO","ABT","DHR","AMGN","GILD","BIIB","MDT","SYK","ZTS","BDX",
    "BAX","EW","HOLX","RMD","STE","WST","DGX","LH","CI","ELV","CNC","MOH","HCA","UHS","MCK","COR","CAH",
    "CAT","DE","RTX","LMT","GE","HON","UNP","UPS","FDX","BA","NOC","GD","LHX","TXT","CSX","NSC","ODFL",
    "WM","RSG","JCI","CARR","OTIS","IR","DOV","SWK","PNR","AME","FTV","XYL","GGG","NDSN","IEX",
    "XOM","CVX","COP","SLB","EOG","OXY","PSX","MPC","VLO","HES","DVN","FANG","HAL","BKR","OKE","WMB",
    "KMI","TRGP","LNG","EQT","AR","CTRA","MRO","APA","FCX","NEM","LIN","APD","SHW","ECL","NUE","STLD",
    "PG","KO","PEP","MDLZ","CL","KMB","GIS","K","HSY","SYY","KR","DG","DLTR","MCD","YUM","QSR","DPZ",
    "DIS","CMCSA","VZ","T","TMUS","CHTR","EA","TTWO","WBD","PARA","FOXA","NWSA","OMC","IPG",
    "NEE","DUK","SO","D","EXC","AEP","XEL","ED","WEC","ES","PEG","SRE","PCG","FE","ETR","AEE","CMS",
    "PLD","AMT","EQIX","SPG","O","CCI","PSA","WELL","VTR","AVB","EQR","MAA","ESS","UDR","INVH","ARE",
    "DLR","IRM","VICI","GLPI","HST","RHP","KIM","REG","FRT","BXP","VNO","SLG","HIW",
    "ADP","PAYX","INTU","FIS","FISV","GPN","JKHY","BR","CTAS","FAST","GWW","WSO","POOL","SITE",
    "MMM","GLW","EMN","DD","PPG","IFF","ALB","CE","LYB","MOS","CF","ADM","BG","TSN","HRL","CAG",
    "SJM","CPB","MKC","CHD","CLX","EL","COTY","ULTA","BBY","AZO","ORLY","GPC","LKQ","TSCO",
])


def _weekly_from_daily(ddf: pd.DataFrame) -> pd.DataFrame:
    """Haftalık mumları GÜNLÜK veriden türetir (API çağrısı yok).
    Twelve Data ücretsiz planı 8 çağrı/dk; hisse başına ayrı haftalık çekmek
    tarama hızını yarıya düşürüyordu. Haftalık = W-FRI toplaması.
    Son hafta TAMAMLANMAMIŞ olabilir — çağıran taraf gerektiğinde atar."""
    d = ddf.copy()
    d["_t"] = pd.to_datetime(d["time"])
    d = d.set_index("_t").sort_index()
    w = pd.DataFrame({
        "open": d["open"].resample("W-FRI").first(),
        "high": d["high"].resample("W-FRI").max(),
        "low": d["low"].resample("W-FRI").min(),
        "close": d["close"].resample("W-FRI").last(),
        "volume": d["volume"].resample("W-FRI").sum(),
    }).dropna()
    w = w.reset_index().rename(columns={"_t": "time"})
    return w


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


ENTRY_EARNINGS_BLOCK_DAYS = 5   # bilanço bu kadar gün içindeyse yeni giriş verilmez


def build_mtf_summary(symbol: str, low_52w: float, high_52w: float,
                      earnings_days: float = float("nan")) -> Dict[str, Any]:
    """Haftalık setup + günlük timing'i tek pakette döndürür.
    Kullanıcının iş akışı: haftalık giriş bandına alarm kur → alarm çalınca
    günlük ile teyit et. Bu özet iki adımı tek ekranda birleştirir."""
    out = {"error": ""}
    try:
        # TEK ÇAĞRI: geniş günlük veri hem günlük hem haftalık için kullanılır.
        # (Haftalık EMA150/200 için ~1000 günlük bar gerekir → 200 haftalık bar.)
        _wide = _fetch_daily_df(symbol, 1000)
        wdf = _add_indicators(_weekly_from_daily(_wide))
        w_plan = build_trade_plan(wdf, low_52w=low_52w, high_52w=high_52w)

        ddf = _add_indicators(_wide.tail(320).reset_index(drop=True))
        d_plan = build_trade_plan(ddf, low_52w=low_52w, high_52w=high_52w)

        # NEW (V7.0): RS Rating hesaplanır ve KARARA BAĞLANIR.
        # Minervini prensibi: endekse karşı zayıf hisse lider değildir —
        # teknik görünüm ne olursa olsun aday bile olamaz.
        rs_rating = float("nan")
        rs_edges = {}
        try:
            spy_df = _fetch_spy_daily(320)
            rs = analyze_relative_strength(ddf, spy_df)
            rs_rating = float(rs.get("rs_rating", float("nan")))
            # NEW (V7.2): GÖLGE ÖLÇÜM — ham SPY farkları (kelepçesiz) kayda
            # akar; hükme DOKUNMAZ. CP-3'te alternatif RS cetvelleri bu ham
            # veriyle simüle edilecek (retro API maliyeti olmadan).
            rs_edges = {k: rs.get(k, float("nan")) for k in ("edge_3m", "edge_6m", "edge_12m")}
        except Exception:
            pass

        # NEW (V7.2): GÖLGE TEYİT (v2-test) — HÜKÜMSÜZ. "Parasız almış gibi"
        # kayıt: aday tanım kimlere ✓ derse CP-3'te akıbetleri yargılanır.
        # Tanım: kapanış üst üste 2 gün günlük EMA20 üstünde + RSI yönü yukarı.
        teyit_v2 = ""
        try:
            if ddf is not None and len(ddf) >= 3 and "ema20" in ddf.columns:
                _c = ddf["close"].astype(float)
                _e = ddf["ema20"].astype(float)
                _two_up = bool(_c.iloc[-1] > _e.iloc[-1] and _c.iloc[-2] > _e.iloc[-2])
                _rsi_up = str(d_plan.rsi_direction_label).startswith("Yükseliyor")
                teyit_v2 = bool(_two_up and _rsi_up)
        except Exception:
            teyit_v2 = ""

        weekly_ok = (w_plan.setup_score >= 60) and (not w_plan.status_tag.startswith(("🔴", "🟣")))

        # ============ TEYİT + KAPI REFORMU — V7.3 / RULE_VER v2 ============
        # SORUN (ölçüldü): Eski kurgu kapı ve tetiği AYNI GÜNE bağlıyordu —
        # "fiyat haftalık bantta" VE "fiyat günlük bantta" aynı anda. 3 yıllık
        # retro-testte 30 hissede yalnız 39 tam yeşil (adayların %0.3'ü).
        # ÇÖZÜM: kapı-tetik SIRALI çalışır (Minervini akışı):
        #   1) Fiyat haftalık banda iner  → setup KURULUR (14 gün geçerli)
        #   2) Sonraki günlerde günlük teyit gelirse → tam yeşil (fiyat bandın
        #      biraz üstünde olsa da olur; %8'e kadar tolerans)
        # RETRO KANITI (kurulu pencere + v2 teyit): 49 tam yeşil, isabet %84,
        # medyan +11.07 rel, K/Z 5.54. Eşzamanlı kurgu: 24 sinyal, +3.02, %67.
        # v3 (hacim şartlı) kapıyla geometrik olarak çelişiyor → reddedildi (12 sinyal).
        # Teyit (v2): kapanış üst üste 2 gün günlük EMA20 üstünde + RSI yükseliyor
        daily_green = False
        teyit_eksik = ""      # teyit yoksa HANGİ şart tutmadı
        try:
            _c = ddf["close"].astype(float)
            _e20 = ddf["ema20"].astype(float)
            _r14 = ddf["rsi14"].astype(float)
            _bugun_ok = float(_c.iloc[-1]) > float(_e20.iloc[-1])
            _dun_ok = float(_c.iloc[-2]) > float(_e20.iloc[-2])
            _rsi_ok = float(_r14.iloc[-1]) > float(_r14.iloc[-4])
            daily_green = bool(_bugun_ok and _dun_ok and _rsi_ok)
            if not daily_green:
                _eksikler = []
                if not _bugun_ok:
                    _eksikler.append(f"kapanışın günlük EMA20 üstüne çıkması ({float(_e20.iloc[-1]):.2f})")
                elif not _dun_ok:
                    _eksikler.append("EMA20 üstünde ikinci kapanış (bugün birincisi)")
                if not _rsi_ok:
                    _eksikler.append("RSI'ın yukarı dönmesi")
                teyit_eksik = " ve ".join(_eksikler)
        except Exception:
            daily_green = d_plan.status_tag.startswith("🟢")

        _band_tol = _band_tolerance_pct(ddf)   # haftalık banda özel kovalama sınırı (%)

        # GÜNLÜK KOVALAMA FRENİ: teyit oluşsa bile giriş noktası uzaksa geçersiz
        _chase = _daily_chase_check(ddf, d_plan)
        if daily_green and not _chase["ok"]:
            daily_green = False

        # KURULU PENCERE: son 14 günde fiyat haftalık banda dokundu mu?
        armed_recent = False
        armed_days_ago = None
        try:
            _wlo_b, _whi_b = float(w_plan.entry_low), float(w_plan.entry_high)
            if np.isfinite(_wlo_b) and np.isfinite(_whi_b):
                _tail = ddf.tail(ARMED_DAYS)
                _touch = (_tail["low"].astype(float) <= _whi_b) & (_tail["high"].astype(float) >= _wlo_b)
                armed_recent = bool(_touch.any())
                if armed_recent:
                    armed_days_ago = int(len(_tail) - 1 - int(np.where(_touch.values)[0][-1]))
        except Exception:
            pass

        # Eski (eşzamanlı) tanım gölgede kalır — karnede karşılaştırma için
        teyit_v1_shadow = bool(d_plan.status_tag.startswith("🟢"))

        # ===== MINERVINI GİRİŞİ (V7.4 / RULE_VER v3) =====
        # Alım noktası artık VCP tabanı + PİVOT KIRILIMI. Stop YAPISAL: taban dibi.
        # Retro kanıtı (3 yıl / 30 hisse, eşit ayarlarla):
        #   eski giriş  → isabet %29, stopla kapanan %66, en büyük düşüş -%13.5
        #   pivot girişi→ isabet %53, stopla kapanan %35, en büyük düşüş  -%3.0
        mv_break = False        # bugün pivot kırıldı mı
        mv_base = False         # taban kurulu mu (daralma + hacim kuruması)
        mv_pivot = float("nan")
        mv_dip = float("nan")
        mv_stop = mv_tp1 = mv_tp2 = float("nan")
        mv_dist = float("nan")  # pivota uzaklık %
        mv_dalga = 0
        mv_son_daralma = float("nan")
        try:
            _vc = detect_vcp(ddf)
            mv_break = bool(_vc.get("kirilim"))
            mv_pivot = float(_vc.get("pivot", float("nan")))
            mv_dip = float(_vc.get("dip", float("nan")))
            mv_dalga = int(_vc.get("dalga", 0))
            mv_son_daralma = float(_vc.get("son_daralma", float("nan")))
            _px = float(ddf["close"].iloc[-1])
            _der = (mv_pivot - mv_dip) / mv_pivot if mv_pivot > 0 else float("nan")
            # Taban darlığı şartı: dip girişten %MAX_BASE_RISK'ten uzaksa geçersiz
            _risk_ok = bool(np.isfinite(mv_dip) and _px > 0 and (_px - mv_dip * 0.99) / _px <= MAX_BASE_RISK)
            mv_base = bool(_vc.get("var") and _risk_ok)
            if np.isfinite(mv_pivot) and mv_pivot > 0:
                mv_dist = (mv_pivot / _px - 1.0) * 100.0
            if np.isfinite(mv_dip) and mv_dip > 0:
                mv_stop = mv_dip * 0.99
                _R = _px - mv_stop
                if _R > 0:
                    mv_tp1, mv_tp2 = _px + 2.0 * _R, _px + 4.0 * _R
        except Exception:
            pass
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
        elif (mv_break and np.isfinite(earnings_days)
              and 0 <= float(earnings_days) <= ENTRY_EARNINGS_BLOCK_DAYS):
            # BİLANÇO FRENİ: kırılım oluştu ama gece gap'i stopu tanımaz → giriş ertelenir
            gate = "BEKLEMEDE"
            verdict = (f"Pivot kırıldı ancak bilanço {int(earnings_days)} gün içinde — "
                       f"gece açılan gap stop korumasını geçersiz kılar, giriş ertelendi. "
                       f"Bilanço sonrası yapı korunuyorsa yeniden değerlendirilir.{rs_note}")
            verdict_kind = "warning"
        elif (mv_break and np.isfinite(mv_stop) and np.isfinite(mv_pivot)
              and (float(ddf["close"].iloc[-1]) - mv_stop) > 0
              and ((float(ddf["close"].iloc[-1]) - mv_stop) / float(ddf["close"].iloc[-1])
                   <= MAX_BASE_RISK)):
            # ===== V7.6 ANA TETİK: VCP PİVOT KIRILIMI =====
            # Retro kanıtı (247 hisse / 3 yıl, gerçek trader kurallarıyla):
            #   mega-cap evreni:  v6 +888$ (PF 1.15) | v8 +3.595$ (PF 2.40)
            #   momentum evreni:  v6 +2.212$ (PF 1.34) | v8 +6.213$ (PF 3.35)
            # v8 her iki evrende, her iki dönem yarısında da üstün.
            gate = "ACIK"
            _px = float(ddf["close"].iloc[-1])
            _riskp = (_px - mv_stop) / _px * 100.0
            verdict = (f"Giriş koşulları oluştu — VCP tabanı ({mv_dalga} daralma dalgası) "
                       f"tamamlandı ve pivot {mv_pivot:.2f} hacimle kırıldı. "
                       f"Stop {mv_stop:.2f} (taban dibi, risk %{_riskp:.1f}).{rs_note}")
            verdict_kind = "success"
        elif mv_break and np.isfinite(mv_stop) and np.isfinite(mv_pivot):
            # Kırılım var ama stop çok uzak → Minervini risk kuralı gereği giriş yok
            gate = "BEKLEMEDE"
            _rp = (float(ddf["close"].iloc[-1]) - mv_stop) / float(ddf["close"].iloc[-1]) * 100.0
            verdict = (f"Pivot {mv_pivot:.2f} kırıldı ancak taban geniş — stop {mv_stop:.2f} "
                       f"(%{_rp:.1f} uzakta), üst sınır %{MAX_BASE_RISK*100:.0f}. "
                       f"Bu noktadan giriş kabul edilebilir riski aşar.{rs_note}")
            verdict_kind = "warning"
        elif mv_base and np.isfinite(mv_pivot):
            # Taban kurulu, kırılım bekleniyor — ALARM BURAYA KURULUR
            gate = "BEKLEMEDE"
            _px = float(ddf["close"].iloc[-1])
            _uz = (mv_pivot / _px - 1.0) * 100.0 if _px > 0 else float("nan")
            verdict = (f"Aday — VCP tabanı kurulu ({mv_dalga} daralma dalgası). "
                       f"Pivot {mv_pivot:.2f}"
                       + (f", fiyatın %{_uz:.1f} üstünde" if np.isfinite(_uz) and _uz > 0 else "")
                       + f". Taban dibi {mv_dip:.2f}. Pivot hacimle kırılırsa giriş koşulu oluşur."
                       + rs_note)
            verdict_kind = "warning"
        else:
            # Yapı uygun ama kurulum yok — sebebi yazılır
            gate = "BEKLEMEDE"
            _neden = "henüz konsolidasyon oluşmamış"
            try:
                if np.isfinite(mv_pivot) and np.isfinite(mv_dip) and mv_pivot > 0:
                    _d = (mv_pivot - mv_dip) / mv_pivot * 100.0
                    _px = float(ddf["close"].iloc[-1])
                    _r = (_px - mv_dip * 0.99) / _px * 100.0 if _px > 0 else float("nan")
                    if _d > 35:
                        _neden = f"taban çok derin (%{_d:.0f}) — sağlıklı konsolidasyon değil"
                    elif _d < 5:
                        _neden = f"taban çok sığ (%{_d:.0f}) — yapı henüz oturmamış"
                    elif np.isfinite(_r) and _r > MAX_BASE_RISK * 100:
                        _neden = (f"taban geniş — stop %{_r:.0f} uzakta "
                                  f"(üst sınır %{MAX_BASE_RISK*100:.0f})")
                    else:
                        _neden = "daralma dalgaları veya hacim kuruması tamamlanmamış"
            except Exception:
                pass
            verdict = (f"Aday — haftalık yapı uygun (setup {w_plan.setup_score}/100), "
                       f"ancak giriş kurulumu yok: {_neden}. "
                       f"Haftalık bant: {_wlo:.2f} – {_whi:.2f}.{rs_note}")
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
            "teyit_v2": teyit_v2,
            "teyit_v1_shadow": teyit_v1_shadow,
            "armed_recent": armed_recent,
            "armed_days_ago": armed_days_ago,
            "last_close": float(ddf["close"].iloc[-1]) if ddf is not None and len(ddf) else float("nan"),
            "mv_break": mv_break, "mv_base": mv_base,
            "mv_dalga": mv_dalga, "mv_son_daralma": mv_son_daralma,
            "mv_risk_pct": (((float(ddf["close"].iloc[-1]) - mv_stop) / float(ddf["close"].iloc[-1]) * 100.0)
                            if np.isfinite(mv_stop) and len(ddf) else float("nan")),
            "band_tol_pct": _band_tol,
            "teyit_eksik": teyit_eksik,
            "chase_ok": _chase.get("ok", True),
            "chase_sebep": _chase.get("sebep", ""),
            "entry_risk_pct": _chase.get("risk_pct", float("nan")),
            "mv_pivot": mv_pivot, "mv_dip": mv_dip,
            "mv_stop": mv_stop, "mv_tp1": mv_tp1, "mv_tp2": mv_tp2,
            "rs_edge_w": (
                sum(rs_edges.get(k, float("nan")) * w for k, w in
                    [("edge_3m", 0.30), ("edge_6m", 0.35), ("edge_12m", 0.35)])
                if all(np.isfinite(rs_edges.get(k, float("nan"))) for k in ("edge_3m", "edge_6m", "edge_12m"))
                else float("nan")
            ),
            "rs_edge_3m": rs_edges.get("edge_3m", float("nan")),
            "rs_edge_6m": rs_edges.get("edge_6m", float("nan")),
            "rs_edge_12m": rs_edges.get("edge_12m", float("nan")),
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


SCAN_GIST_FILE = "minerwin_scan.json"


def scan_gist_kaydet(rows: list, rs_sira: dict) -> bool:
    """Son tarama sonucunu + RS sıralamasını Gist'e yazar (oturumdan bağımsız).
    Uygulama yeniden başlasa da son tarama korunur."""
    if not GITHUB_TOKEN or not rows:
        return False
    try:
        gid = _gist_find()
        if not gid:
            return False
        paket = json.dumps({
            "zaman": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M"),
            "rows": rows, "rs_sira": rs_sira,
        }, ensure_ascii=False)
        r = requests.patch(f"https://api.github.com/gists/{gid}", headers=_gh_headers(),
                           json={"files": {SCAN_GIST_FILE: {"content": paket}}}, timeout=20)
        r.raise_for_status()
        return True
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def scan_gist_oku(_nonce: int = 0) -> dict:
    """Gist'teki son taramayı okur. Dönen: {zaman, rows, rs_sira} veya {}."""
    if not GITHUB_TOKEN:
        return {}
    try:
        gid = _gist_find()
        if not gid:
            return {}
        r = requests.get(f"https://api.github.com/gists/{gid}", headers=_gh_headers(), timeout=15)
        r.raise_for_status()
        f = (r.json().get("files") or {}).get(SCAN_GIST_FILE) or {}
        icerik = f.get("content", "")
        if f.get("truncated") and f.get("raw_url"):
            icerik = requests.get(f["raw_url"], headers=_gh_headers(), timeout=20).text
        return json.loads(icerik) if icerik else {}
    except Exception:
        return {}


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

    # KALİBRASYON (V7.4): stop mesafesi × STOP_TIGHTEN.
    # 125 hisse / 3 yıl portföy simülasyonu: aynı girişlerle
    #   ×1.00 (eski) → 10.650 $   |   ×0.75 (kalibre) → 14.674 $
    # Daha yakın stop = kaybeden pozisyon erken kesiliyor, risk başına
    # daha büyük pozisyon açılabiliyor; kazananlar iz süren stopla taşınıyor.
    try:
        if STOP_TIGHTEN != 1.0 and entry_mid > stop > 0:
            stop = entry_mid - (entry_mid - stop) * STOP_TIGHTEN
            stop_dbg["stop_tighten"] = STOP_TIGHTEN
    except Exception:
        pass

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
        "value":    S("value", fontName=fn_bold, fontSize=12, le