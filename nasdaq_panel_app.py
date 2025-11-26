import os
import time
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from scipy.signal import argrelextrema
from typing import List, Dict, Any, Optional
import yfinance as yf

FINNHUB_API_KEY = "d4alt99r01qseda29umgd4alt99r01qseda29un0"  # Artık kullanılmıyor ama kalabilir

# ================== GENEL AYARLAR ==================

st.set_page_config(
    page_title="📈 NASDAQ M30 AI Trading Panel",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = "nasdaq_training_dataset_v2.parquet"
MODEL_PATH = "models/nasdaq_meta_optuna_cv_v2.pkl"
CALIBRATOR_PATH = "models/nasdaq_calibrator.pkl"  # Kalibratör dosyası yolu (opsiyonel)

# Ana timeframe M30 kolonları
# ================== KOLON TANIMLARI (H1 FORMATI) ==================
TIME_COL = "datetime"  # Bu zaten doğru
OPEN_COL = "H1_open"
HIGH_COL = "H1_high"
LOW_COL = "H1_low"
CLOSE_COL = "H1_close"
VOL_COL = "H1_volume"
EMA20_COL = "H1_ema_20"
EMA50_COL = "H1_ema_50"
EMA200_COL = "H1_sma_200"  # veya "H1_ema_200" ne varsa
SUP_COL = "H1_support_strength"
RES_COL = "H1_resistance_strength"
# Sinyal için threshold'lar
THR_BUY = 0.55   # p(1) >= 0.55 → AL
THR_SELL = 0.45  # p(1) <= 0.45 → SAT


# ================== STİL / TEMA ==================

st.markdown("""
<style>
    /* ===== MODERN DARK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Glassmorphism cards */
    .signal-box, .metric-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .signal-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Gradient text effects */
    .main-header {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    /* ===== BUY/SELL BADGES ===== */
    .buy-badge {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 14px rgba(46, 160, 67, 0.3);
        animation: pulse-green 2s infinite;
    }
    
    .sell-badge {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 14px rgba(248, 81, 73, 0.3);
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 4px 14px rgba(46, 160, 67, 0.3); }
        50% { box-shadow: 0 4px 24px rgba(46, 160, 67, 0.6); }
        100% { box-shadow: 0 4px 14px rgba(46, 160, 67, 0.3); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 14px rgba(248, 81, 73, 0.3); }
        50% { box-shadow: 0 4px 24px rgba(248, 81, 73, 0.6); }
        100% { box-shadow: 0 4px 14px rgba(248, 81, 73, 0.3); }
    }
    
    /* ===== BIG MARKET DATA HEADER ===== */
    .market-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.95));
        backdrop-filter: blur(20px);
        border: 2px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin: 16px 0 24px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .market-title {
        font-size: 20px;
        font-weight: 600;
        color: #8b949e;
        margin-bottom: 8px;
    }
    
    .market-price {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: price-glow 3s ease-in-out infinite;
        letter-spacing: -1px;
    }
    
    .market-change {
        font-size: 24px;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .change-positive {
        color: #2ea043;
        animation: pulse-green 2s infinite;
    }
    
    .change-negative {
        color: #f85149;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes price-glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    .market-info {
        text-align: right;
    }
    
    .market-timestamp {
        font-size: 14px;
        color: #6e7681;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)
# ================== YARDIMCI FONKSİYONLAR ==================
def find_swings(df: pd.DataFrame, high_col: str, low_col: str, order: int = 5):
    """Find swing highs/lows using scipy"""
    from scipy.signal import argrelextrema
    highs = argrelextrema(df[high_col].values, np.greater_equal, order=order)[0]
    lows = argrelextrema(df[low_col].values, np.less_equal, order=order)[0]
    return highs, lows

def detect_flag_pattern(df: pd.DataFrame, close_col: str, lookback: int = 20):
    """Basic flag detection (returns last swing points)"""
    if len(df) < lookback:
        return {"detected": False, "pole_start": None, "pole_end": None}
    recent = df.tail(lookback)
    highs, lows = find_swings(recent, close_col, close_col, order=3)
    return {
        "detected": len(highs) >= 2 and len(lows) >= 2,
        "pole_start": recent.iloc[0][close_col],
        "pole_end": recent.iloc[-1][close_col]
    }

@st.cache_data(ttl=15)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        # 🔥 KRİTİK: Önce sırala, sonra duplicate'leri kaldır
        df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep='last')
        df = df.reset_index(drop=True)
    
    # Model feature'larını seç
    model_bundle = load_model_bundle(MODEL_PATH)
    required_features = model_bundle['features']
    
    keep_cols = [TIME_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]
    for feat in required_features:
        if feat not in keep_cols and feat in df.columns:
            keep_cols.append(feat)
    
    keep_cols = list(dict.fromkeys(keep_cols))
    df_filtered = df[keep_cols].copy()
    
    # NaN doldur
    df_filtered = df_filtered.fillna(0.0001)
    
    # İlk 200 barı at
    if len(df_filtered) > 200:
        df_filtered = df_filtered.iloc[200:].reset_index(drop=True)
    
    return df_filtered

# ✅ YENİ FONKSİYON (Finnhub'dan canlı veri):
# ============ SATIR 140'DAKİ FONKSİYONU SİLİN VE BUNU YAPIŞTIRIN ============


# ======== EKONOMİK TAKVİM WIDGET ========
@st.cache_data(ttl=300)  # 5 dakika cache
def get_finnhub_calendar():
    """Finnhub'dan ekonomik takvim verileri"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        response = requests.get(
            f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={next_week}&token={FINNHUB_API_KEY}"
        )
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get("economicCalendar", []))
        return pd.DataFrame()
        
    except:
        return pd.DataFrame()

# ======== HABER WIDGET'I ========
# ============ SATIR 580'DEKİ get_finnhub_news'I DEĞİŞTİRİN ============
@st.cache_data(ttl=600)
def get_finnhub_news():
    """Finnhub'dan market haberleri - TARİH DÜZELTİLMİŞ"""
    try:
        response = requests.get(
            f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}",
            timeout=10
        )
        
        if response.status_code == 200:
            news_data = response.json()
            df = pd.DataFrame(news_data)
            
            # ✅ Tarih düzeltmesi
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
                df = df.dropna(subset=['datetime'])
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Haber çekme hatası: {e}")
        return pd.DataFrame()
def create_news_widget():
    """Finnhub ekonomik takvim ve haber widget'ı"""
    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
    st.markdown("#### 📰 Economic Calendar (Finnhub)")
    
    df_cal = get_finnhub_calendar()
    
    if not df_cal.empty:
        # Sadece önemli olanları göster
        df_cal = df_cal[df_cal["impact"].isin(["High", "Medium"])]
        df_cal = df_cal.sort_values("time").head(5)
        
        for _, row in df_cal.iterrows():
            event_time = pd.to_datetime(row["time"])
            time_diff = event_time - pd.Timestamp.now()
            hours = int(time_diff.total_seconds() // 3600)
            minutes = int((time_diff.total_seconds() % 3600) // 60)
            
            impact_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}[row["impact"]]
            
            st.markdown(f"""
                <div style="border-left: 4px solid {impact_color}; padding: 8px; margin: 8px 0; 
                           background: rgba(22,27,34,0.5); border-radius: 4px;">
                    <strong>{row['event']}</strong> ({row['country']})<br>
                    <small>⏰ {hours}h {minutes}m remaining</small> | 
                    <span style="color: {impact_color};">Impact: {row['impact']}</span> | 
                    <span>Expected: {row.get('estimate', 'N/A')}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No upcoming events in next 7 days.")
    
    # Haberler (Takvimin altına ekle)
    st.markdown("---")
    st.markdown("#### 📊 Market News")
    df_news = get_finnhub_news()
    
    if not df_news.empty:
        for _, article in df_news.head(3).iterrows():
            # Tarih formatını düzelt
            try:
                article_date = pd.to_datetime(article['datetime']).strftime("%Y-%m-%d")
            except:
                article_date = str(article.get('datetime', 'N/A'))[:10] if 'datetime' in article else 'N/A'
            
            st.markdown(f"""
                <div style="padding: 8px; margin: 8px 0; background: rgba(22,27,34,0.5); 
                           border-radius: 4px; border-left: 2px solid #58a6ff;">
                    <strong>{article.get('headline', 'No headline')}</strong><br>
                    <small>{article.get('source', 'Unknown')} • {article_date}</small><br>
                    <a href="{article.get('url', '#')}" target="_blank" style="color: #58a6ff;">Read more →</a>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news available at the moment.")
    
    st.markdown("</div>", unsafe_allow_html=True)
@st.cache_resource
def load_model_bundle(path: str):
    """Modeli yükle"""
    if not os.path.exists(path):
        st.error(f"❌ Model not found: {path}")
        return {"ensemble": [], "scaler": None, "features": None}
    
    try:
        obj = joblib.load(path)
        ensemble = obj.get("models") or obj.get("ensemble") or []
        if not isinstance(ensemble, list):
            ensemble = [ensemble]
        
        ensemble = [m for m in ensemble if hasattr(m, "predict_proba")]
        
        return {
            "ensemble": ensemble,
            "scaler": obj.get("scaler"),
            "features": obj.get("features") or obj.get("feature_names")
        }
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return {"ensemble": [], "scaler": None, "features": None}

# 3. yfinance data loader - TAMAMEN ÜCRETSİZ, API KEY GEREKMİYOR
@st.cache_data(ttl=0)  # 🔥 DEBUG: Cache kapalı
def load_finnhub_data():
    """yfinance ile canlı NASDAQ M30 verisi - MODEL UYUMLU DEBUG VERSİYON"""
    try:
        # 🎯 NASDAQ sembolleri (sırayla denenecek)
        symbols = ['^NDX', 'QQQ', '^IXIC']
        
        df = pd.DataFrame()
        success_symbol = None
        
        for symbol in symbols:
            try:
                st.info(f"🔄 {symbol} sembolü deneniyor...")
                
                # yfinance ile son 10 gün 30 dakikalık veri çek
                ticker = yf.Ticker(symbol)
                df_raw = ticker.history(period='10d', interval='30m')
                
                # 🔍 DEBUG: Ham veriyi kontrol et
                if not df_raw.empty:
                    st.success(f"✅ Ham veri alındı: {len(df_raw)} bar")
                    
                    # 🔥 Tarih formatını düzelt - timezone sorununu çöz
                    if df_raw.index.tz is not None:
                        # UTC'den New York saatine çevir, sonra timezone bilgisini kaldır
                        dates = df_raw.index.tz_convert('America/New_York').tz_localize(None)
                    else:
                        dates = df_raw.index
                    
                    # 🔍 DEBUG: Tarih aralığını göster
                    st.caption(f"📅 Ham veri aralığı: {dates[0]} → {dates[-1]}")
                    
                    # ✅ MODEL UYUMLU: H1_ prefix ekle (model bunu bekliyor)
                    df = pd.DataFrame({
                        TIME_COL: pd.to_datetime(dates),  # Pandas datetime objesi
                        OPEN_COL: df_raw['Open'].astype(float).values,
                        HIGH_COL: df_raw['High'].astype(float).values,
                        LOW_COL: df_raw['Low'].astype(float).values,
                        CLOSE_COL: df_raw['Close'].astype(float).values,
                        VOL_COL: df_raw['Volume'].astype(float).values
                    })
                    
                    # 🔍 DEBUG: İlk birkaç satırı göster
                    st.caption(f"📊 İlk OHLC: O={df[OPEN_COL].iloc[0]:.2f} H={df[HIGH_COL].iloc[0]:.2f} L={df[LOW_COL].iloc[0]:.2f} C={df[CLOSE_COL].iloc[0]:.2f}")
                    st.caption(f"📊 Son OHLC: O={df[OPEN_COL].iloc[-1]:.2f} H={df[HIGH_COL].iloc[-1]:.2f} L={df[LOW_COL].iloc[-1]:.2f} C={df[CLOSE_COL].iloc[-1]:.2f}")
                    
                    # 🔍 Veri kalitesi kontrolü
                    if (df[OPEN_COL] == 0).any() or (df[CLOSE_COL] == 0).any():
                        st.warning(f"⚠️ {symbol} verilerinde 0 değerler var, atlıyorum...")
                        continue
                    
                    success_symbol = symbol
                    st.success(f"✅ Veri işlendi: {symbol} ({len(df)} bar)")
                    break
                else:
                    st.warning(f"⚠️ {symbol} için veri yok")
                    
            except Exception as e:
                st.error(f"❌ {symbol} hatası: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                continue
        
        # Hiçbir sembol çalışmadıysa yedek veriyi yükle
        if df.empty:
            st.error(f"❌ Tüm semboller başarısız: {symbols}")
            st.warning("🔄 Yedek veriye geçiliyor...")
            return load_backup_data()
        
        # EMA hesapla (model bunları bekliyor)
        st.info("📊 EMA'lar hesaplanıyor...")
        df[EMA20_COL] = df[CLOSE_COL].ewm(span=20, adjust=False).mean()
        df[EMA50_COL] = df[CLOSE_COL].ewm(span=50, adjust=False).mean()
        df[EMA200_COL] = df[CLOSE_COL].rolling(200).mean()
        
        # NaN kontrolü
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            st.warning(f"⚠️ {nan_count} adet NaN değer var, dolduruluyor...")
            df = df.ffill().fillna(0)
        
        # Son kontrol: Tarihlerin doğru olup olmadığını kontrol et
        year_check = df[TIME_COL].dt.year.iloc[0]
        if year_check < 2020:
            st.error(f"❌ Tarih hatası tespit edildi: {year_check}. Yedek veriye geçiliyor...")
            return load_backup_data()
        
        # Sıralama ve temizleme
        df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep='last').reset_index(drop=True)
        
        st.success(f"✅ Veri hazır: {len(df)} bar | {df[TIME_COL].iloc[0]} → {df[TIME_COL].iloc[-1]}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Kritik hata: {e}")
        import traceback
        st.code(traceback.format_exc())
        return load_backup_data()

def resample_data(df, timeframe):
    """Veriyi verilen timeframe'e göre yeniden örnekler (resample)"""
    if df.empty:
        return df
    
    # Timeframe mapping (FutureWarning'leri önlemek için 'min' ve 'h' kullan)
    tf_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '1d': '1D'
    }
    
    resample_rule = tf_map.get(timeframe, '30min')
    
    # TIME_COL'u index yap
    df_resampled = df.set_index(TIME_COL).resample(resample_rule).agg({
        OPEN_COL: 'first',
        HIGH_COL: 'max',
        LOW_COL: 'min',
        CLOSE_COL: 'last',
        VOL_COL: 'sum',
    })
    
    # EMA'lar varsa onları da ekle
    if EMA20_COL in df.columns:
        df_resampled[EMA20_COL] = df.set_index(TIME_COL)[EMA20_COL].resample(resample_rule).last()
    if EMA50_COL in df.columns:
        df_resampled[EMA50_COL] = df.set_index(TIME_COL)[EMA50_COL].resample(resample_rule).last()
    if EMA200_COL in df.columns:
        df_resampled[EMA200_COL] = df.set_index(TIME_COL)[EMA200_COL].resample(resample_rule).last()
    
    # NaN'ları temizle ve index'i sıfırla
    df_resampled = df_resampled.dropna(subset=[CLOSE_COL]).reset_index()
    
    return df_resampled

def load_backup_data():
    """Yedek veri yükleme (parquet) - MODEL UYUMLU"""
    backup_file = "nasdaq_training_dataset_v2.parquet"
    
    if os.path.exists(backup_file):
        st.warning(f"⚠️ Yedek veri kullanılıyor: {backup_file}")
        df = pd.read_parquet(backup_file)
        
        # 🔍 DEBUG: Hangi kolonlar var?
        st.caption(f"📊 Yedek veri kolonları: {list(df.columns)[:10]}...")
        
        # Datetime kolonunu kontrol et
        if TIME_COL not in df.columns and 'datetime' in df.columns:
            df = df.rename(columns={'datetime': TIME_COL})
        
        # Datetime tipini düzelt
        if TIME_COL in df.columns:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        
        # Kolonları H1 formatına çevir (eğer değilse)
        if OPEN_COL not in df.columns:
            st.info("🔄 Kolon isimleri H1 formatına çevriliyor...")
            rename_map = {
                'open': OPEN_COL,
                'high': HIGH_COL,
                'low': LOW_COL,
                'close': CLOSE_COL,
                'volume': VOL_COL
            }
            df = df.rename(columns=rename_map)
        
        # Gerekli kolonların varlığını kontrol et
        required_cols = [TIME_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Yedek veride eksik kolonlar: {missing_cols}")
            return pd.DataFrame()
        
        st.success(f"✅ Yedek veri yüklendi: {len(df)} bar")
        return df
    else:
        st.error("❌ Yedek veri dosyası bulunamadı!")
        return pd.DataFrame()
def load_calibrator_bundle(path: str):
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        return obj
    except Exception:
        return None

def prepare_features_for_model(last_row: pd.Series, model_bundle: dict) -> np.ndarray:
    feat_names = model_bundle["features"]
    scaler = model_bundle["scaler"]
    
    row_df = last_row.to_frame().T
    row_df = row_df.reindex(columns=feat_names, fill_value=0.0)
    
    # Sadece sayısal olmayanları temizle
    for c in row_df.columns:
        if not np.issubdtype(row_df[c].dtype, np.number):
            row_df[c] = pd.to_numeric(row_df[c], errors="coerce")
    
    row_df = row_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_arr = row_df.values.astype(float)
    
    if scaler is not None:
        X_arr = scaler.transform(X_arr)
    
    return X_arr


def get_model_signal(last_row: pd.Series, model_bundle: dict, thr_buy: float, thr_sell: float):
    """
    Ensemble modelden p_up/p_down ve sinyal label'ı üretir.
    Kalibratör KULLANMIYORUZ. Sade versiyon.
    """
    ensemble = model_bundle.get("ensemble", None)

    # Model hiç yüklenmemişse burada duralım
    if ensemble is None or len(ensemble) == 0:
        return {
            "label": "BELİRSİZ",
            "p_up": 0.5,
            "p_down": 0.5,
            "confidence": 0.0,
        }

    # Özellikleri hazırla
    X_arr = prepare_features_for_model(last_row, model_bundle)

    probs_list = []
    for m in ensemble:
        proba = m.predict_proba(X_arr)
        # class=1 yukarı olsun demiştik
        probs_list.append(proba[:, 1])

    p_up = float(np.mean(probs_list))
    p_down = 1.0 - p_up
    confidence = max(p_up, p_down)

    # Eşiklere göre label
    if p_up >= thr_buy:
        label = "AL"
    elif p_up <= thr_sell:
        label = "SAT"
    else:
        label = "PAS"

    return {
        "label": label,
        "p_up": p_up,
        "p_down": p_down,
        "confidence": confidence,
    }
def simulate_model_trades(
    df: pd.DataFrame,
    model_bundle: dict,
    thr_buy: float,
    thr_sell: float,
    horizon_bars: int = 10,
    conf_threshold: float = 0.55,  # slider'dan gelecek, default 0.55
) -> pd.DataFrame:
    trades = []

    start_idx = 50  # ilk barlarda indikatörler NaN olabiliyor

    for i in range(start_idx, len(df) - horizon_bars):
        row = df.iloc[i]

        signal = get_model_signal(row, model_bundle, thr_buy, thr_sell)
        if signal is None:
            continue

        p_up = float(signal["p_up"])
        p_down = float(signal["p_down"])
        confidence = float(signal["confidence"])
        label = signal["label"]

        # 1) Güven filtresi
        if confidence < conf_threshold:
            continue

        # 2) Sadece net AL / SAT al
        if label == "AL":
            direction = 1   # LONG
        elif label == "SAT":
            direction = -1  # SHORT
        else:
            continue

        # 3) Giriş bilgileri
        entry_price = float(row[CLOSE_COL])
        entry_time = row[TIME_COL]

        future = df.iloc[i + 1 : i + 1 + horizon_bars].copy()
        if future.empty:
            break

        prices = future[CLOSE_COL].astype(float).to_numpy()
        times = future[TIME_COL].to_list()

        # 4) TP / SL puanları
        TP_POINTS = 25.0
        SL_POINTS = 25.0

        tp = entry_price + TP_POINTS * direction
        sl = entry_price - SL_POINTS * direction

        exit_idx = len(prices) - 1
        exit_price = float(prices[-1])
        exit_time = times[-1]
        final_pnl = float((exit_price - entry_price) * direction)

        for j, price in enumerate(prices):
            price = float(price)

            hit_tp = (direction == 1 and price >= tp) or (direction == -1 and price <= tp)
            hit_sl = (direction == 1 and price <= sl) or (direction == -1 and price >= sl)

            if hit_tp:
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = TP_POINTS
                break
            if hit_sl:
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = -SL_POINTS
                break

        # 5) PnL analiz
        pnl_series_full = (prices - entry_price) * direction
        max_fav = float(pnl_series_full.max())
        max_adv = float(pnl_series_full.min())

        pnl_series_open = pnl_series_full[: exit_idx + 1]
        bars_in_profit = int((pnl_series_open > 0).sum())
        bars_in_loss = int((pnl_series_open < 0).sum())
        minutes_in_profit = bars_in_profit * 30
        minutes_in_loss = bars_in_loss * 30

        trades.append(
            {
                "entry_idx": i,
                "entry_time": entry_time,
                "signal_label": label,
                "p_up": p_up,
                "p_down": p_down,
                "confidence": confidence,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "holding_bars": exit_idx + 1,
                "holding_minutes": (exit_idx + 1) * 30,
                "pnl_points": final_pnl,
                "max_favorable": max_fav,
                "max_adverse": max_adv,
                "minutes_in_profit": minutes_in_profit,
                "minutes_in_loss": minutes_in_loss,
            }
        )

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)
def calculate_pattern_completion(pattern: Dict, current_price: float) -> float:
    """
    Calculate remaining pips to pattern completion
    Returns: pips remaining (float)
    """
    if pattern["name"] in ["Head and Shoulders", "Inverse Head and Shoulders"]:
        # Next leg is the breakout from neckline
        neckline = pattern.get("neckline", current_price)
        return abs(current_price - neckline) * 0.3  # Approximate
    
    elif "Triangle" in pattern["name"]:
        # Distance to apex or breakout
        support = pattern.get("support", current_price)
        resistance = pattern.get("resistance", current_price)
        return min(abs(current_price - support), abs(resistance - current_price))
    
    elif "Double" in pattern["name"]:
        # Distance to confirm breakout
        avg_level = pattern.get("resistance" if "Top" in pattern["name"] else "support", current_price)
        return abs(current_price - avg_level) * 0.5
    
    return 50.0  # Default fallback

def calculate_trend_channels(df: pd.DataFrame, window: int = 80):
    """Basit otomatik trend kanalı (üst, orta, alt)."""
    if len(df) < window + 10:
        return None, None, None

    highs = df[HIGH_COL].rolling(window=window).max()
    lows = df[LOW_COL].rolling(window=window).min()

    x = np.arange(len(df))
    mask = ~np.isnan(highs.values) & ~np.isnan(lows.values)
    if mask.sum() < window // 2:
        return None, None, None

    z_high = np.polyfit(x[mask], highs.values[mask], 1)
    z_low = np.polyfit(x[mask], lows.values[mask], 1)

    upper_channel = np.poly1d(z_high)(x)
    lower_channel = np.poly1d(z_low)(x)
    middle_channel = (upper_channel + lower_channel) / 2.0
    return upper_channel, middle_channel, lower_channel

def find_zones_price_action(
    df: pd.DataFrame,
    price_col: str,
    last_price: float,
    side: str,
    zone_half_width: float = 25.0,  # toplam 50 puanlık bölge
    min_touches: int = 3,           # en az 3 dokunma
    top_k: int = 3,                 # en yakın/güçlü ilk 3
):
    """
    Fiyat bazlı (price-action) destek/direnç tespiti.

    Mantık:
      - Önce local dip/tepe (pivot) noktalarını buluyoruz.
      - Bu pivotları 50 puanlık (±25) zonelara kümeliyoruz.
      - Her zonedaki toplam dokunma sayısını sayıyoruz.
      - En az 3 dokunma olan zoneları destek/direnç olarak alıyoruz.
    """
    prices = df[price_col].to_numpy(dtype=float)

    # Yeterli veri yoksa boş dön
    if len(prices) < 20:
        return []

    # Pivotları bul
    if side == "support":
        # local dip: önceki ve sonraki mumlardan daha düşük/eşit
        piv_idx = argrelextrema(prices, np.less_equal, order=3)[0]
    else:
        # local tepe
        piv_idx = argrelextrema(prices, np.greater_equal, order=3)[0]

    if len(piv_idx) == 0:
        return []

    # Sadece ilgili taraftaki (aşağı/yukarı) pivotları al
    pivots = []
    for idx in piv_idx:
        p = prices[idx]
        if side == "support" and p > last_price:
            continue
        if side == "resistance" and p < last_price:
            continue
        pivots.append((idx, p))

    if not pivots:
        return []

    # Fiyata göre sırala
    pivots.sort(key=lambda x: x[1])

    # Pivotları zone'lara (±zone_half_width) göre kümele
    clusters = []
    for idx, p in pivots:
        if not clusters:
            clusters.append({"level": p, "indices": [idx]})
        else:
            last = clusters[-1]
            if abs(p - last["level"]) <= zone_half_width:
                last["indices"].append(idx)
                # merkez seviyeyi güncelle
                last["level"] = float(
                    np.mean([prices[i] for i in last["indices"]])
                )
            else:
                clusters.append({"level": p, "indices": [idx]})

    zones = []
    for c in clusters:
        center = float(c["level"])
        low = center - zone_half_width
        high = center + zone_half_width

        # Bu bölgeye dokunan tüm mum sayısı
        touches_mask = (prices >= low) & (prices <= high)
        touch_indices = np.where(touches_mask)[0]
        touches = int(len(touch_indices))

        if touches < min_touches:
            continue

        first_touch = int(touch_indices[0])
        last_touch = int(touch_indices[-1])

        zones.append(
            {
                "level": center,
                "touches": touches,                    # toplam dokunma
                "count_pivots": len(c["indices"]),     # pivot sayısı
                "first_touch": first_touch,
                "last_touch": last_touch,
                "touch_span": last_touch - first_touch,
            }
        )

    # Dokunma sayısı + yayılımına göre sırala
    zones.sort(key=lambda z: (-z["touches"], -z["touch_span"]))
    return zones[:top_k]
# ============================================================
#  BÜYÜK FORMASYON ALGILAMA YARDIMCILARI
#  - Swing noktası çıkarma
#  - HS / Double Top / Üçgen örnekleri
# ============================================================
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

# ==================== BÜYÜK FORMASYONLAR ====================
def detect_swings_from_price(
    price_series: pd.Series,
    left: int = 3,
    right: int = 3,
    min_prominence: float = 10.0,
) -> list:
    """
    Basit swing noktası tespiti:
      - 'H': lokal tepe (high)
      - 'L': lokal dip (low)

    min_prominence: ardışık swing'ler arası min fiyat farkı (noise filtre).
    """
    values = price_series.to_numpy(dtype=float)
    idxs = np.arange(len(values))

    swings = []
    n = len(values)
    if n < left + right + 1:
        return swings

    last_price = None

    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        center = values[i]

        is_max = center == window.max()
        is_min = center == window.min()

        if not (is_max or is_min):
            continue

        # Prominence filtresi: çok küçük zikzakları at
        if last_price is not None and abs(center - last_price) < min_prominence:
            continue

        s_type = "H" if is_max else "L"
        swings.append(
            {
                "idx": int(idxs[i]),
                "price": float(center),
                "type": s_type,
            }
        )
        last_price = center

    return swings


def _pattern_leg_info(swings: list, total_legs: int) -> tuple:
    """
    Kaç bacak yakalanmış, hangi ayağı dolduruyoruz.
    """
    legs_found = min(len(swings), total_legs)
    current_leg = legs_found  # 1-based
    return legs_found, current_leg

def detect_head_shoulders(swings: list) -> Optional[Dict]:
    """
    Head and Shoulders (Baş-Omuz) formasyonu tespiti.
    L-H-L-H-L veya H-L-H-L-H pattern'i arar.
    """
    if len(swings) < 5:
        return None
    
    # Son 5 swing'i al
    recent = swings[-5:] if len(swings) >= 5 else swings
    
    # Baş-Omuz: H-L-H-L-H (inverse: L-H-L-H-L)
    # Normal HS: Left Shoulder (H) - Dip (L) - Head (H, en yüksek) - Dip (L) - Right Shoulder (H, düşük)
    if len(recent) == 5:
        types = [s["type"] for s in recent]
        prices = [s["price"] for s in recent]
        
        # Normal Head and Shoulders: H-L-H-L-H
        if types == ["H", "L", "H", "L", "H"]:
            left_shoulder = prices[0]
            head = prices[2]
            right_shoulder = prices[4]
            
            # Head, her iki shoulder'dan yüksek olmalı
            if head > left_shoulder and head > right_shoulder:
                # Shoulder'lar yaklaşık aynı seviyede olmalı
                shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                if shoulder_diff < 0.05:  # %5 tolerans
                    neckline = (prices[1] + prices[3]) / 2
                    return {
                        "name": "Head and Shoulders (Baş-Omuz)",
                        "legs_total": 5,
                        "legs_found": 5,
                        "current_leg": 5,
                        "swing_points": recent,
                        "direction": "down",
                        "neckline": neckline,
                        "remaining_pips_to_next_leg": abs(prices[-1] - neckline) * 0.3,
                        "stage_text": "Baş-omuz formasyonu tamamlandı - aşağı kırılım bekleniyor",
                    }
        
        # Inverse Head and Shoulders: L-H-L-H-L
        elif types == ["L", "H", "L", "H", "L"]:
            left_shoulder = prices[0]
            head = prices[2]
            right_shoulder = prices[4]
            
            # Head, her iki shoulder'dan düşük olmalı
            if head < left_shoulder and head < right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                if shoulder_diff < 0.05:
                    neckline = (prices[1] + prices[3]) / 2
                    return {
                        "name": "Inverse Head and Shoulders (Ters Baş-Omuz)",
                        "legs_total": 5,
                        "legs_found": 5,
                        "current_leg": 5,
                        "swing_points": recent,
                        "direction": "up",
                        "neckline": neckline,
                        "remaining_pips_to_next_leg": abs(neckline - prices[-1]) * 0.3,
                        "stage_text": "Ters baş-omuz formasyonu tamamlandı - yukarı kırılım bekleniyor",
                    }
    
    return None

def detect_double_top_bottom(swings: list, tolerance: float = 20.0) -> Optional[Dict]:
    """
    Double Top (Çift Tepe) ve Double Bottom (Çift Dip) tespiti.
    """
    if len(swings) < 4:
        return None
    
    # Son 4-5 swing'i al
    recent = swings[-5:] if len(swings) >= 5 else swings[-4:]
    
    highs = [s for s in recent if s["type"] == "H"]
    lows = [s for s in recent if s["type"] == "L"]
    
    # Double Top: İki tepe yaklaşık aynı seviyede, aralarında bir dip
    if len(highs) >= 2:
        # Son iki tepeyi al
        top1 = highs[-2]
        top2 = highs[-1]
        
        price_diff = abs(top1["price"] - top2["price"])
        if price_diff <= tolerance:
            # Aralarında bir dip olmalı
            dip_between = [s for s in recent if s["type"] == "L" and top1["idx"] < s["idx"] < top2["idx"]]
            if dip_between:
                avg_top = (top1["price"] + top2["price"]) / 2
                return {
                    "name": "Double Top (Çift Tepe)",
                    "legs_total": 4,
                    "legs_found": len(recent),
                    "current_leg": len(recent),
                    "swing_points": recent,
                    "direction": "down",
                    "resistance": avg_top,
                    "remaining_pips_to_next_leg": price_diff * 0.5,
                    "stage_text": "Çift tepe formasyonu - aşağı kırılım bekleniyor",
                }
    
    # Double Bottom: İki dip yaklaşık aynı seviyede, aralarında bir tepe
    if len(lows) >= 2:
        bottom1 = lows[-2]
        bottom2 = lows[-1]
        
        price_diff = abs(bottom1["price"] - bottom2["price"])
        if price_diff <= tolerance:
            # Aralarında bir tepe olmalı
            top_between = [s for s in recent if s["type"] == "H" and bottom1["idx"] < s["idx"] < bottom2["idx"]]
            if top_between:
                avg_bottom = (bottom1["price"] + bottom2["price"]) / 2
                return {
                    "name": "Double Bottom (Çift Dip)",
                    "legs_total": 4,
                    "legs_found": len(recent),
                    "current_leg": len(recent),
                    "swing_points": recent,
                    "direction": "up",
                    "support": avg_bottom,
                    "remaining_pips_to_next_leg": price_diff * 0.5,
                    "stage_text": "Çift dip formasyonu - yukarı kırılım bekleniyor",
                }
    
    return None

def detect_triangle(swings: list) -> Optional[Dict]:
    """
    Triangle (Üçgen) formasyonu tespiti.
    Ascending, Descending, Symmetrical triangle'ları tespit eder.
    """
    if len(swings) < 4:
        return None
    
    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]
    
    if len(highs) < 2 or len(lows) < 2:
        return None
    
    # Son 3-4 tepe/dip al
    recent_highs = highs[-3:] if len(highs) >= 3 else highs
    recent_lows = lows[-3:] if len(lows) >= 3 else lows
    
    # Trend çizgilerinin eğimleri
    xh = np.array([h["idx"] for h in recent_highs])
    yh = np.array([h["price"] for h in recent_highs])
    xl = np.array([l["idx"] for l in recent_lows])
    yl = np.array([l["price"] for l in recent_lows])
    
    slope_h = np.polyfit(xh, yh, 1)[0] if len(xh) >= 2 else 0
    slope_l = np.polyfit(xl, yl, 1)[0] if len(xl) >= 2 else 0
    
    # Ascending Triangle: Yatay üst çizgi, yükselen alt çizgi
    if abs(slope_h) < 0.1 and slope_l > 0.1:
        all_pts = sorted(recent_highs + recent_lows, key=lambda s: s["idx"])
        return {
            "name": "Ascending Triangle (Yükselen Üçgen)",
            "legs_total": 5,
            "legs_found": len(all_pts),
            "current_leg": len(all_pts),
            "swing_points": all_pts[-5:] if len(all_pts) >= 5 else all_pts,
            "direction": "up",
            "resistance": float(np.mean(yh)),
            "remaining_pips_to_next_leg": abs(all_pts[-1]["price"] - np.mean(yh)) * 0.3,
            "stage_text": "Yükselen üçgen - yukarı kırılım bekleniyor",
        }
    
    # Descending Triangle: Düşen üst çizgi, yatay alt çizgi
    elif slope_h < -0.1 and abs(slope_l) < 0.1:
        all_pts = sorted(recent_highs + recent_lows, key=lambda s: s["idx"])
        return {
            "name": "Descending Triangle (Düşen Üçgen)",
            "legs_total": 5,
            "legs_found": len(all_pts),
            "current_leg": len(all_pts),
            "swing_points": all_pts[-5:] if len(all_pts) >= 5 else all_pts,
            "direction": "down",
            "support": float(np.mean(yl)),
            "remaining_pips_to_next_leg": abs(np.mean(yl) - all_pts[-1]["price"]) * 0.3,
            "stage_text": "Düşen üçgen - aşağı kırılım bekleniyor",
        }
    
    # Symmetrical Triangle: Her iki çizgi de birbirine yaklaşıyor
    elif (slope_h < -0.05 and slope_l > 0.05) or (slope_h > 0.05 and slope_l < -0.05):
        all_pts = sorted(recent_highs + recent_lows, key=lambda s: s["idx"])
        return {
            "name": "Symmetrical Triangle (Simetrik Üçgen)",
            "legs_total": 5,
            "legs_found": len(all_pts),
            "current_leg": len(all_pts),
            "swing_points": all_pts[-5:] if len(all_pts) >= 5 else all_pts,
            "direction": "either",
            "remaining_pips_to_next_leg": abs(all_pts[-1]["price"] - all_pts[-2]["price"]) * 0.5,
            "stage_text": "Simetrik üçgen - kırılım yönü belirsiz",
        }
    
    return None

def detect_wedge(swings: list, min_points: int = 5) -> Optional[Dict]:
    """
    Rising/Falling Wedge (Yükselen/Düşen Kama) tespiti.
    Üçgenden farkı: her iki trend çizgisi aynı yöne gider.
    """
    if len(swings) < min_points:
        return None
    
    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]
    
    if len(highs) < 2 or len(lows) < 2:
        return None
    
    # Son 3-4 tepe/dip al
    highs = highs[-3:]
    lows = lows[-3:]
    
    # Trend çizgilerinin eğimleri
    xh = np.array([h["idx"] for h in highs])
    yh = np.array([h["price"] for h in highs])
    xl = np.array([l["idx"] for l in lows])
    yl = np.array([l["price"] for l in lows])
    
    slope_h = np.polyfit(xh, yh, 1)[0] if len(xh) >= 2 else 0
    slope_l = np.polyfit(xl, yl, 1)[0] if len(xl) >= 2 else 0
    
    # Rising Wedge: Her ikisi de yukarı eğimli ama üst daha yatık
    if slope_h > 0 and slope_l > 0 and abs(slope_h) < abs(slope_l):
        all_pts = sorted(highs + lows, key=lambda s: s["idx"])
        return {
            "name": "Rising Wedge (Yükselen Kama)",
            "legs_total": 5,
            "legs_found": len(all_pts),
            "current_leg": len(all_pts),
            "swing_points": all_pts[-5:] if len(all_pts) >= 5 else all_pts,
            "direction": "down",  # Genelde aşağı kırılır
            "remaining_pips_to_next_leg": abs(all_pts[-1]["price"] - all_pts[-2]["price"]) * 0.5,
            "stage_text": "Yükselen kama - aşağı kırılım bekleniyor",
        }
    
    # Falling Wedge: Her ikisi de aşağı eğimli
    elif slope_h < 0 and slope_l < 0 and abs(slope_h) > abs(slope_l):
        all_pts = sorted(highs + lows, key=lambda s: s["idx"])
        return {
            "name": "Falling Wedge (Düşen Kama)",
            "legs_total": 5,
            "legs_found": len(all_pts),
            "current_leg": len(all_pts),
            "swing_points": all_pts[-5:] if len(all_pts) >= 5 else all_pts,
            "direction": "up",  # Genelde yukarı kırılır
            "remaining_pips_to_next_leg": abs(all_pts[-1]["price"] - all_pts[-2]["price"]) * 0.5,
            "stage_text": "Düşen kama - yukarı kırılım bekleniyor",
        }
    
    return None


def detect_flag_pennant(swings: list, df: pd.DataFrame) -> Optional[Dict]:
    """
    Flag (Bayrak) ve Pennant (Flama) formasyonu tespiti.
    Güçlü bir hareket + kısa konsolidasyon dönemi.
    """
    if len(swings) < 4:
        return None
    
    # Son 4-6 swing'i al
    recent = swings[-6:] if len(swings) >= 6 else swings
    
    # İlk güçlü hareketi tespit et (flagpole)
    if len(recent) >= 3:
        first_move = recent[1]["price"] - recent[0]["price"]
        
        # Sonraki swinglerin dar bir aralıkta olup olmadığını kontrol et
        consolidation_swings = recent[2:]
        prices = [s["price"] for s in consolidation_swings]
        range_size = max(prices) - min(prices)
        
        # Bayrak/Flama tespiti: konsolidasyon aralığı ilk hareketin %50'sinden az
        if abs(range_size) < abs(first_move) * 0.5:
            pattern_name = "Flag (Bayrak)" if len(consolidation_swings) >= 3 else "Pennant (Flama)"
            direction = "up" if first_move > 0 else "down"
            
            return {
                "name": pattern_name,
                "legs_total": 4,
                "legs_found": len(recent),
                "current_leg": len(recent),
                "swing_points": recent,
                "direction": direction,
                "remaining_pips_to_next_leg": abs(first_move) * 0.3,
                "stage_text": f"Konsolidasyon tamamlanıyor, {direction} yönlü devam bekleniyor",
            }
    
    return None


def detect_cup_handle(swings: list, min_cup_depth: float = 30.0) -> Optional[Dict]:
    """
    Cup and Handle (Fincan-Kulp) formasyonu tespiti.
    U şeklinde dip + küçük düzeltme (kulp).
    """
    if len(swings) < 6:
        return None
    
    # U şekli arayalım (H -> L -> gradual recovery -> H)
    for i in range(len(swings) - 5):
        segment = swings[i:i+6]
        
        # İlk ve son yüksek noktalar
        if segment[0]["type"] == "H" and segment[-1]["type"] == "H":
            # Ortada dip var mı?
            lows_in_middle = [s for s in segment[1:-1] if s["type"] == "L"]
            if lows_in_middle:
                deepest = min(lows_in_middle, key=lambda x: x["price"])
                cup_depth = segment[0]["price"] - deepest["price"]
                
                if cup_depth >= min_cup_depth:
                    # Son iki swing kulp olabilir mi?
                    if len(swings) > i+6 and abs(segment[-1]["price"] - segment[0]["price"]) < cup_depth * 0.3:
                        return {
                            "name": "Cup and Handle (Fincan-Kulp)",
                            "legs_total": 7,
                            "legs_found": len(segment),
                            "current_leg": len(segment),
                            "swing_points": segment,
                            "direction": "up",
                            "remaining_pips_to_next_leg": cup_depth * 0.2,
                            "stage_text": "Kulp formasyonu tamamlanıyor, yukarı kırılım bekleniyor",
                        }
    
    return None


def detect_rectangle(swings: list, tolerance: float = 20.0) -> Optional[Dict]:
    """
    Rectangle (Dikdörtgen) konsolidasyon formasyonu.
    Fiyat yatay destek-direnç arasında hareket eder.
    """
    if len(swings) < 4:
        return None
    
    highs = [s["price"] for s in swings if s["type"] == "H"]
    lows = [s["price"] for s in swings if s["type"] == "L"]
    
    if len(highs) >= 2 and len(lows) >= 2:
        # Tepeler ve dipler yakın seviyelerde mi?
        high_range = max(highs) - min(highs)
        low_range = max(lows) - min(lows)
        
        if high_range <= tolerance and low_range <= tolerance:
            avg_high = np.mean(highs)
            avg_low = np.mean(lows)
            
            return {
                "name": "Rectangle (Dikdörtgen Konsolidasyon)",
                "legs_total": 6,
                "legs_found": len(swings),
                "current_leg": len(swings),
                "swing_points": swings[-6:] if len(swings) >= 6 else swings,
                "resistance": avg_high,
                "support": avg_low,
                "direction": "either",
                "remaining_pips_to_next_leg": (avg_high - avg_low) * 0.3,
                "stage_text": f"Yatay konsolidasyon: {avg_low:.1f} - {avg_high:.1f} aralığında",
            }
    
    return None


def detect_diamond(swings: list) -> Optional[Dict]:
    """
    Diamond (Elmas) formasyonu - nadir ama güçlü dönüş formasyonu.
    Önce genişleyen sonra daralan volatilite.
    """
    if len(swings) < 7:
        return None
    
    # İlk yarı: genişleyen aralık
    first_half = swings[:4]
    second_half = swings[3:7]
    
    # Volatilite hesapla
    first_prices = [s["price"] for s in first_half]
    second_prices = [s["price"] for s in second_half]
    
    first_vol = np.std(first_prices)
    second_vol = np.std(second_prices)
    
    # Elmas: önce genişleyen sonra daralan
    if first_vol > second_vol * 1.5:
        return {
            "name": "Diamond (Elmas)",
            "legs_total": 7,
            "legs_found": len(swings[:7]),
            "current_leg": 7,
            "swing_points": swings[:7],
            "direction": "reversal",
            "remaining_pips_to_next_leg": first_vol,
            "stage_text": "Elmas formasyonu - güçlü dönüş sinyali",
        }
    
    return None


def detect_rounding_pattern(swings: list, min_swings: int = 7) -> Optional[Dict]:
    """
    Rounding Bottom/Top (Yuvarlak Dip/Tepe) tespiti.
    """
    if len(swings) < min_swings:
        return None
    
    prices = [s["price"] for s in swings]
    indices = list(range(len(prices)))
    
    # Parabolik uyum dene (2. derece polinom)
    coeffs = np.polyfit(indices, prices, 2)
    a, b, c = coeffs
    
    # a > 0: U şekli (rounding bottom)
    # a < 0: Ters U (rounding top)
    if abs(a) > 0.001:  # Anlamlı eğrilik var
        fitted = np.poly1d(coeffs)
        residuals = [abs(prices[i] - fitted(i)) for i in indices]
        avg_residual = np.mean(residuals)
        
        # İyi uyum varsa
        if avg_residual < np.std(prices) * 0.3:
            if a > 0:
                return {
                    "name": "Rounding Bottom (Yuvarlak Dip)",
                    "legs_total": 8,
                    "legs_found": len(swings),
                    "current_leg": len(swings),
                    "swing_points": swings[-8:] if len(swings) >= 8 else swings,
                    "direction": "up",
                    "remaining_pips_to_next_leg": abs(prices[-1] - min(prices)) * 0.2,
                    "stage_text": "Yuvarlak dip formasyonu - yükseliş başlıyor",
                }
            else:
                return {
                    "name": "Rounding Top (Yuvarlak Tepe)",
                    "legs_total": 8,
                    "legs_found": len(swings),
                    "current_leg": len(swings),
                    "swing_points": swings[-8:] if len(swings) >= 8 else swings,
                    "direction": "down",
                    "remaining_pips_to_next_leg": abs(max(prices) - prices[-1]) * 0.2,
                    "stage_text": "Yuvarlak tepe formasyonu - düşüş başlıyor",
                }
    
    return None


# ==================== KÜÇÜK MUM FORMASYONLARI ====================

def detect_candlestick_patterns(df: pd.DataFrame, 
                               open_col: str = "open",
                               high_col: str = "high", 
                               low_col: str = "low",
                               close_col: str = "close") -> Dict[str, Any]:
    """
    Gelişmiş mum formasyonu tespiti.
    Son 3 mumu analiz eder.
    """
    if len(df) < 3:
        return {"pattern": "Yetersiz veri", "strength": 0}
    
    # Son 3 mumu al
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Mum özellikleri
    def candle_features(row):
        o, h, l, c = float(row[open_col]), float(row[high_col]), float(row[low_col]), float(row[close_col])
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l + 1e-9
        is_bullish = c > o
        body_ratio = body / total_range
        return {
            'open': o, 'high': h, 'low': l, 'close': c,
            'body': body, 'upper_wick': upper_wick, 'lower_wick': lower_wick,
            'range': total_range, 'is_bullish': is_bullish, 'body_ratio': body_ratio
        }
    
    c0 = candle_features(curr)
    c1 = candle_features(prev1)
    c2 = candle_features(prev2)
    
    # Formasyon tespiti (öncelik sırasına göre)
    
    # 1. MORNING/EVENING STAR (3 mum)
    if not c2['is_bullish'] and c2['body_ratio'] > 0.6:  # Güçlü kırmızı
        if c1['body_ratio'] < 0.3:  # Küçük gövdeli (star)
            if c0['is_bullish'] and c0['body_ratio'] > 0.6:  # Güçlü yeşil
                if c0['close'] > c2['open'] * 0.5 + c2['close'] * 0.5:
                    return {
                        "pattern": "🌟 Morning Star",
                        "description": "Güçlü dip dönüş formasyonu - 3 mumlu",
                        "direction": "bullish",
                        "strength": 9
                    }
    
    if c2['is_bullish'] and c2['body_ratio'] > 0.6:
        if c1['body_ratio'] < 0.3:
            if not c0['is_bullish'] and c0['body_ratio'] > 0.6:
                if c0['close'] < c2['close'] * 0.5 + c2['open'] * 0.5:
                    return {
                        "pattern": "🌙 Evening Star",
                        "description": "Güçlü tepe dönüş formasyonu - 3 mumlu",
                        "direction": "bearish",
                        "strength": 9
                    }
    
    # 2. THREE WHITE SOLDIERS / THREE BLACK CROWS
    if c2['is_bullish'] and c1['is_bullish'] and c0['is_bullish']:
        if c0['close'] > c1['close'] > c2['close']:
            if c0['open'] > c1['open'] > c2['open']:
                return {
                    "pattern": "⚔️ Three White Soldiers",
                    "description": "Güçlü yükseliş devamı - 3 ardışık yeşil",
                    "direction": "bullish",
                    "strength": 8
                }
    
    if not c2['is_bullish'] and not c1['is_bullish'] and not c0['is_bullish']:
        if c0['close'] < c1['close'] < c2['close']:
            if c0['open'] < c1['open'] < c2['open']:
                return {
                    "pattern": "🦅 Three Black Crows",
                    "description": "Güçlü düşüş devamı - 3 ardışık kırmızı",
                    "direction": "bearish",
                    "strength": 8
                }
    
    # 3. HARAMI (2 mum)
    if c1['body'] > c0['body'] * 2:
        if c1['is_bullish'] and not c0['is_bullish']:
            if c0['open'] < c1['close'] and c0['close'] > c1['open']:
                return {
                    "pattern": "📦 Bearish Harami",
                    "description": "Yükselişte dönüş sinyali - içerde küçük kırmızı",
                    "direction": "bearish",
                    "strength": 6
                }
        elif not c1['is_bullish'] and c0['is_bullish']:
            if c0['open'] > c1['close'] and c0['close'] < c1['open']:
                return {
                    "pattern": "📦 Bullish Harami",
                    "description": "Düşüşte dönüş sinyali - içerde küçük yeşil",
                    "direction": "bullish",
                    "strength": 6
                }
    
    # 4. PIERCING LINE / DARK CLOUD COVER
    if not c1['is_bullish'] and c0['is_bullish']:
        if c0['open'] < c1['low'] and c0['close'] > c1['open'] + c1['body'] * 0.5:
            return {
                "pattern": "⚡ Piercing Line",
                "description": "Düşüşte güçlü dönüş - yarıdan fazla kapanış",
                "direction": "bullish",
                "strength": 7
            }
    
    if c1['is_bullish'] and not c0['is_bullish']:
        if c0['open'] > c1['high'] and c0['close'] < c1['close'] - c1['body'] * 0.5:
            return {
                "pattern": "☁️ Dark Cloud Cover",
                "description": "Yükselişte güçlü dönüş - yarıdan fazla düşüş",
                "direction": "bearish",
                "strength": 7
            }
    
    # 5. TWEEZER TOP/BOTTOM
    if abs(c1['high'] - c0['high']) < c0['range'] * 0.1:
        if c1['is_bullish'] and not c0['is_bullish']:
            return {
                "pattern": "🔧 Tweezer Top",
                "description": "Çift tepe - direnç onayı",
                "direction": "bearish",
                "strength": 5
            }
    
    if abs(c1['low'] - c0['low']) < c0['range'] * 0.1:
        if not c1['is_bullish'] and c0['is_bullish']:
            return {
                "pattern": "🔧 Tweezer Bottom",
                "description": "Çift dip - destek onayı",
                "direction": "bullish",
                "strength": 5
            }
    
    # 6. DOJI ÇEŞİTLERİ
    if c0['body_ratio'] < 0.1:
        if c0['lower_wick'] > c0['upper_wick'] * 3:
            return {
                "pattern": "🦋 Dragonfly Doji",
                "description": "Güçlü alım baskısı - uzun alt fitil",
                "direction": "bullish",
                "strength": 6
            }
        elif c0['upper_wick'] > c0['lower_wick'] * 3:
            return {
                "pattern": "🪦 Gravestone Doji",
                "description": "Güçlü satış baskısı - uzun üst fitil",
                "direction": "bearish",
                "strength": 6
            }
        else:
            return {
                "pattern": "⚪ Standard Doji",
                "description": "Kararsızlık - trend dönüşü olabilir",
                "direction": "neutral",
                "strength": 4
            }
    
    # 7. MARUBOZU
    if c0['body_ratio'] > 0.95:
        if c0['is_bullish']:
            return {
                "pattern": "🟩 Bullish Marubozu",
                "description": "Çok güçlü alım - fitilsiz yeşil",
                "direction": "bullish",
                "strength": 8
            }
        else:
            return {
                "pattern": "🟥 Bearish Marubozu",
                "description": "Çok güçlü satış - fitilsiz kırmızı",
                "direction": "bearish",
                "strength": 8
            }
    
    # 8. SPINNING TOP
    if 0.2 < c0['body_ratio'] < 0.4:
        if c0['upper_wick'] > c0['body'] and c0['lower_wick'] > c0['body']:
            return {
                "pattern": "🌀 Spinning Top",
                "description": "Kararsızlık - küçük gövde, uzun fitiller",
                "direction": "neutral",
                "strength": 3
            }
    
    # Temel formasyonlar (mevcut kodunuzdan)
    if c0['lower_wick'] > c0['body'] * 2 and c0['upper_wick'] < c0['body'] * 1.2 and c0['is_bullish']:
        return {
            "pattern": "🔨 Hammer",
            "description": "Dipte güçlenme sinyali",
            "direction": "bullish",
            "strength": 6
        }
    
    if c0['upper_wick'] > c0['body'] * 2 and c0['lower_wick'] < c0['body'] * 1.2 and not c0['is_bullish']:
        return {
            "pattern": "🌠 Shooting Star",
            "description": "Tepede zayıflama sinyali",
            "direction": "bearish",
            "strength": 6
        }
    
    # Engulfing (mevcut kodunuzdan geliştirilmiş)
    if c1['is_bullish'] != c0['is_bullish']:
        if not c1['is_bullish'] and c0['is_bullish']:
            if c0['close'] >= c1['open'] and c0['open'] <= c1['close']:
                return {
                    "pattern": "🟢 Bullish Engulfing",
                    "description": "Güçlü dip dönüş sinyali",
                    "direction": "bullish",
                    "strength": 7
                }
        elif c1['is_bullish'] and not c0['is_bullish']:
            if c0['close'] <= c1['open'] and c0['open'] >= c1['close']:
                return {
                    "pattern": "🔴 Bearish Engulfing",
                    "description": "Güçlü tepe dönüş sinyali",
                    "direction": "bearish",
                    "strength": 7
                }
    
    return {
        "pattern": "📊 Standart Mum",
        "description": "Belirgin formasyon yok",
        "direction": "neutral",
        "strength": 0
    }


def detect_structural_patterns_enhanced(df_for_pattern: pd.DataFrame) -> List[Dict]:
    """
    Geliştirilmiş formasyon algılama - tüm büyük formasyonları kontrol eder.
    """
    # CLOSE_COL kullan (H1_close)
    price_series = df_for_pattern[CLOSE_COL].astype(float)
    swings = detect_swings_from_price(price_series, left=3, right=3, min_prominence=20.0)
    
    if len(swings) < 3:
        return []
    
    patterns = []
    
    # Tüm formasyonları kontrol et (öncelik sırasına göre)
    
    # 1. Baş-Omuz (en güvenilir)
    hs = detect_head_shoulders(swings)
    if hs is not None:
        hs["priority"] = 10
        patterns.append(hs)
    
    # 2. Çift Tepe/Dip
    dtb = detect_double_top_bottom(swings)
    if dtb is not None:
        dtb["priority"] = 9
        patterns.append(dtb)
    
    # 3. Fincan-Kulp
    cup = detect_cup_handle(swings)
    if cup is not None:
        cup["priority"] = 8
        patterns.append(cup)
    
    # 4. Kama formasyonları
    wedge = detect_wedge(swings)
    if wedge is not None:
        wedge["priority"] = 7
        patterns.append(wedge)
    
    # 5. Bayrak/Flama
    flag = detect_flag_pennant(swings, df_for_pattern)
    if flag is not None:
        flag["priority"] = 6
        patterns.append(flag)
    
    # 6. Dikdörtgen
    rect = detect_rectangle(swings)
    if rect is not None:
        rect["priority"] = 5
        patterns.append(rect)
    
    # 7. Üçgen
    tri = detect_triangle(swings)
    if tri is not None:
        tri["priority"] = 4
        patterns.append(tri)
    
    # 8. Elmas
    diamond = detect_diamond(swings)
    if diamond is not None:
        diamond["priority"] = 8
        patterns.append(diamond)
    
    # 9. Yuvarlak formasyonlar
    rounding = detect_rounding_pattern(swings)
    if rounding is not None:
        rounding["priority"] = 6
        patterns.append(rounding)
    
    # Öncelik ve zamana göre sırala
    patterns.sort(key=lambda p: (p.get("priority", 0), -p["swing_points"][-1]["idx"]), reverse=True)
    
    return patterns
def detect_structural_patterns(df_for_pattern: pd.DataFrame):
    """
    Geriye dönük uyumluluk için alias:
    Eski 'detect_structural_patterns' çağrıları artık
    geliştirilen 'detect_structural_patterns_enhanced' fonksiyonuna yönlenir.
    """
    return detect_structural_patterns_enhanced(df_for_pattern)

# ==================== STREAMLIT UI KODU ====================

def display_pattern_analysis(df_graph, df_plot, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL):
    """
    Geliştirilmiş formasyon analizi UI.
    """
    st.markdown("<hr style='opacity:0.2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🧩 Büyük Formasyon Analizi", unsafe_allow_html=True)
    
    patterns = detect_structural_patterns_enhanced(df_graph)
    
    if patterns:
        # En önemli formasyonu göster
        main_pat = patterns[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Ana Formasyon</div>
                    <div class="metric-value" style="color: {'#10b981' if main_pat.get('direction') == 'up' else '#ef4444'}">
                        {main_pat['name']}
                    </div>
                    <div class="explanation">
                        • Ayak: {main_pat.get('legs_found', 'N/A')}/{main_pat.get('legs_total', 'N/A')}<br>
                        • Yön: {main_pat.get('direction', 'belirsiz')}<br>
                        • {main_pat.get('stage_text', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if len(patterns) > 1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Diğer Formasyonlar</div>
                        <div class="explanation">
                            {'<br>'.join([f"• {p['name']}" for p in patterns[1:3]])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                remaining = main_pat.get('remaining_pips_to_next_leg', None)
                if remaining:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Kritik Seviye</div>
                            <div class="metric-value">{remaining:.1f} puan</div>
                            <div class="explanation">Sonraki ayağa mesafe</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("📊 Şu an belirgin bir büyük formasyon algılanmadı.")
    
    # Mum formasyonu analizi
    st.markdown("<hr style='opacity:0.2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🕯 Mum Formasyonu Analizi", unsafe_allow_html=True)
    
    candle_result = detect_candlestick_patterns(df_plot, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL)
    
    # Güç göstergesi
    strength_bar = "█" * candle_result['strength'] + "░" * (10 - candle_result['strength'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Formasyon</div>
                <div class="metric-value">{candle_result['pattern']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        direction_color = {
            'bullish': '#10b981',
            'bearish': '#ef4444',
            'neutral': '#6b7280'
        }.get(candle_result['direction'], '#6b7280')
        
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sinyal</div>
                <div class="metric-value" style="color: {direction_color}">
                    {candle_result['direction'].upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Güç</div>
                <div class="metric-value">{strength_bar}</div>
                <div class="explanation">{candle_result['strength']}/10</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="explanation">
                <strong>Açıklama:</strong> {candle_result['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
def detect_simple_trend(df: pd.DataFrame, window: int = 80):
    """Şimdilik basit trend yorumu (son window bar üzerinden)."""
    if len(df) < window + 5:
        return "Veri kısa"
    
    closes = df[CLOSE_COL].tail(window).values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]

    if slope > 0:
        return "Yükselen trend (son ~%d bar)" % window
    elif slope < 0:
        return "Düşen trend (son ~%d bar)" % window
    else:
        return "Yatay / nötr trend"


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"

def run_trade_simulation(
    df: pd.DataFrame,
    model_bundle,
    thr_buy: float,
    thr_sell: float,
    horizon_bars: int = 10,
    conf_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Model sinyallerine göre trade simülasyonu.
    - Sadece AL / SAT + yeterli model güveni olan barlarda işlem açar
    - TP / SL puan bazlı
    """

    trades = []

    # Çok erken barlarda indikatörler NaN olabiliyor
    start_idx = 50

    # TP / SL (puan)
    TP_POINTS = 25.0
    SL_POINTS = 25.0

    for i in range(start_idx, len(df) - horizon_bars):
        row = df.iloc[i]
        signal = get_model_signal(row, model_bundle, thr_buy, thr_sell)

        # Label ve güven
        label = signal.get("label", "PAS")
        confidence = float(signal.get("confidence", 0.0))

        # 1) Güven filtresi
        if confidence < conf_threshold:
            continue

        # 2) Yön filtresi
        if label == "AL":
            direction = 1
        elif label == "SAT":
            direction = -1
        else:
            continue  # PAS veya saçma bir şeyse işlem açma

        entry_price = float(row[CLOSE_COL])
        entry_time = row[TIME_COL]

        # Gelecek barlar (horizon)
        future = df.iloc[i + 1 : i + 1 + horizon_bars].copy()
        if future.empty:
            break

        prices = future[CLOSE_COL].astype(float).to_numpy()
        times = future[TIME_COL].to_list()

        # ========== TP / SL HESABI ==========
        tp = entry_price + TP_POINTS * direction
        sl = entry_price - SL_POINTS * direction

        exit_idx = len(prices) - 1  # default: horizon sonunda
        exit_price = float(prices[-1])
        exit_time = times[-1]
        final_pnl = float((exit_price - entry_price) * direction)

        # Fiyat yolunu gez ve ilk TP/SL vuran yeri bul
        for j, price in enumerate(prices):
            price = float(price)

            hit_tp = (direction == 1 and price >= tp) or (direction == -1 and price <= tp)
            hit_sl = (direction == 1 and price <= sl) or (direction == -1 and price >= sl)

            if hit_tp:
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = TP_POINTS
                break
            elif hit_sl:
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = -SL_POINTS
                break

        # Tüm horizon için PnL serisi (analiz için)
        pnl_series_full = (prices - entry_price) * direction
        max_fav = float(pnl_series_full.max())
        max_adv = float(pnl_series_full.min())

        # Pozisyon açıkken PnL
        pnl_series_open = pnl_series_full[: exit_idx + 1]
        bars_in_profit = int((pnl_series_open > 0).sum())
        bars_in_loss = int((pnl_series_open < 0).sum())
        minutes_in_profit = bars_in_profit * 30
        minutes_in_loss = bars_in_loss * 30

        trades.append(
            {
                "entry_idx": i,
                "entry_time": entry_time,
                "signal_label": label,
                "p_up": signal.get("p_up", None),
                "p_down": signal.get("p_down", None),
                "confidence": confidence,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "holding_bars": exit_idx + 1,
                "holding_minutes": (exit_idx + 1) * 30,
                "pnl_points": final_pnl,
                "max_favorable": max_fav,
                "max_adverse": max_adv,
                "minutes_in_profit": minutes_in_profit,
                "minutes_in_loss": minutes_in_loss,
            }
        )

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    return trades_df
    """
    Modeli sanki canlı çalışıyormuş gibi tarihsel veride ilerletip
    AL / SAT sinyallerinde sanal işlem açar ve sonuçlarını kayıt altına alır. """

def backtest_trades(
    df: pd.DataFrame,
    model_bundle,
    thr_buy: float,
    thr_sell: float,
    horizon_bars: int = 10,
) -> pd.DataFrame:
    """
    AI sinyaline göre trade simülasyonu.
    - Güven filtresi uygular (confidence >= CONF_TH)
    - Sadece AL / SAT sinyallerini alır
    - TP / SL tabanlı çıkış kullanır (puan cinsinden)
    """

    # ==== PARAMETRELER ====
    CONF_TH = 0.65      # minimum güven eşiği
    TP_POINTS = 25.0    # hedef kâr (puan)
    SL_POINTS = 25.0    # maksimum zarar (puan)

    trades = []

    # Çok erken barlarda bazı indikatörler NaN olabilir, o yüzden biraz kenara çekilelim
    start_idx = 50

    for i in range(start_idx, len(df) - horizon_bars):
        row = df.iloc[i]
        signal = get_model_signal(row, model_bundle, thr_buy, thr_sell)

        # =========================
        # 1) GÜVEN FİLTRESİ
        # =========================
        confidence = float(signal.get("confidence", 0.0))

        # Güveni düşükse hiç trade açma
        if confidence < CONF_TH:
            continue

        # Yönsüz / pas sinyal ise trade açma
        label = signal["label"]
        if label == "PAS":
            continue

        # Beklenmeyen bir label gelirse de trade açma
        if label == "AL":
            direction = 1
        elif label == "SAT":
            direction = -1
        else:
            continue

        # =========================
        # 2) GİRİŞ BİLGİLERİ
        # =========================
        entry_price = float(row[CLOSE_COL])
        entry_time = row[TIME_COL]

        future = df.iloc[i + 1 : i + 1 + horizon_bars].copy()
        if future.empty:
            break

        prices = future[CLOSE_COL].astype(float).to_numpy()
        times = future[TIME_COL].to_list()

        # ================== TP/SL TABANLI ÇIKIŞ MANTIĞI ==================

        # Long için: TP = entry + 25, SL = entry - 25
        # Short için: TP = entry - 25, SL = entry + 25
        tp = entry_price + TP_POINTS * direction
        sl = entry_price - SL_POINTS * direction

        exit_idx = len(prices) - 1  # default: horizon sonunda kapanır
        exit_price = float(prices[-1])
        exit_time = times[-1]
        final_pnl = float((exit_price - entry_price) * direction)

        hit_tp = False
        hit_sl = False

        # Fiyat yolunu gez ve ilk TP/SL vuran yeri bul
        for j, price in enumerate(prices):
            price = float(price)

            hit_tp_now = (direction == 1 and price >= tp) or (direction == -1 and price <= tp)
            hit_sl_now = (direction == 1 and price <= sl) or (direction == -1 and price >= sl)

            if hit_tp_now:
                hit_tp = True
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = TP_POINTS
                break
            elif hit_sl_now:
                hit_sl = True
                exit_idx = j
                exit_price = price
                exit_time = times[j]
                final_pnl = -SL_POINTS
                break

        # Tüm horizon için potansiyel max/min PnL (analiz için)
        pnl_series_full = (prices - entry_price) * direction
        max_favorable = float(pnl_series_full.max())
        max_adverse = float(pnl_series_full.min())

        # Pozisyon açıkken geçen süre için PnL serisi
        pnl_series_open = pnl_series_full[: exit_idx + 1]
        bars_in_profit = int((pnl_series_open > 0).sum())
        bars_in_loss = int((pnl_series_open < 0).sum())
        minutes_in_profit = bars_in_profit * 30
        minutes_in_loss = bars_in_loss * 30

        trades.append(
            {
                "entry_idx": i,
                "entry_time": entry_time,
                "signal_label": label,
                "p_up": signal["p_up"],
                "p_down": signal["p_down"],
                "confidence": confidence,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "holding_bars": exit_idx + 1,
                "holding_minutes": (exit_idx + 1) * 30,
                "pnl_points": final_pnl,
                "max_favorable": max_favorable,
                "max_adverse": max_adverse,
                "minutes_in_profit": minutes_in_profit,
                "minutes_in_loss": minutes_in_loss,
                "hit_tp": hit_tp,
                "hit_sl": hit_sl,
            }
        )

    trades_df = pd.DataFrame(trades)
    return trades_df# ================== UYGULAMA GÖVDESİ ==================
def label_trades_for_training(
    trades_df: pd.DataFrame,
    good_pnl: float = 30.0,
    bad_pnl: float = 0.0,
) -> pd.DataFrame:
    """
    Trade sonuçlarından 'iyi sinyal / kötü sinyal' label'ı üretir.

    pnl_points zaten direction-normalized:
      - AL'de yukarı gidince +
      - SAT'ta aşağı gidince +
    Yani:
      pnl_points >= good_pnl  → 1 (iyi sinyal)
      pnl_points <= bad_pnl   → 0 (kötü sinyal)
      aradaki bölge           → NaN (belirsiz, eğitimden drop)
    """
    df_lab = trades_df.copy()

    df_lab["label_good"] = np.where(
        df_lab["pnl_points"] >= good_pnl,
        1,
        np.where(
            df_lab["pnl_points"] <= bad_pnl,
            0,
            np.nan,  # ne çok iyi ne çok kötü, eğitimde kullanmayız
        ),
    )

    # Tamamen NaN olanları elemek için:
    df_lab = df_lab.dropna(subset=["label_good"])
    df_lab["label_good"] = df_lab["label_good"].astype(int)

    return df_lab

st.markdown('<div class="main-header">📈 NASDAQ M30 AI TRADING PANEL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Canlı fiyat, trend kanalları, destek/direnç, EMA mesafeleri ve AI sinyali tek ekranda</div>',
    unsafe_allow_html=True,
)

# --------- Sidebar: Ayarlar ---------
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    
    # Grafik Ayarları
    st.markdown("#### 📈 Grafik Ayarları")
    timeframe = st.selectbox(
        "⏱️ Zaman Aralığı",
        ['5m', '15m', '30m', '1h', '2h', '4h'],
        index=2,  # Default: 30m
        help="Ham veriyi bu zaman dilimine göre yeniden örnekler"
    )
    
    num_candles = st.slider(
        "📊 Gösterilecek Mum Sayısı", 
        min_value=50, 
        max_value=500, 
        value=150,
        help="Grafikte gösterilecek maksimum mum sayısı"
    )
    
    # Model Ayarları
    st.markdown("---")
    st.markdown("#### 🤖 Model Ayarları")
    auto_refresh = st.checkbox("🔄 Otomatik yenile (10 sn)", value=False)
    thr_buy = st.slider("AL eşiği (p_up)", min_value=0.50, max_value=0.80, value=THR_BUY, step=0.01)
    thr_sell = st.slider("SAT eşiği (p_up)", min_value=0.20, max_value=0.50, value=THR_SELL, step=0.01)

    st.markdown("---")
    st.caption(f"**Aktif Timeframe:** {timeframe.upper()}")
    st.caption("**Kaynak:** yfinance (ücretsiz)")
    
    # Ekonomik takvim widget'ı
    create_news_widget()

if auto_refresh:
    # 10 saniyede bir sayfayı yenile
    st.experimental_rerun()

# --------- Veri ve model yükleme ---------
# ✅ Canlı veri yükleme (yfinance ile)
df_raw = load_finnhub_data()

if df_raw.empty:
    st.error("Veri seti boş görünüyor.")
    st.stop()

# ✅ Timeframe'e göre resample et
st.info(f"🔄 Veri {timeframe.upper()} timeframe'ine dönüştürülüyor...")
df = resample_data(df_raw, timeframe)

if df.empty:
    st.error(f"❌ {timeframe} timeframe'ine dönüştürme başarısız.")
    st.stop()

st.success(f"✅ {len(df)} adet {timeframe.upper()} mumu hazır!")

# ✅ Son N mumu al
df = df.tail(num_candles * 2)  # x2 çünkü indicator'lar için daha fazla gerekli

# ================== BÜYÜK ANİMASYONLU MARKET DATA HEADER ==================
# Son fiyat ve değişim hesapla
if len(df) >= 2:
    current_price = df[CLOSE_COL].iloc[-1]
    prev_price = df[CLOSE_COL].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Timestamp
    last_update = df[TIME_COL].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    
    # Renk belirleme
    change_class = "change-positive" if price_change >= 0 else "change-negative"
    change_symbol = "▲" if price_change >= 0 else "▼"
    
    st.markdown(f"""
    <div class="market-header">
        <div>
            <div class="market-title">NASDAQ-100 INDEX (^NDX) • {timeframe.upper()}</div>
            <div class="market-price">${current_price:,.2f}</div>
            <div class="market-change {change_class}">
                {change_symbol} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
            </div>
        </div>
        <div class="market-info">
            <div style="font-size: 18px; color: #8b949e; margin-bottom: 8px;">
                <strong>VOL:</strong> {df[VOL_COL].iloc[-1]:,.0f}
            </div>
            <div style="font-size: 18px; color: #8b949e; margin-bottom: 8px;">
                <strong>24H High:</strong> ${df[HIGH_COL].tail(48).max():,.2f}
            </div>
            <div style="font-size: 18px; color: #8b949e;">
                <strong>24H Low:</strong> ${df[LOW_COL].tail(48).min():,.2f}
            </div>
            <div class="market-timestamp">📡 Son Güncelleme: {last_update}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

model_bundle = load_model_bundle(MODEL_PATH)

# ============ DEBUG PANELİNİ SIDEBAR'A EKLE (VERİ YÜKLENDIKTEN SONRA) ============
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Debug Paneli")
    
    # Veri bilgileri
    st.metric("Toplam Bar", len(df))
    st.metric("Son Tarih", df[TIME_COL].iloc[-1].strftime("%Y-%m-%d %H:%M") if len(df) > 0 else "N/A")
    st.metric("İlk Tarih", df[TIME_COL].iloc[0].strftime("%Y-%m-%d %H:%M") if len(df) > 0 else "N/A")
    
    # Model bilgisi
    model_count = len(model_bundle.get("ensemble", []))
    st.metric("Model Sayısı", model_count)
    
    # yfinance API test
    if st.checkbox("🔍 API Test (yfinance)"):
        try:
            ticker = yf.Ticker('^NDX')
            info = ticker.history(period='1d', interval='1m').tail(1)
            if not info.empty:
                last_price = info['Close'].iloc[0]
                st.success(f"✅ Son fiyat: ${last_price:.2f}")
                st.json({
                    "symbol": "^NDX (NASDAQ-100)",
                    "last_price": f"{last_price:.2f}",
                    "timestamp": str(info.index[0]),
                    "source": "yfinance (ücretsiz)"
                })
            else:
                st.error("Veri alınamadı")
        except Exception as e:
            st.error(f"API Hatası: {e}")

# Son N bar grafik için (kullanıcı ayarına göre)
# Önce daha fazla veri al (temizlik sonrası yeterli olsun diye)
df_plot = df.copy()
df_graph = df_plot.tail(num_candles * 2).copy()  # x2 çünkü dropna/drop_duplicates sonrası azalabilir
df_graph = df_graph.drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)
df_graph["bar_idx"] = range(len(df_graph))

# 🔍 Emniyet için iloc kullan
last_row = df.iloc[-1]  # Tam olarak son satır
last_price = float(last_row[CLOSE_COL])

# Debug: Eğer hata alırsak alternatif
# --- Formasyon analizi için hazırlık --- #
small_patterns = detect_candlestick_patterns(
    df_graph,
    open_col=OPEN_COL,
    high_col=HIGH_COL,
    low_col=LOW_COL,
    close_col=CLOSE_COL,
)

flag_info = detect_flag_pattern(df_graph, close_col=CLOSE_COL)

swing_high_idx, swing_low_idx = find_swings(
    df_graph,
    high_col=HIGH_COL,
    low_col=LOW_COL,
)
# Model sinyali
signal = get_model_signal(last_row, model_bundle, thr_buy=thr_buy, thr_sell=thr_sell)

# Trend kanalı
upper_ch, middle_ch, lower_ch = calculate_trend_channels(df_graph)

# Destek / direnç
supports = find_zones_price_action(
    df_graph,
    price_col=CLOSE_COL,
    last_price=last_price,
    side="support",
    zone_half_width=25.0,  # toplam 50 puan
    min_touches=3,
    top_k=3,
)

resistances = find_zones_price_action(
    df_graph,
    price_col=CLOSE_COL,
    last_price=last_price,
    side="resistance",
    zone_half_width=25.0,
    min_touches=3,
    top_k=3,
)

trend_text = detect_simple_trend(df_plot)


# ================== LAYOUT: ÜST KISIM ==================
top_left, top_mid, top_right = st.columns([1.2, 2.4, 1.2])

# --------- SOL: MODEL SİNYAL KUTUSU ---------
with top_left:
    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI Model Sinyali")

    if signal["label"] == "AL":
        st.markdown('<div class="buy-signal">🟢 AL</div>', unsafe_allow_html=True)
    elif signal["label"] == "SAT":
        st.markdown('<div class="sell-signal">🔴 SAT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="neutral-signal">⚪ PAS GEÇ</div>', unsafe_allow_html=True)

    # --- Model güveni (raw confidence) ---
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Model güveni</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{format_pct(signal["confidence"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Yukarı / aşağı ihtimali (p_up / p_down) ---
    col_up, col_dn = st.columns(2)
    with col_up:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Yukarı ihtimali (p_up)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{format_pct(signal["p_up"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_dn:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Aşağı ihtimali (p_down)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{format_pct(signal["p_down"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Kalibre edilmiş başarı olasılığı (p_good) ---
    # Eğer kalibratör yoksa signal["p_good"] olmayabilir; o zaman p_up'u kullanıyoruz.
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Kalibre edilmiş başarı olasılığı</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-value">'
        f'{format_pct(signal.get("p_good", signal["p_up"]))}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Model yorumu ---
    st.markdown('<div class="section-title">Model yorumu</div>', unsafe_allow_html=True)
    if signal["label"] == "AL":
        st.markdown(
            '<div class="explanation">Model, yukarı yönlü hareket olasılığını aşağıya göre daha güçlü görüyor. '
            "Ancak EMA mesafeleri ve destek/direnç bölgelerine mutlaka göz at.</div>",
            unsafe_allow_html=True,
        )
    elif signal["label"] == "SAT":
        st.markdown(
            '<div class="explanation">Model, aşağı yönlü hareket olasılığını öne çıkarıyor. '
            "Yakındaki güçlü desteklerde tepki gelebileceğini unutma.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="explanation">Model, yukarı/aşağı yön arasında net bir avantaj görmüyor. '
            "Destek/direnç ve haber akışı ön plana çıkıyor.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------- ORTA: ANA GRAFİK ---------
with top_mid:
    st.markdown(f"#### 🕒 NASDAQ {timeframe.upper()} Fiyat Grafiği (Son {num_candles} Mum)")

    # Veri temizliği ve hazırlığı
    df_graph_clean = df_graph.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    
    # ✅ KULLANICI AYARINA GÖRE SON N MUMU AL (slider'dan gelen değer)
    if len(df_graph_clean) > num_candles:
        df_graph_clean = df_graph_clean.tail(num_candles).reset_index(drop=True)
        st.info(f"📊 {num_candles} mum gösteriliyor (slider ayarı)")
    else:
        st.warning(f"⚠️ Sadece {len(df_graph_clean)} mum var (istek: {num_candles})")
    
    # 🔥 TARİH DÜZELTMESİ: Pandas datetime objesine çevir
    if not pd.api.types.is_datetime64_any_dtype(df_graph_clean[TIME_COL]):
        df_graph_clean[TIME_COL] = pd.to_datetime(df_graph_clean[TIME_COL])
    
    # 🔍 DETAYLI DEBUG
    st.markdown("### 🔍 Grafik Debug Bilgileri")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gösterilen Mum", len(df_graph_clean))
        st.caption(f"Slider: {num_candles}")
    with col2:
        st.metric("İlk O", f"{df_graph_clean[OPEN_COL].iloc[0]:.2f}")
    with col3:
        st.metric("İlk C", f"{df_graph_clean[CLOSE_COL].iloc[0]:.2f}")
    with col4:
        st.metric("Son C", f"{df_graph_clean[CLOSE_COL].iloc[-1]:.2f}")
    
    st.caption(f"📅 İlk: {df_graph_clean[TIME_COL].iloc[0]} | Son: {df_graph_clean[TIME_COL].iloc[-1]}")
    st.caption(f"💰 Fiyat Aralığı: {df_graph_clean[LOW_COL].min():.2f} - {df_graph_clean[HIGH_COL].max():.2f}")
    
    # İlk 3 bar debug
    st.caption("📊 İlk 3 Bar OHLC:")
    for i in range(min(3, len(df_graph_clean))):
        row = df_graph_clean.iloc[i]
        st.caption(f"  Bar {i}: O={row[OPEN_COL]:.2f} H={row[HIGH_COL]:.2f} L={row[LOW_COL]:.2f} C={row[CLOSE_COL]:.2f}")
    
    # =============== GRAFİK ÇİZİMİ ===============
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_graph_clean[TIME_COL],
        open=df_graph_clean[OPEN_COL],
        high=df_graph_clean[HIGH_COL],
        low=df_graph_clean[LOW_COL],
        close=df_graph_clean[CLOSE_COL],
        name=f'NASDAQ {timeframe.upper()}',
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444',
        increasing_fillcolor='#10b981',
        decreasing_fillcolor='#ef4444',
        line=dict(width=1),
        whiskerwidth=0.5,
        showlegend=False
    ))
    
    st.success(f"✅ {len(df_graph_clean)} mum çizildi (Slider: {num_candles})")

    # Trend Kanalı
    if upper_ch is not None:
        x_dates = df_graph_clean[TIME_COL]
        fig.add_trace(go.Scatter(x=x_dates, y=upper_ch[:len(x_dates)], name="Üst Kanal",
            mode="lines", line=dict(color='#fbbf24', width=1.5, dash="dash"), opacity=0.8))
        fig.add_trace(go.Scatter(x=x_dates, y=lower_ch[:len(x_dates)], name="Alt Kanal",
            mode="lines", line=dict(color='#fbbf24', width=1.5, dash="dash"), opacity=0.8))

    # EMA'lar
    if EMA20_COL in df_graph_clean.columns:
        fig.add_trace(go.Scatter(x=df_graph_clean[TIME_COL], y=df_graph_clean[EMA20_COL],
            name="EMA 20", mode="lines", line=dict(width=1.2, color='#2563eb'), opacity=0.9))
    if EMA50_COL in df_graph_clean.columns:
        fig.add_trace(go.Scatter(x=df_graph_clean[TIME_COL], y=df_graph_clean[EMA50_COL],
            name="EMA 50", mode="lines", line=dict(width=1.2, color='#10b981'), opacity=0.9))
    if EMA200_COL in df_graph_clean.columns:
        fig.add_trace(go.Scatter(x=df_graph_clean[TIME_COL], y=df_graph_clean[EMA200_COL],
            name="EMA 200", mode="lines", line=dict(width=1.5, color='#f59e0b'), opacity=0.9))

    # Destek/Direnç
    x_max = len(df_graph_clean) - 1
    for i, zone in enumerate(supports, start=1):
        fig.add_shape(type="rect", x0=0, x1=x_max, y0=zone["level"]-10, y1=zone["level"]+10,
            fillcolor='rgba(16,185,129,0.1)', line=dict(color='rgba(16,185,129,0.3)', width=1), layer="below")
        fig.add_annotation(x=x_max, y=zone["level"], text=f" S{i}: {zone['level']:.1f} ",
            showarrow=False, xanchor="left", bgcolor='#064e3b', bordercolor='#10b981',
            font=dict(size=10, color='#86efac'))
    
    for i, zone in enumerate(resistances, start=1):
        fig.add_shape(type="rect", x0=0, x1=x_max, y0=zone["level"]-10, y1=zone["level"]+10,
            fillcolor='rgba(239,68,68,0.1)', line=dict(color='rgba(239,68,68,0.3)', width=1), layer="below")
        fig.add_annotation(x=x_max, y=zone["level"], text=f" R{i}: {zone['level']:.1f} ",
            showarrow=False, xanchor="left", bgcolor='#7f1d1d', bordercolor='#ef4444',
            font=dict(size=10, color='#fca5a5'))

    # Paternler
    patterns = detect_structural_patterns_enhanced(df_graph_clean)
    for i, pat in enumerate(patterns[:2]):
        xs, ys = [], []
        for s in pat["swing_points"]:
            if 0 <= s["idx"] < len(df_graph_clean):
                xs.append(s["idx"])
                ys.append(s["price"])
        if xs:
            color = {"up": "#10b981", "down": "#ef4444"}.get(pat.get("direction", "either"), "#8b5cf6")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=pat["name"],
                line=dict(width=2, color=color), marker=dict(size=6, color=color)))

    # Son fiyat
    last_price_color = '#089981' if df_graph_clean[CLOSE_COL].iloc[-1] >= df_graph_clean[OPEN_COL].iloc[-1] else '#f23645'
    fig.add_shape(type="line", x0=0, x1=x_max, y0=last_price, y1=last_price,
        line=dict(color=last_price_color, width=1.5, dash="dot"))
    fig.add_annotation(x=x_max, y=last_price, text=f" {last_price:.2f} ",
        showarrow=False, xanchor="left", bgcolor=last_price_color,
        font=dict(color='white', size=12, family='Arial Black'))

    # Layout
    y_min, y_max = float(df_graph_clean[LOW_COL].min()), float(df_graph_clean[HIGH_COL].max())
    y_padding = (y_max - y_min) * 0.1
    
    fig.update_layout(
        height=650, paper_bgcolor='#131722', plot_bgcolor='#131722',
        font=dict(family="Inter, sans-serif", size=12, color='#d1d4dc'),
        margin=dict(l=10, r=100, t=50, b=50),
        title={'text': f'NASDAQ {timeframe.upper()} | Technical Analysis',
               'y': 0.98, 'x': 0.01, 'font': dict(size=18, color='#e1e4ed')},
        showlegend=True, hovermode='x unified', dragmode='zoom'
    )
    
    # X/Y ekseni (df_graph_clean zaten num_candles ile sınırlandırılmış)
    start_date = df_graph_clean[TIME_COL].iloc[0]
    end_date = df_graph_clean[TIME_COL].iloc[-1]
    
    fig.update_xaxes(type='date', tickformat='%H:%M\n%b %d', range=[start_date, end_date],
        rangeslider=dict(visible=False), showgrid=True, gridcolor='#1e222d')
    fig.update_yaxes(side='right', tickformat=',.1f', range=[y_min-y_padding, y_max+y_padding],
        showgrid=True, gridcolor='#1e222d')

    st.plotly_chart(fig, use_container_width=True)
        
# ================== UYGULAMA GÖVDESİ ==================
# --------- SAĞ: EMA & S/R MESAFE KUTULARI ---------
with top_right:
    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
    st.markdown("#### 📏 EMA & S/R Mesafeleri")

    ema20 = float(last_row.get(EMA20_COL, np.nan))
    ema50 = float(last_row.get(EMA50_COL, np.nan))
    ema200 = float(last_row.get(EMA200_COL, np.nan))

    # EMA mesafeleri
    st.markdown('<div class="section-title">EMA Mesafeleri</div>', unsafe_allow_html=True)

    def ema_card(label, ema_val):
        if np.isnan(ema_val):
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">Veri yok</div></div>',
                unsafe_allow_html=True,
            )
            return
        diff = last_price - ema_val
        direction = "üstünde" if diff > 0 else "altında"
        cls = "distance-positive" if diff > 0 else "distance-negative"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value"><span class="{cls}">{diff:+.1f}</span> puan {direction}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    ema_card("EMA 20", ema20)
    ema_card("EMA 50", ema50)
    ema_card("EMA 200 (SMA tabanlı)", ema200)

    st.markdown('<div class="section-title">En Yakın Destekler</div>', unsafe_allow_html=True)
    if supports:
        for i, s in enumerate(supports[:3], start=1):
            dist = last_price - s["level"]
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">S{i} • {s["touches"]} dokunma</div>'
                f'<div class="metric-value">{s["level"]:.1f}  |  ↑ {dist:.1f} puan</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">Anlamlı güçlü destek tespit edilmedi.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">En Yakın Dirençler</div>', unsafe_allow_html=True)
    if resistances:
        for i, r in enumerate(resistances[:3], start=1):
            dist = r["level"] - last_price
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">R{i} • {r["touches"]} dokunma</div>'
                f'<div class="metric-value">{r["level"]:.1f}  |  ↓ {dist:.1f} puan</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">Anlamlı güçlü direnç tespit edilmedi.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ================== ALT PANEL ==================
st.markdown("---")
bottom_left, bottom_right = st.columns([1.4, 1.6])

# --------- ALT SOL: Trend & formasyon açıklaması ---------
with bottom_left:
    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
    st.markdown("#### 🧭 Trend & Yapısal Görünüm", unsafe_allow_html=True)

    # === 1) ÖZET TREND METNİ (trend_text) ===
    st.markdown('<div class="section-title">Trend Özeti</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{trend_text}</div></div>',
        unsafe_allow_html=True,
    )

    # === 2) MODEL SİNYALİ + DESTEK / DİRENÇ YORUMU ===
    st.markdown('<div class="section-title">Yorum</div>', unsafe_allow_html=True)
    explanation_lines = []

    if signal["label"] == "AL":
        explanation_lines.append("• Model yukarı yönü tercih ediyor.")
    elif signal["label"] == "SAT":
        explanation_lines.append("• Model aşağı yönü tercih ediyor.")
    else:
        explanation_lines.append("• Model yön konusunda net değil, daha çok yatay / haber odaklı bir yapı var.")

    if supports:
        dist_sup = last_price - supports[0]["level"]
        explanation_lines.append(
            f"• Fiyat en yakın desteğe yaklaşık {dist_sup:.1f} puan uzaklıkta."
        )

    if resistances:
        dist_res = resistances[0]["level"] - last_price
        explanation_lines.append(
            f"• Fiyat en yakın dirence yaklaşık {dist_res:.1f} puan uzaklıkta."
        )

    if not explanation_lines:
        explanation_lines.append("• Şu an için ekstra belirgin sinyal yok.")

    st.markdown(
        '<div class="metric-card"><div class="explanation">'
        + "<br>".join(explanation_lines)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # === 3) BÜYÜK FORMASYON ÖZETİ (HS / Double / Üçgen) ===
    st.markdown("<hr style='opacity:0.2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🧩 Büyük Formasyon Analizi", unsafe_allow_html=True)

    patterns = detect_structural_patterns(df_graph)

    if patterns:
        main_pat = patterns[0]  # en güncel olanı
        name = main_pat["name"]
        legs_total = main_pat.get("legs_total", None)
        legs_found = main_pat.get("legs_found", None)
        current_leg = main_pat.get("current_leg", None)
        remaining = main_pat.get("remaining_pips_to_next_leg", None)
        stage_text = main_pat.get("stage_text", "")

        lines = []
        lines.append(f"• Algılanan formasyon: **{name}**")

        if legs_total is not None and legs_found is not None:
            lines.append(f"• Ayak sayısı: {legs_found}/{legs_total}")
        if current_leg is not None and legs_total is not None:
            lines.append(f"• Şu an yaklaşık **{current_leg}. ayak** bölgesindeyiz.")

        if remaining is not None:
            lines.append(f"• Bir sonraki kritik ayağa tahmini mesafe: **{remaining:.1f} puan**")

        if stage_text:
            lines.append(f"• Durum: {stage_text}")

        st.markdown(
            '<div class="metric-card"><div class="explanation">'
            + "<br>".join(lines)
            + "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            'Şu an için belirgin bir büyük formasyon algılanmadı.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # === 4) SON MUM FORMASYONU (küçük formasyonlar) ===
    st.markdown("<hr style='opacity:0.2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🕯 Mum Formasyonu (son bar)", unsafe_allow_html=True)

    try:
        last_row = df_plot.iloc[-1]
        prev_row = df_plot.iloc[-2] if len(df_plot) >= 2 else None

        o = float(last_row[OPEN_COL])
        h = float(last_row[HIGH_COL])
        l = float(last_row[LOW_COL])
        c = float(last_row[CLOSE_COL])

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l + 1e-9

        pattern = "Belirgin bir mum formasyonu yok"

        # DOJI
        if body / total_range < 0.1:
            pattern = "⚪ Doji – kararsızlık bölgesi."

        # Çekiç (Hammer)
        if (
            lower_wick > body * 2
            and upper_wick < body * 1.2
            and c > o
        ):
            pattern = "🟢 Çekiç (Hammer) – dipte güçlenme sinyali."

        # Shooting star
        if (
            upper_wick > body * 2
            and lower_wick < body * 1.2
            and c < o
        ):
            pattern = "🔴 Shooting Star – tepede zayıflama sinyali."

        # Engulfing formasyonları (2 mum)
        if prev_row is not None:
            o_prev = float(prev_row[OPEN_COL])
            c_prev = float(prev_row[CLOSE_COL])

            # Bullish engulfing
            if (c_prev < o_prev) and (c > o) and (c >= o_prev) and (o <= c_prev):
                pattern = "🟢 Bullish Engulfing – aşağı trendde güçlü dönüş sinyali."
            # Bearish engulfing
            elif (c_prev > o_prev) and (c < o) and (c <= o_prev) and (o >= c_prev):
                pattern = "🔴 Bearish Engulfing – yukarı trendde güçlü dönüş sinyali."

        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{pattern}</div></div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">Mum formasyonu analizinde hata: {e}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # ================= EK TREND & FORMASYON ANALİZİ =================
    st.markdown("<hr style='opacity:0.2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 📐 Formasyon & Trend Analizi", unsafe_allow_html=True)

    # --- Basit sayısal trend (son 80 mum) ---
    closes = df_plot[CLOSE_COL].astype(float)
    if len(closes) >= 20:
        window = min(80, len(closes))
        y = closes.iloc[-window:]
        x = np.arange(len(y))

        coef = np.polyfit(x, y, 1)[0]  # eğim

        if coef > 0:
            trend_dir = "🟢 Yükselen trend"
        elif coef < 0:
            trend_dir = "🔴 Düşen trend"
        else:
            trend_dir = "⚪ Yatay"

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Sayısal trend (son {window} mum)</div>'
            f'<div class="metric-value">{trend_dir} (eğim: {coef:.2f})</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">'
            'Trend analizi için yeterli mum yok.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # --- Son mum için mum formasyonu tespiti ---
    try:
        last_row = df_plot.iloc[-1]
        prev_row = df_plot.iloc[-2] if len(df_plot) >= 2 else None

        o = float(last_row[OPEN_COL])
        h = float(last_row[HIGH_COL])
        l = float(last_row[LOW_COL])
        c = float(last_row[CLOSE_COL])

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l + 1e-9  # sıfıra bölme koruması

        pattern = "Belirgin bir mum formasyonu yok"

        # Çekiç (hammer) – uzun alt fitil, küçük gövde, üst fitil kısa, yeşil gövde
        if (
            lower_wick > body * 2
            and upper_wick < body * 1.2
            and c > o
        ):
            pattern = "🟢 Çekiç (hammer) – dip bölgede ise güçlü tepki sinyali olabilir."

        # Shooting star – uzun üst fitil, küçük gövde, kırmızı gövde
        elif (
            upper_wick > body * 2
            and lower_wick < body * 1.2
            and c < o
        ):
            pattern = "🔴 Shooting star – tepe bölgede ise dönüş sinyali olabilir."

        # Engulfing formasyonları için önceki mum gerekli
        if prev_row is not None:
            o_prev = float(prev_row[OPEN_COL])
            c_prev = float(prev_row[CLOSE_COL])

            # Bullish engulfing
            if (c_prev < o_prev) and (c > o) and (c >= o_prev) and (o <= c_prev):
                pattern = "🟢 Bullish engulfing – aşağı trendde güçlü dönüş sinyali."
            # Bearish engulfing
            elif (c_prev > o_prev) and (c < o) and (c <= o_prev) and (o >= c_prev):
                pattern = "🔴 Bearish engulfing – yukarı trendde güçlü dönüş sinyali."

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Son mum formasyonu</div>'
            f'<div class="metric-value">{pattern}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'Formasyon analizi sırasında hata: {e}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # signal-box kapanışı
    st.markdown("</div>", unsafe_allow_html=True)

# --------- ALT SAĞ: Hacim analizi ---------
with bottom_right:
    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
    st.markdown("#### 📊 Hacim Analizi")

    # --- Ham hacim serisi ---
    vol_series = df_plot[VOL_COL].astype(float)
    last_vol = float(vol_series.iloc[-1])
    vol_ma20 = float(vol_series.rolling(20, min_periods=5).mean().iloc[-1])
    vol_ma50 = float(vol_series.rolling(50, min_periods=10).mean().iloc[-1])

    # --- Oranlar ---
    vol_ratio_20 = last_vol / vol_ma20 if vol_ma20 > 0 else 0.0
    vol_ratio_50 = last_vol / vol_ma50 if vol_ma50 > 0 else 0.0

    # --- Son 40 bar alım / satım hacmi ---
    recent = df_plot.tail(40)
    green_vol = recent[recent[CLOSE_COL] >= recent[OPEN_COL]][VOL_COL].sum()
    red_vol = recent[recent[CLOSE_COL] < recent[OPEN_COL]][VOL_COL].sum()
    total_vol = green_vol + red_vol

    if total_vol > 0:
        buy_pressure = (green_vol / total_vol) * 100.0
        sell_pressure = (red_vol / total_vol) * 100.0
    else:
        buy_pressure = sell_pressure = 50.0

    # -------- ÜST METRİKLER --------
    c1, c2 = st.columns(2)

    with c1:
        # Son hacim
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Son hacim</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{last_vol:,.0f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 20 bar ortalama
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">20 bar ort.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{vol_ma20:,.0f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # 20 bar oranı
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Hacim oranı (20)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{vol_ratio_20:.2f}x</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 50 bar oranı
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Hacim oranı (50)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{vol_ratio_50:.2f}x</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Hacim yorumu (seviyeler) --------
    # 20-bar oranını referans alıyorum
    if vol_ratio_20 < 0.5:
        vol_comment = "💤 **Çok düşük hacim** – piyasa ilgisi zayıf, hareketler güvenilmez olabilir."
    elif vol_ratio_20 < 0.8:
        vol_comment = "🟡 **Düşük hacim** – normalin altında, kırılımlar zayıf kalabilir."
    elif vol_ratio_20 < 1.2:
        vol_comment = "⚪ **Normal hacim** – işlem aktivitesi standart seviyede."
    elif vol_ratio_20 < 1.8:
        vol_comment = "⚡ **Yüksek hacim** – normalin üzerinde, hareketler daha anlamlı."
    else:
        vol_comment = "🚀 **Aşırı yüksek hacim** – panik / FOMO bölgesi, sert hareketler mümkün."

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{vol_comment}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Alım / satım baskısı --------
    st.markdown('<div class="section-title">Alım / Satım baskısı (son 40 bar)</div>', unsafe_allow_html=True)

    if buy_pressure > 60:
        press_str = f"🟢 Alım baskısı baskın (**{buy_pressure:.1f}%**)"
    elif sell_pressure > 60:
        press_str = f"🔴 Satım baskısı baskın (**{sell_pressure:.1f}%**)"
    else:
        press_str = f"⚪ Dengeli hacim (Alım: {buy_pressure:.1f}%, Satım: {sell_pressure:.1f}%)"

    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{press_str}</div></div>',
        unsafe_allow_html=True,
    )

    # -------- Mini rehber: bu sayılar ne demek? --------
    st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)
    st.markdown("**Hacmi nasıl okuyacaksın?**", unsafe_allow_html=True)
    st.markdown(
        """
- `Hacim oranı (20)` ≈ **1.00x** → son mum hacmi, son 20 mum ortalaması civarında (normal).
- `0.50x` altı → bariz düşük hacim, piyasa uykuda.
- `1.50x` üstü → ortalamanın çok üstünde, güçlü ilgi / haber etkisi olabilir.
- Alım baskısı %60+ ise **alıcılar**, %60+ satım ise **satıcılar** daha agresif.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
# ================== AUTO-TRADE LAB (BACKTEST PANELİ) ==================
st.markdown("---")
st.markdown("### 🧪 Auto-Trade Lab – Model Sinyallerinin Gerçek Performansı")

with st.expander("Modeli tarihsel veride otomatik trade ettir ve sonuçlarına bak"):
    # 🔍 HIZLI TEST PANELİ
    st.markdown("### 🧪 Model Sinyal Testi")
    test_bar = st.slider("Test etmek istediğiniz bar numarası", 0, len(df)-1, len(df)-1)
    test_row = df.iloc[test_bar]

    # Model sinyalini al
    test_signal = get_model_signal(test_row, model_bundle, thr_buy=0.52, thr_sell=0.48)

    st.json(test_signal)

    if test_signal.get('confidence', 0.0) >= 0.10:
        st.success(f"✅ İşlem AÇILIR - Label: {test_signal.get('label', 'N/A')}")
    else:
        st.error("❌ İşlem AÇILMAZ - Güven düşük")
    
    # 🔍 MODEL İÇERİĞİ DETAYLI
    if st.checkbox("🛠️ Model detaylarını göster"):
        try:
            model_data = joblib.load(MODEL_PATH)
            
            if 'models' in model_data:
                st.subheader("'models' anahtarının içeriği")
                st.write(f"Tip: {type(model_data['models'])}")
                
                if isinstance(model_data['models'], (list, tuple)):
                    st.write(f"Uzunluk: {len(model_data['models'])}")
                    for i, m in enumerate(model_data['models']):
                        st.write(f"Model {i}: {type(m)}")
                else:
                    st.write(f"Tek model: {type(model_data['models'])}")
            
            # Diğer anahtarları da göster
            st.subheader("Model dosyasındaki tüm anahtarlar")
            st.write(list(model_data.keys()))
            
        except Exception as e:
            st.error(f"❌ Model detayları yüklenirken hata: {e}")
    
    # 🔍 GERÇEK NaN KONTROLÜ
    st.markdown("### 🔍 NaN Değer Analizi")
    
    # Son 5 satırı kontrol et
    nan_summary = df.isna().sum()
    if nan_summary.sum() > 0:
        st.error(f"❌ Toplam {nan_summary.sum()} adet NaN değer var!")
        
        # Hangi kolonlarda NaN var?
        nan_cols = nan_summary[nan_summary > 0].sort_values(ascending=False)
        st.write("NaN olan kolonlar:", nan_cols.head(10))
    else:
        st.success("✅ Veride NaN yok!")
    
    # Son satırda kaç 0 var?
    last_row_zeros = (df.iloc[-1] == 0).sum()
    st.info(f"Son satırda {last_row_zeros} adet 0 değer var")
    
    # 🔍 MUM SAYISI KONTROLÜ
    st.markdown("### 🔍 Grafik Veri Kontrolü")
    
    # df_graph_clean'i oluştur (grafik bölümündeki gibi)
    df_graph_check = df_plot.tail(num_candles).copy()
    df_graph_check = df_graph_check.drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)
    df_graph_clean_check = df_graph_check.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toplam Bar Sayısı", len(df_graph_clean_check))
    
    with col2:
        st.metric("Unique Tarih Sayısı", df_graph_clean_check[TIME_COL].nunique())
    
    with col3:
        duplicates = len(df_graph_clean_check) - df_graph_clean_check[TIME_COL].nunique()
        st.metric("Tekrarlanan Bar", duplicates)
    
    # İlk 5 ve son 5 barı göster (kontrol için)
    st.markdown("**İlk 5 Bar:**")
    st.dataframe(df_graph_clean_check[[TIME_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].head())
    
    st.markdown("**Son 5 Bar:**")
    st.dataframe(df_graph_clean_check[[TIME_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]].tail())
    
    st.markdown("---")

    cols = st.columns(3)
    with cols[0]:
        horizon_bars = st.slider(
            "Sinyal sonrası izlenecek bar sayısı (M30)",
            min_value=4,
            max_value=20,
            value=10,
            step=1,
            help="M30 grafikte 10 bar ≈ 5 saat demek."
        )
    with cols[1]:
        thr_buy_bt = st.slider(
            "Backtest için AL eşiği",
            min_value=0.50,
            max_value=0.85,
            value=thr_buy,
            step=0.01,
        )
    with cols[2]:
        thr_sell_bt = st.slider(
            "Backtest için SAT eşiği",
            min_value=0.15,
            max_value=0.50,
            value=thr_sell,
            step=0.01,
        )

    # ================== MODEL TRADE SİMÜLASYONU ==================
    st.markdown("### 📈 Model Trade Simülasyonu")

    # 1) Model güven eşiği slider’ı
    conf_th = st.slider(
        "Model güven eşiği (confidence)",
        min_value=0.01,
        max_value=0.90,
        value=0.01,   # varsayılan: %60
        step=0.01,
        help="Model sinyaline ne kadar güvenirsek trade açalım? Ne kadar yüksek, o kadar az ama daha seçici işlem açar."
    )

    if st.button("▶️ Simülasyonu Çalıştır"):
        with st.spinner("Model sinyallerini simüle ediyor..."):
            trades_df = simulate_model_trades(
                df=df,
                model_bundle=model_bundle,
                thr_buy=thr_buy_bt,
                thr_sell=thr_sell_bt,
                horizon_bars=horizon_bars,
                conf_threshold=conf_th,  # 🔴 yeni parametre
            )

        if trades_df is None or trades_df.empty:
            st.warning("Hiç işlem oluşmadı. Güven eşiği veya threshold'lar çok agresif olabilir, biraz gevşetmeyi dene.")
        else:
            # trades_df'yi session state'e kaydet (rapor için)
            st.session_state['trades_df'] = trades_df
            
            st.success(f"✅ {len(trades_df)} adet işlem simüle edildi!")

            # İstersen disk'e kaydet (sonradan model eğitimi için)
            save_path = "auto_trades_log.parquet"
            try:
                if os.path.exists(save_path):
                    # Eski kayıtlarla birleştir
                    old = pd.read_parquet(save_path)
                    combined = pd.concat([old, trades_df], ignore_index=True)
                    combined.to_parquet(save_path, index=False)
                else:
                    trades_df.to_parquet(save_path, index=False)
                st.success(
                    f"İşlem kayıtları `{save_path}` dosyasına kaydedildi. "
                    "(Gelecekte meta-model / kalibrasyon eğitiminde kullanabilirsin.)"
                )
            except Exception as e:
                st.warning(f"Kayıt dosyasına yazarken hata oldu: {e}")
    
    # ================== DETAYLI İŞLEM RAPORU ==================
    st.markdown("---")
    st.markdown("### 📊 İŞLEM GEÇMİŞİ & PERFORMANS RAPORU")
    
    # Session state'ten veya yeni çalıştırılan simülasyondan trades_df'yi al
    if 'trades_df' in st.session_state:
        trades_df = st.session_state['trades_df']
    else:
        trades_df = None
    
    if trades_df is not None and not trades_df.empty:
        # Kümülatif PnL hesapla
        trades_df['cumulative_pnl'] = trades_df['pnl_points'].cumsum()
        
        # Metrik kartları
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trades_df)
        wins = (trades_df['pnl_points'] > 0).sum()
        losses = (trades_df['pnl_points'] < 0).sum()
        win_rate = wins / total_trades * 100
        
        with col1:
            st.metric("Toplam İşlem", f"{total_trades}")
            st.metric("Kazanan", f"{wins}", delta_color="inverse")
        
        with col2:
            st.metric("Kazanç Oranı", f"{win_rate:.1f}%")
            st.metric("Kaybeden", f"{losses}", delta_color="inverse")
        
        with col3:
            avg_pnl = trades_df['pnl_points'].mean()
            st.metric("Ortalama PnL", f"{avg_pnl:.1f} puan")
        
        with col4:
            total_pnl = trades_df['pnl_points'].sum()
            st.metric("Toplam PnL", f"{total_pnl:.1f} puan", 
                     delta="✅ Kar" if total_pnl > 0 else "❌ Zarar")
        
        # Kümülatif PnL grafiği
        st.markdown("#### 📈 Kümülatif PnL Grafiği")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=trades_df['entry_time'],
            y=trades_df['cumulative_pnl'],
            mode='lines+markers',
            name='Kümülatif PnL',
            line=dict(color='#10b981', width=3),
            marker=dict(size=6)
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig_pnl.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Son 10 işlem detayı (kartlar)
        st.markdown("#### 🎴 Son 10 İşlem Detayı")
        recent_trades = trades_df.sort_values('entry_time', ascending=False).head(10)
        
        for idx, trade in recent_trades.iterrows():
            pnl_color = "#10b981" if trade['pnl_points'] > 0 else "#ef4444"
            pnl_rgb = "16, 185, 129" if trade['pnl_points'] > 0 else "239, 68, 68"
            
            st.markdown(f"""
            <div style="border:1px solid {pnl_color}; border-radius:8px; padding:12px; margin:8px 0; background:rgba({pnl_rgb}, 0.1)">
                <strong>🕒 {trade['entry_time']}</strong> | 
                <strong style="color:{pnl_color}">{trade['direction']}</strong> | 
                Giriş: <strong>{trade['entry_price']:.1f}</strong> | 
                Çıkış: <strong>{trade['exit_price']:.1f}</strong> | 
                PnL: <strong style="color:{pnl_color}">{trade['pnl_points']:.1f}</strong> puan | 
                Süre: <strong>{trade['holding_minutes']//60}h {(trade['holding_minutes']%60)}m</strong> | 
                Güven: <strong>{trade['confidence']:.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # TP/SL isabet oranı
        tp_hits = (trades_df['pnl_points'] == 25.0).sum()
        sl_hits = (trades_df['pnl_points'] == -25.0).sum()
        manual_exit = total_trades - tp_hits - sl_hits
        
        st.markdown("#### 🎯 TP/SL İsabet Oranı")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TP Vuruşu", f"{tp_hits} ({tp_hits/total_trades*100:.1f}%)")
        with col2:
            st.metric("SL Vuruşu", f"{sl_hits} ({sl_hits/total_trades*100:.1f}%)")
        with col3:
            st.metric("Manuel Çıkış", f"{manual_exit} ({manual_exit/total_trades*100:.1f}%)")
        
        # Tüm işlemleri indir
        if st.download_button(
            label="📥 İşlem Geçmişini İndir (CSV)",
            data=trades_df.to_csv(index=False).encode('utf-8'),
            file_name='nasdaq_trades_history.csv',
            mime='text/csv',
        ):
            st.success("İndirme başladı!")
            
    else:
        st.info("Henüz işlem geçmişi yok. Simülasyonu çalıştırın.")
        # ================== İŞLEM RAPORU (EN ALTA EKLEYİN) ==================
if 'trades_df' in locals() and trades_df is not None and not trades_df.empty:
    st.markdown("---")
    st.markdown("### 📊 SONUÇLAR")
    
    total = len(trades_df)
    wins = (trades_df['pnl_points'] > 0).sum()
    win_rate = wins / total * 100
    
    st.success(f"✅ {total} işlem sonuçlandı! Win Rate: {win_rate:.1f}%")
    
    # Kümülatif PnL
    trades_df['cumulative_pnl'] = trades_df['pnl_points'].cumsum()
    st.line_chart(trades_df['cumulative_pnl'])