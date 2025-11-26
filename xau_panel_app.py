import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from scipy.signal import argrelextrema
from typing import List, Dict, Any, Optional
from patterns_nasdaq import (
    find_swings,
    detect_flag_pattern,  # detect_candlestick_patterns artık bu dosyada tanımlanacak
)

# ================== GENEL AYARLAR ==================

st.set_page_config(
    page_title="📈 NASDAQ M30 AI Trading Panel",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DATA_FILE = "xau_training_dataset_v2.parquet"          # senin XAU parquet ismin
MODEL_PATH = "models/xau_meta_optuna_cv_v2.pkl"  

# Ana timeframe M30 kolonları
TIME_COL = "datetime"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOL_COL = "volume"

EMA20_COL = "ema_20"
EMA50_COL = "ema_50"
EMA200_COL = "sma_200"  # EMA200 yoksa sma_200'ü proxy gibi kullanırız

SUP_COL = "support_strength"
RES_COL = "resistance_strength"

# Sinyal için threshold'lar
THR_BUY = 0.55   # p(1) >= 0.55 → AL
THR_SELL = 0.45  # p(1) <= 0.45 → SAT


# ================== STİL / TEMA ==================

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top, #101622 0%, #05070b 45%, #020305 100%);
        color: #f5f5f5;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    .main-header {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #42a5ff, #7b61ff, #ffcc33);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #9fa6b2;
        margin-bottom: 25px;
        font-size: 13px;
    }
    .signal-box {
        background: linear-gradient(135deg, rgba(15,30,60,0.98), rgba(10,10,20,0.98));
        border-radius: 18px;
        padding: 18px 16px;
        border: 1px solid rgba(93, 160, 255, 0.35);
        box-shadow: 0 18px 35px rgba(0,0,0,0.65);
    }
    .metric-card {
        background: rgba(10, 12, 22, 0.95);
        border-radius: 12px;
        padding: 10px 12px;
        border: 1px solid rgba(120, 144, 255, 0.20);
        margin-bottom: 8px;
        font-size: 13px;
        color: #e5e7eb;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 15px;
        font-weight: 600;
        margin-top: 2px;
    }
    .buy-signal {
        color: #4ade80;
        font-weight: 700;
        font-size: 22px;
    }
    .sell-signal {
        color: #f97373;
        font-weight: 700;
        font-size: 22px;
    }
    .neutral-signal {
        color: #e5e7eb;
        font-weight: 700;
        font-size: 20px;
    }
    .distance-positive {
        color: #4ade80;
        font-weight: 600;
    }
    .distance-negative {
        color: #f97373;
        font-weight: 600;
    }
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 6px;
    }
    .explanation {
        font-size: 12px;
        color: #9ca3af;
        line-height: 1.4;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ================== YARDIMCI FONKSİYONLAR ==================


@st.cache_data(ttl=15)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


@st.cache_resource
def load_model_bundle(path: str):
    obj = joblib.load(path)
    # Beklenen anahtarlar: 'ensemble', 'scaler', 'features'
    ensemble = obj.get("ensemble", None)
    scaler = obj.get("scaler", None)
    feat_names = obj.get("features", None)
    return {"ensemble": ensemble, "scaler": scaler, "features": feat_names}
@st.cache_resource
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

    # last_row: pandas Series, biz bunu tek satırlık DataFrame'e çevirip feature'lara göre hizalayalım
    row_df = last_row.to_frame().T

    # Sadece modellerin beklediği kolonlar
    row_df = row_df.reindex(columns=feat_names, fill_value=0.0)

    # Sayısal olmayan varsa zorla float'a çevir (hata olursa 0 yap)
    for c in row_df.columns:
        if not np.issubdtype(row_df[c].dtype, np.number):
            row_df[c] = pd.to_numeric(row_df[c], errors="coerce")
    row_df = row_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_arr = row_df.values.astype(float)
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    return X_arr


def get_model_signal(last_row: pd.Series, model_bundle: dict, thr_buy: float, thr_sell: float):
    ensemble = model_bundle["ensemble"]
    if ensemble is None or len(ensemble) == 0:
        return {
            "label": "BELİRSİZ",
            "p_up": 0.5,
            "p_down": 0.5,
            "confidence": 0.0,
            "p_good": 0.5,  # kalibrasyon yoksa default
        }

    X_arr = prepare_features_for_model(last_row, model_bundle)

    probs_list = []
    for m in ensemble:
        proba = m.predict_proba(X_arr)
        probs_list.append(proba[:, 1])  # class=1 (yukarı) olasılığı

    p1 = float(np.mean(probs_list))
    p0 = 1.0 - p1
    conf = max(p0, p1)

    if p1 >= thr_buy:
        label = "AL"
    elif p1 <= thr_sell:
        label = "SAT"
    else:
        label = "PAS"

    # === KALİBRASYON: p_up → p_good ===
    cal_bundle = load_calibrator_bundle(CALIBRATOR_PATH)
    if cal_bundle is not None and "calibrator" in cal_bundle:
        calibrator = cal_bundle["calibrator"]
        # X = [[p_up]]
        p_good = float(calibrator.predict_proba(np.array([[p1]]))[:, 1][0])
    else:
        # Kalibratör yoksa p_good = p_up gibi davran
        p_good = p1

    return {
        "label": label,
        "p_up": p1,
        "p_down": p0,
        "confidence": conf,
        "p_good": p_good,
    }

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
    price_series = df_for_pattern["close"].astype(float)
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

def simulate_model_trades(
    df: pd.DataFrame,
    model_bundle: dict,
    thr_buy: float,
    thr_sell: float,
    horizon_bars: int = 10,
) -> pd.DataFrame:
    """
    Modeli sanki canlı çalışıyormuş gibi tarihsel veride ilerletip
    AL / SAT sinyallerinde sanal işlem açar ve sonuçlarını kayıt altına alır.

    horizon_bars: sinyal sonrası kaç bar boyunca performans ölçülecek (M30 → 10 bar = 5 saat)
    """
    trades = []

    # Çok erken barlarda bazı indikatörler NaN olabilir, o yüzden biraz kenara çekilelim
    start_idx = 50

    for i in range(start_idx, len(df) - horizon_bars):
        row = df.iloc[i]
        signal = get_model_signal(row, model_bundle, thr_buy, thr_sell)

        if signal["label"] == "PAS":
            continue  # sadece net AL / SAT sinyallerini test edelim

        direction = 1 if signal["label"] == "AL" else -1
        entry_price = float(row[CLOSE_COL])
        entry_time = row[TIME_COL]

        future = df.iloc[i + 1 : i + 1 + horizon_bars].copy()
        if future.empty:
            break

        prices = future[CLOSE_COL].astype(float).to_numpy()
        times = future[TIME_COL].to_list()

        # Her bar için PnL serisi (puanda)
        pnl_series = (prices - entry_price) * direction

        # Basit senaryo: pozisyon horizon sonunda kapanıyor
        exit_price = float(prices[-1])
        exit_time = times[-1]

        max_fav = float(pnl_series.max())   # max bizim yönümüzde ne kadar gitmiş
        max_adv = float(pnl_series.min())   # ters yönde ne kadar gitmiş (genelde negatif)
        final_pnl = float(pnl_series[-1])   # kapanıştaki PnL

        bars_in_profit = int((pnl_series > 0).sum())
        bars_in_loss = int((pnl_series < 0).sum())
        minutes_in_profit = bars_in_profit * 30
        minutes_in_loss = bars_in_loss * 30

        trades.append(
            {
                "entry_idx": i,
                "entry_time": entry_time,
                "signal_label": signal["label"],
                "p_up": signal["p_up"],
                "p_down": signal["p_down"],
                "confidence": signal["confidence"],
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "holding_bars": horizon_bars,
                "holding_minutes": horizon_bars * 30,
                "pnl_points": final_pnl,
                "max_favorable": max_fav,
                "max_adverse": max_adv,
                "minutes_in_profit": minutes_in_profit,
                "minutes_in_loss": minutes_in_loss,
            }
        )

    trades_df = pd.DataFrame(trades)
    return trades_df
# ================== UYGULAMA GÖVDESİ ==================
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
st.markdown('<div class="main-header">📈 XAUUSD M30 AI TRADING PANEL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Canlı fiyat, trend kanalları, destek/direnç, EMA mesafeleri ve AI sinyali tek ekranda</div>',
    unsafe_allow_html=True,
)

# --------- Sidebar: Ayarlar ---------
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    auto_refresh = st.checkbox("🔄 Otomatik yenile (10 sn)", value=False)
    n_bars = st.slider("Grafikte gösterilecek bar sayısı", min_value=120, max_value=500, value=200, step=20)
    thr_buy = st.slider("AL eşiği (p_up)", min_value=0.50, max_value=0.80, value=THR_BUY, step=0.01)
    thr_sell = st.slider("SAT eşiği (p_up)", min_value=0.20, max_value=0.50, value=THR_SELL, step=0.01)

    st.markdown("---")
    st.markdown("**Not:** Bu panel şu an M30 NASDAQ verisi ve NASDAQ v2 AI modeli ile çalışıyor.")

if auto_refresh:
    # 10 saniyede bir sayfayı yenile
    st.experimental_rerun()

# --------- Veri ve model yükleme ---------
df = load_data(DATA_FILE)
model_bundle = load_model_bundle(MODEL_PATH)

if df.empty:
    st.error("Veri seti boş görünüyor.")
    st.stop()

# Son N bar grafik için
df_plot = df.copy()
df_graph = df_plot.tail(n_bars).copy()
df_graph = df_graph.drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)
df_graph["bar_idx"] = range(len(df_graph))

last_row = df_plot.iloc[-1].copy()
last_price = float(last_row[CLOSE_COL])
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
# --------- ORTA: ANA GRAFİK ---------
with top_mid:
    st.markdown("#### 🕒 M30 NASDAQ Fiyat & Trend Kanalı")

    # Veri temizliği ve hazırlığı
    df_graph_clean = df_graph.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    
    # X ekseni için sürekli index kullan (mum aralıkları düzgün olsun)
    x_indices = list(range(len(df_graph_clean)))
    
    fig = go.Figure()

    # ===================== TRADINGVIEW TARZINDA MUM GRAFİĞİ =====================
    # ===================== TRADINGVIEW TARZINDA MUM GRAFİĞİ =====================
        # ===================== TRADINGVIEW TARZINDA MUM GRAFİĞİ =====================
    fig.add_trace(
    go.Candlestick(
        x=x_indices,  # Sürekli index
        open=df_graph_clean[OPEN_COL].values,
        high=df_graph_clean[HIGH_COL].values,
        low=df_graph_clean[LOW_COL].values,
        close=df_graph_clean[CLOSE_COL].values,
        name="NASDAQ M30",

        increasing_line_color="#089981",
        increasing_fillcolor="#089981",
        decreasing_line_color="#f23645",
        decreasing_fillcolor="#f23645",

        whiskerwidth=0.4,
        showlegend=False,
    )
)

    # ===================== TREND KANALI =====================
    if upper_ch is not None:
        # X indices'e göre yeniden hesapla
        upper_ch_clean = upper_ch[:len(x_indices)]
        lower_ch_clean = lower_ch[:len(x_indices)]
        middle_ch_clean = middle_ch[:len(x_indices)]
        
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=upper_ch_clean,
                name="Üst Kanal",
                mode="lines",
                line=dict(color='#fbbf24', width=1.5, dash="dash"),
                opacity=0.8,
                hovertemplate='Üst: %{y:.2f}<extra></extra>'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=lower_ch_clean,
                name="Alt Kanal",
                mode="lines",
                line=dict(color='#fbbf24', width=1.5, dash="dash"),
                opacity=0.8,
                hovertemplate='Alt: %{y:.2f}<extra></extra>'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=middle_ch_clean,
                name="Orta Çizgi",
                mode="lines",
                line=dict(color='#94a3b8', width=1, dash="dot"),
                opacity=0.5,
                showlegend=False,
                hovertemplate='Orta: %{y:.2f}<extra></extra>'
            )
        )

    # ===================== EMA ÇİZGİLERİ =====================
    if EMA20_COL in df_graph_clean.columns:
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df_graph_clean[EMA20_COL].values,
                name="EMA 20",
                mode="lines",
                line=dict(width=1.2, color='#2563eb'),
                opacity=0.9,
                hovertemplate='EMA20: %{y:.2f}<extra></extra>'
            )
        )

    if EMA50_COL in df_graph_clean.columns:
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df_graph_clean[EMA50_COL].values,
                name="EMA 50",
                mode="lines",
                line=dict(width=1.2, color='#10b981'),
                opacity=0.9,
                hovertemplate='EMA50: %{y:.2f}<extra></extra>'
            )
        )

    if EMA200_COL in df_graph_clean.columns:
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=df_graph_clean[EMA200_COL].values,
                name="EMA 200",
                mode="lines",
                line=dict(width=1.5, color='#f59e0b'),
                opacity=0.9,
                hovertemplate='EMA200: %{y:.2f}<extra></extra>'
            )
        )

    # ===================== DESTEK BÖLGELERİ =====================
    for i, zone in enumerate(supports, start=1):
        # Dikdörtgen destek bölgesi
        fig.add_shape(
            type="rect",
            x0=0,
            x1=len(x_indices) - 1,
            y0=zone["level"] - 10,
            y1=zone["level"] + 10,
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
            layer="below",
        )
        
        # Destek çizgisi (daha belirgin)
        fig.add_shape(
            type="line",
            x0=0,
            x1=len(x_indices) - 1,
            y0=zone["level"],
            y1=zone["level"],
            line=dict(color='rgba(16, 185, 129, 0.6)', width=1, dash="solid"),
            layer="below",
        )
        
        # Destek etiketi
        fig.add_annotation(
            x=len(x_indices) - 1,
            y=zone["level"],
            xanchor="left",
            text=f" S{i}: {zone['level']:.1f} ({zone['touches']}x) ",
            showarrow=False,
            bgcolor='#064e3b',
            bordercolor='#10b981',
            borderwidth=1,
            font=dict(size=11, color='#86efac', family='monospace'),
        )

    # ===================== DİRENÇ BÖLGELERİ =====================
    for i, zone in enumerate(resistances, start=1):
        # Dikdörtgen direnç bölgesi
        fig.add_shape(
            type="rect",
            x0=0,
            x1=len(x_indices) - 1,
            y0=zone["level"] - 10,
            y1=zone["level"] + 10,
            fillcolor='rgba(239, 68, 68, 0.1)',
            line=dict(color='rgba(239, 68, 68, 0.3)', width=1),
            layer="below",
        )
        
        # Direnç çizgisi (daha belirgin)
        fig.add_shape(
            type="line",
            x0=0,
            x1=len(x_indices) - 1,
            y0=zone["level"],
            y1=zone["level"],
            line=dict(color='rgba(239, 68, 68, 0.6)', width=1, dash="solid"),
            layer="below",
        )
        
        # Direnç etiketi
        fig.add_annotation(
            x=len(x_indices) - 1,
            y=zone["level"],
            xanchor="left",
            text=f" R{i}: {zone['level']:.1f} ({zone['touches']}x) ",
            showarrow=False,
            bgcolor='#7f1d1d',
            bordercolor='#ef4444',
            borderwidth=1,
            font=dict(size=11, color='#fca5a5', family='monospace'),
        )

    # ===================== BÜYÜK FORMASYONLARI ÇİZ =====================
    patterns = detect_structural_patterns_enhanced(df_graph_clean)

    for i, pat in enumerate(patterns[:2]):  # En önemli 2 formasyon
        swings_pat = pat["swing_points"]

        xs = []
        ys = []
        for s in swings_pat:
            if 0 <= s["idx"] < len(df_graph_clean):
                xs.append(s["idx"])  # Doğrudan index kullan
                ys.append(s["price"])

        if not xs:
            continue

        # Formasyona göre renk
        color_map = {
            "up": "#10b981",      # Yeşil
            "down": "#ef4444",    # Kırmızı
            "either": "#8b5cf6",  # Mor
            "reversal": "#f59e0b", # Turuncu
        }
        color = color_map.get(pat.get("direction", "either"), "#6b7280")
        
        # Formasyon çizgileri
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=pat["name"],
                line=dict(
                    width=2.5 if i == 0 else 2,
                    dash="solid" if i == 0 else "dash",
                    color=color,
                ),
                marker=dict(
                    size=8,
                    color=color,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                opacity=0.8,
                showlegend=True,
                hovertemplate='%{y:.2f}<extra></extra>'
            )
        )

        # Harfleri ekle (A,B,C,D...)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for j, (xv, yv) in enumerate(zip(xs, ys)):
            if j >= len(letters):
                break
            fig.add_annotation(
                x=xv,
                y=yv,
                text=f"<b>{letters[j]}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.7,
                arrowwidth=1.5,
                arrowcolor=color,
                ax=0,
                ay=-30,
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor=color,
                borderwidth=1.5,
                font=dict(size=12, color='white', family='Arial Black'),
            )

        # Neckline varsa
        if "neckline" in pat and pat["neckline"] is not None:
            fig.add_shape(
                type="line",
                x0=0,
                x1=len(x_indices) - 1,
                y0=pat["neckline"],
                y1=pat["neckline"],
                line=dict(color=color, width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=len(x_indices) - 1,
                y=pat["neckline"],
                text=f" {pat['name']} Neckline: {pat['neckline']:.1f} ",
                showarrow=False,
                xanchor="left",
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=color,
                borderwidth=1,
                font=dict(size=10, color=color, family='monospace'),
            )

    # ===================== SON FİYAT ÇİZGİSİ =====================
    last_price_color = '#089981' if df_graph_clean[CLOSE_COL].iloc[-1] >= df_graph_clean[OPEN_COL].iloc[-1] else '#f23645'
    
    fig.add_shape(
        type="line",
        x0=0,
        x1=len(x_indices) - 1,
        y0=last_price,
        y1=last_price,
        line=dict(color=last_price_color, width=1.5, dash="dot"),
    )
    
    # Son fiyat etiketi (sağ tarafta)
    fig.add_annotation(
        x=len(x_indices) - 1,
        y=last_price,
        text=f" {last_price:.2f} ",
        showarrow=False,
        xanchor="left",
        bgcolor=last_price_color,
        font=dict(color='white', size=12, family='Arial Black'),
        bordercolor=last_price_color,
        borderwidth=2,
    )

    # ===================== TRADINGVIEW LAYOUT AYARLARI =====================
    
    # Y ekseni aralığını hesapla
    y_min = float(df_graph_clean[LOW_COL].min())
    y_max = float(df_graph_clean[HIGH_COL].max())
    y_padding = (y_max - y_min) * 0.1
    
    fig.update_layout(
        height=580,
        
        # TradingView renk şeması
        template=None,
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        
        # Font ayarları
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color='#d1d4dc'
        ),
        
        # Margin ayarları
        margin=dict(l=5, r=90, t=40, b=40),
        
        # Başlık
        title={
            'text': 'NASDAQ M30 | Technical Analysis',
            'y': 0.98,
            'x': 0.01,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': dict(size=16, color='#e1e4ed', family='Arial Black')
        },
        
        # Legend ayarları
        showlegend=True,
        legend=dict(
            bgcolor='rgba(19, 23, 34, 0.9)',
            bordercolor='#2a2e39',
            borderwidth=1,
            font=dict(size=11, color='#d1d4dc'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            orientation="h",
            itemwidth=30,
        ),
        
        # Hover modu
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e222d',
            bordercolor='#3c434c',
            font=dict(size=11, color='#e1e4ed', family='monospace'),
            align='left',
        ),
        
        # Grafik etkileşimleri
        dragmode='pan',
    )
    
    # X Ekseni ayarları
    fig.update_xaxes(
        # Grid
        showgrid=True,
        gridcolor='#1e222d',
        gridwidth=1,
        
        # Eksen çizgisi
        showline=True,
        linewidth=1,
        linecolor='#3c434c',
        
        # Tick labels - Bar numaraları yerine boş bırak veya tarih göster
        showticklabels=True,
        tickfont=dict(size=10, color='#787b86'),
        tickmode='linear',
        tick0=0,
        dtick=max(1, len(x_indices) // 10),  # Her 10 mumda bir tick
        
        # Crosshair
        showspikes=True,
        spikecolor='rgba(131,136,141,0.5)',
        spikethickness=0.5,
        spikemode='across',
        spikesnap='cursor',
        spikedash='solid',
        
        # Range slider kapalı
        rangeslider=dict(visible=False),
        
        # Zoom ayarları
        fixedrange=False,
    )
    
    # Y Ekseni ayarları (sağ taraf)
    fig.update_yaxes(
        # Grid
        showgrid=True,
        gridcolor='#1e222d',
        gridwidth=1,
        
        # Eksen çizgisi
        showline=True,
        linewidth=1,
        linecolor='#3c434c',
        
        # Sağ tarafta göster
        side='right',
        
        # Tick ayarları
        tickfont=dict(size=11, color='#b8bcc8'),
        tickformat='.1f',
        
        # Crosshair
        showspikes=True,
        spikecolor='rgba(131,136,141,0.5)',
        spikethickness=0.5,
        spikemode='across',
        spikesnap='cursor',
        spikedash='solid',
        
        # Range ayarı
        range=[y_min - y_padding, y_max + y_padding],
        fixedrange=False,
        
        # Otomatik ayarlama
        automargin=True,
    )
    
    # Eğer çok fazla mum varsa son 100'ü göster
    if len(x_indices) > 100:
        fig.update_xaxes(range=[len(x_indices) - 100, len(x_indices)])

    # Config ayarları
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'xauusd_m30_chart',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
  
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

        # Prominence filtresi (çok yakın salınımları at)
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
    # === 👉 BURASI CLAUDE'UN FORMASYON ÇİZİM KODU 👇 ===
    patterns = detect_structural_patterns_enhanced(df_graph)

    for i, pat in enumerate(patterns[:2]):  # En önemli 2 formasyonu çiz
        swings_pat = pat["swing_points"]
        
        # Güvenli indeks kontrolü
        xs = []
        ys = []
        for s in swings_pat:
            if s["idx"] < len(df_graph):
                xs.append(df_graph.iloc[s["idx"]]["bar_idx"])
                ys.append(s["price"])
        
        if xs:  # Veri varsa çiz
            # Formasyona göre renk seç
            color = {
                'up': 'rgba(34, 197, 94, 0.7)',
                'down': 'rgba(239, 68, 68, 0.7)',
                'either': 'rgba(168, 85, 247, 0.7)',
                'reversal': 'rgba(251, 191, 36, 0.7)'
            }.get(pat.get('direction', 'either'), 'rgba(156, 163, 175, 0.7)')
            
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=pat["name"],
                    line=dict(
                        width=2.5 if i == 0 else 2,
                        dash="solid" if i == 0 else "dot",
                        color=color
                    ),
                    marker=dict(size=10 if i == 0 else 8, color=color),
                    showlegend=True
                )
            )
            
            # Harfleri ekle
            letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
            for j, (x_val, y_val) in enumerate(zip(xs, ys)):
                if j < len(letters):
                    fig.add_annotation(
                        x=x_val,
                        y=y_val,
                        text=f"<b>{letters[j]}</b>",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-25,
                        bgcolor=color.replace('0.7', '0.9'),
                        bordercolor=color,
                        borderwidth=1,
                        font=dict(size=11, color="white"),
                    )
            
            # Kritik seviyeleri göster
            if 'neckline' in pat and pat['neckline'] is not None:
                fig.add_hline(
                    y=pat['neckline'],
                    line_dash="dash",
                    line_color=color,
                    line_width=1,
                    annotation_text=f"{pat['name']} neckline: {pat['neckline']:.1f}",
                    annotation_position="right",
                )
            
            if 'support' in pat and pat['support'] is not None:
                fig.add_hline(
                    y=pat['support'],
                    line_dash="dot",
                    line_color="rgba(34, 197, 94, 0.5)",
                    annotation_text=f"Destek: {pat['support']:.1f}",
                )
            
            if 'resistance' in pat and pat['resistance'] is not None:
                fig.add_hline(
                    y=pat['resistance'],
                    line_dash="dot",
                    line_color="rgba(239, 68, 68, 0.5)",
                    annotation_text=f"Direnç: {pat['resistance']:.1f}",
                )

    # === Y EKSENİ & LAYOUT ===
    y_min = float(df_graph[LOW_COL].min())
    y_max = float(df_graph[HIGH_COL].max())
    y_margin = (y_max - y_min) * 0.08
    fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])

    fig.update_layout(
        height=520,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        hovermode="x unified",
        xaxis=dict(showgrid=False, showticklabels=False, rangeslider_visible=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(51,65,85,0.7)",
            side="right",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.75)",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})# --------- SAĞ: EMA & S/R MESAFE KUTULARI ---------
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

    if st.button("▶️ Simülasyonu Çalıştır"):
        with st.spinner("Model sinyallerini simüle ediyor..."):
            trades_df = simulate_model_trades(
                df,
                model_bundle,
                thr_buy=thr_buy_bt,
                thr_sell=thr_sell_bt,
                horizon_bars=horizon_bars,
            )

        if trades_df.empty:
            st.warning("Hiç işlem oluşmadı. Eşikler çok agresif olabilir, biraz gevşet.")
        else:
            # Genel istatistikler
            total_trades = len(trades_df)
            wins = (trades_df["pnl_points"] > 0).sum()
            losses = (trades_df["pnl_points"] < 0).sum()
            win_rate = wins / total_trades * 100.0

            avg_pnl = trades_df["pnl_points"].mean()
            avg_max_fav = trades_df["max_favorable"].mean()
            avg_max_adv = trades_df["max_adverse"].mean()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Toplam İşlem", total_trades)
            with c2:
                st.metric("Kazanç Oranı", f"{win_rate:.1f} %")
            with c3:
                st.metric("Ortalama PnL (puan)", f"{avg_pnl:.1f}")
            with c4:
                st.metric("Ort. Max Lehimize", f"{avg_max_fav:.1f} / Ort. Max Aleyhimize {avg_max_adv:.1f}")

            st.markdown("#### Son 20 İşlemin Detayı")
            st.dataframe(
                trades_df.sort_values("entry_time", ascending=False)
                .head(20)[
                    [
                        "entry_time",
                        "direction",
                        "entry_price",
                        "exit_price",
                        "pnl_points",
                        "max_favorable",
                        "max_adverse",
                        "minutes_in_profit",
                        "minutes_in_loss",
                        "confidence",
                    ]
                ]
            )

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
                st.success(f"İşlem kayıtları `{save_path}` dosyasına kaydedildi. (Gelecekte eğitimde kullanabilirsin.)")
            except Exception as e:
                st.warning(f"Kayıt dosyasına yazarken hata oldu: {e}")
# ================== EN ALT: ZAMAN BİLGİSİ ==================
st.markdown("---")

c_t1, c_t2, c_t3 = st.columns([1, 2, 1])
with c_t2:
    last_time = last_row[TIME_COL]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"<center>🕒 Son bar zamanı: <b>{last_time}</b> &nbsp;|&nbsp; Panel güncelleme: <b>{now_str}</b></center>",
        unsafe_allow_html=True,
    )