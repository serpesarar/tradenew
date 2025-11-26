#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ================================
# PATH & PARAMS (GÜNCEL)
# ================================

# 1) Haber impact event datası
EVENT_PATH   = "./staging/nasdaq_news_impact_events.parquet"
MODEL_PATH   = "./models/nasdaq_news_impact_dir_xgb.pkl"
SCALER_PATH  = "./models/nasdaq_news_impact_scaler.pkl"
FEATS_PATH   = "./models/nasdaq_news_impact_features.pkl"

# 2) Fiyat + wave + daily macro ana dataset
# 👉 Artık BASE_PATH olarak WAVE V3 dosyasını kullanıyoruz
BASE_PATH        = "./staging/nasdaq_full_wave_v3.parquet"
WAVE_DATA_PATH   = "./staging/nasdaq_full_wave_v3.parquet"

# 3) Wave direction modeli (v3)
WAVE_MODEL_PATH  = "./models/nasdaq_wave_dir_v3_xgb.pkl"
WAVE_SCALER_PATH = "./models/nasdaq_wave_v3_scaler.pkl"
WAVE_FEATS_PATH  = "./models/nasdaq_wave_v3_features.pkl"

# 4) News impact modeli için ayarlar
CONF_THRESH  = 0.75   # news için high-confidence eşiği
PIP_COL      = "fut_pips_mid"
LABEL_COL    = "impact_dir_mid"

# 5) Label map’ler
LABEL_MAP_NEWS = {
    0: "CHOP",
    1: "BULLISH",
    2: "BEARISH",
}

LABEL_MAP_WAVE = {
    0: "CHOP",
    1: "LONG_WAVE",
    2: "SHORT_WAVE",
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def load_news_model_and_scaler():
    """News impact modeli + scaler + feature list yükler ve boyutları döner."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"News modeli yok: {MODEL_PATH}")
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(f"News scaler yok: {SCALER_PATH}")
    if not Path(FEATS_PATH).exists():
        raise FileNotFoundError(f"News feature list yok: {FEATS_PATH}")

    model: XGBClassifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_list = joblib.load(FEATS_PATH)

    # scaler kaç feature ile fit edilmiş
    scaler_n = getattr(scaler, "n_features_in_", len(feature_list))

    # xgboost içinden gerçek feature sayısı
    booster = model.get_booster()
    model_n = int(booster.num_features())

    return model, scaler, feature_list, model_n, scaler_n


def scale_for_model(X: pd.DataFrame, scaler, model_n: int, scaler_n: int) -> np.ndarray:
    """
    Scaler 185, model 184 feature bekliyorsa:
      - önce tüm 185 feature ile scaler.transform
      - sonra X_scaled[:, :model_n] model'e ver
    Hiçbir şeyi silmeyip, sadece X'i doğru şekle getiriyoruz.
    """
    # scaler input'u scaler_n feature olmalı
    if X.shape[1] != scaler_n:
        # eksik kolonu events'ten zaten yakalamış oluruz, burada sadece assert gibi düşün
        raise ValueError(
            f"Scaler {scaler_n} feature bekliyor ama X {X.shape[1]} feature içeriyor."
        )

    X_scaled_full = scaler.transform(X)  # (N, scaler_n)

    # Model daha az feature bekliyorsa son sütunları at
    if scaler_n > model_n:
        X_scaled = X_scaled_full[:, :model_n]
    else:
        X_scaled = X_scaled_full

    return X_scaled


def prepare_news_features(events: pd.DataFrame):
    """Event datasından X (news modeli için) hazırlar, scaler + model boyutları ile birlikte döner."""
    model, scaler, feature_list, model_n, scaler_n = load_news_model_and_scaler()

    # Feature list tüm kolonlarda olmalı
    missing_feats = [f for f in feature_list if f not in events.columns]
    if missing_feats:
        raise ValueError(
            f"Event datasında eksik feature var (ilk 10): {missing_feats[:10]}"
        )

    X_full = events[feature_list].copy()
    return model, scaler, feature_list, model_n, scaler_n, X_full


# ============================================================
# A) NEWS IMPACT HIGH-CONFIDENCE ANALYSIS
# ============================================================

def run_high_confidence_analysis():
    print("=" * 80)
    print("🚀 NASDAQ NEWS IMPACT HIGH-CONFIDENCE ANALYSIS")
    print("=" * 80)

    # 1) Event data
    if not Path(EVENT_PATH).exists():
        raise FileNotFoundError(f"Event file yok: {EVENT_PATH}")

    events = pd.read_parquet(EVENT_PATH)
    print(f"   ✅ Event data shape: {events.shape}")

    if PIP_COL not in events.columns:
        raise ValueError(f"{PIP_COL} kolonu event datasında yok.")

    # 2) Model + scaler + feature list
    model, scaler, feature_list, model_n, scaler_n, X_full = prepare_news_features(events)

    print(f"   ✅ Model yüklendi: {MODEL_PATH}")
    print(f"   ✅ Scaler yüklendi: {SCALER_PATH}")
    print(f"   ✅ Feature list yüklendi: {FEATS_PATH}")
    print(f"   🔢 Feature list length (file): {len(feature_list)}")
    print(f"   🔢 Model beklenen feature sayısı: {model_n}")
    print(f"   🔢 Scaler beklenen feature sayısı: {scaler_n}")
    print(f"   ✅ X_full shape (scaler input): {X_full.shape}")

    # 3) Scale + predict_proba
    # Burada hiçbir şeyi silmiyoruz, sadece scaler+model boyutlarını uyumlu hale getiriyoruz
    from sklearn.utils.validation import _is_arraylike_not_scalar  # sadece uyarı için (opsiyonel)

    X_scaled = scale_for_model(X_full, scaler, model_n, scaler_n)
    print(f"   ✅ X_scaled shape (model input): {X_scaled.shape}")

    print("\n   🔮 Predict_proba hesaplanıyor...")

    proba = model.predict_proba(X_scaled)
    pred_class = proba.argmax(axis=1)
    max_conf = proba.max(axis=1)

    events["pred_class"] = pred_class
    events["pred_label"] = events["pred_class"].map(LABEL_MAP_NEWS)
    events["pred_conf"] = max_conf

    total_events = len(events)
    hc_mask = events["pred_conf"] >= CONF_THRESH
    hc_events = events[hc_mask].copy()
    hc_n = len(hc_events)

    print(f"\n   🔢 Toplam event sayısı: {total_events:,}")
    print(
        f"   🎯 High-confidence (>= {CONF_THRESH:.2f}) event sayısı: {hc_n:,} "
        f"({hc_n/total_events*100:.2f}%)"
    )

    # 4) Class bazlı pip istatistikleri
    print("\n" + "-" * 80)
    print("📊 HIGH-CONFIDENCE CLASS-BAZLI PIP İSTATİSTİKLERİ (mid horizon)")
    print("-" * 80)

    for cid, name in LABEL_MAP_NEWS.items():
        sub = hc_events[hc_events["pred_class"] == cid]
        if sub.empty:
            print(f"   Class {cid} ({name}): n=0")
            continue

        p50_signed = sub[PIP_COL].quantile(0.5)
        p50_abs = sub[PIP_COL].abs().quantile(0.5)
        p90_abs = sub[PIP_COL].abs().quantile(0.9)

        print(
            f"   Class {cid} ({name}): "
            f"n={len(sub):5d} | "
            f"P50_signed={p50_signed:.1f} pip | "
            f"P50_abs={p50_abs:.1f} pip | "
            f"P90_abs={p90_abs:.1f} pip"
        )

    # 5) Son 5 high-confidence örnek
    print("\n" + "-" * 80)
    print("🧾 SON 5 HIGH-CONFIDENCE EVENT ÖRNEĞİ")
    print("-" * 80)

    time_cols = [c for c in ["event_time", "timestamp", "datetime"] if c in hc_events.columns]
    show_cols = []
    if time_cols:
        show_cols.append(time_cols[0])
    show_cols += ["pred_label", "pred_conf", PIP_COL]
    show_cols = [c for c in show_cols if c in hc_events.columns]

    if not hc_events.empty:
        print(
            hc_events.sort_values(time_cols[0] if time_cols else "pred_conf")
                     .tail(5)[show_cols]
        )
    else:
        print("   (High-confidence event yok)")

    print("\n" + "=" * 80)
    print("✅ NASDAQ NEWS IMPACT HIGH-CONFIDENCE ANALYSIS BİTTİ")
    print("=" * 80)

    # Sonuç datasını geri döndür, signal card tarafında kullanacağız
    return events, model, scaler, feature_list, model_n, scaler_n


# ============================================================
# B) SIGNAL CARD (SON BAR + SON NEWS)
# ============================================================

def build_nasdaq_signal_card(
    events: pd.DataFrame,
    news_model,
    news_scaler,
    news_feature_list,
    news_model_n: int,
    news_scaler_n: int,
):
    """
    Son M30 bar + son news event için:
      - Wave v3 model tahmini
      - News impact model tahmini
      - Kombine bir yorum döner (dict + panel text)
    Hiçbir feature veya modeli silmiyoruz; sadece join + hesap yapıyoruz.
    """

    print("\n" + "=" * 80)
    print("🧪 NASDAQ SIGNAL CARD DEMO (LAST BAR + LAST NEWS)")
    print("=" * 80)

    # ---------- 1) BASE / WAVE TARAFI ----------
    if not Path(BASE_PATH).exists():
        raise FileNotFoundError(f"Base M30 dosyası yok: {BASE_PATH}")
    if not Path(WAVE_DATA_PATH).exists():
        raise FileNotFoundError(f"Wave v3 data yok: {WAVE_DATA_PATH}")
    if not Path(WAVE_MODEL_PATH).exists():
        raise FileNotFoundError(f"Wave v3 model yok: {WAVE_MODEL_PATH}")
    if not Path(WAVE_SCALER_PATH).exists():
        raise FileNotFoundError(f"Wave v3 scaler yok: {WAVE_SCALER_PATH}")
    if not Path(WAVE_FEATS_PATH).exists():
        raise FileNotFoundError(f"Wave v3 feature list yok: {WAVE_FEATS_PATH}")

    base = pd.read_parquet(BASE_PATH)
    wave_data = pd.read_parquet(WAVE_DATA_PATH)
    wave_model: XGBClassifier = joblib.load(WAVE_MODEL_PATH)
    wave_scaler = joblib.load(WAVE_SCALER_PATH)
    wave_feats = joblib.load(WAVE_FEATS_PATH)

    # timestamp eşleştirme
    dt_col = None
    for c in ["timestamp", "datetime", "time"]:
        if c in base.columns:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError("Base data içinde timestamp/datetime yok.")

    base[dt_col] = pd.to_datetime(base[dt_col])
    base = base.sort_values(dt_col).reset_index(drop=True)

    # wave_data'da da datetime/timestamp bul
    wave_dt_col = None
    for c in ["timestamp", "datetime", "time"]:
        if c in wave_data.columns:
            wave_dt_col = c
            break
    if wave_dt_col is None:
        raise ValueError("Wave data içinde datetime yok.")

    wave_data[wave_dt_col] = pd.to_datetime(wave_data[wave_dt_col])
    wave_data = wave_data.sort_values(wave_dt_col).reset_index(drop=True)

    last_ts = base[dt_col].iloc[-1]
    # en yakın wave kaydını al (<= last_ts)
    wave_last = wave_data[wave_data[wave_dt_col] <= last_ts]
    if wave_last.empty:
        raise ValueError("Wave datasında son bara karşılık gelen kayıt bulunamadı.")
    wave_last = wave_last.iloc[-1]

    # wave strength & duration
    wave_strength_pips = float(wave_last.get("wave_strength_pips", 0.0))
    wave_duration_bars = float(wave_last.get("wave_duration_bars", 0.0))
    wave_class_id = int(wave_last.get("signal_wave", 0))
    wave_label = LABEL_MAP_WAVE.get(wave_class_id, "CHOP")

    # wave proba için modelden geçirelim
    missing_wave_feats = [f for f in wave_feats if f not in base.columns]
    if missing_wave_feats:
        raise ValueError(
            f"Base datasında eksik wave feature var (ilk 10): {missing_wave_feats[:10]}"
        )

    X_wave = base[wave_feats].iloc[[-1]].copy()
    X_wave_scaled = wave_scaler.transform(X_wave)
    wave_proba = wave_model.predict_proba(X_wave_scaled)[0]
    wave_conf = float(wave_proba[wave_class_id])

    # crude meta P50/P90 (XAUUSD'deki gibi kalibrasyon yok, ama mantıklı ölçek)
    abs_raw = abs(wave_strength_pips)
    meta_p50 = abs_raw * 0.35
    meta_p90 = abs_raw * 0.9
    sign = 0
    if wave_class_id == 1:
        sign = 1
    elif wave_class_id == 2:
        sign = -1

    meta_p50_dir = sign * meta_p50
    meta_p90_dir = sign * meta_p90

    # ---------- 2) NEWS TARAFI ----------
    # Son news event
    news = events.copy()
    # zaman sütunu
    news_dt_col = None
    for c in ["event_time", "timestamp", "datetime", "time"]:
        if c in news.columns:
            news_dt_col = c
            break
    if news_dt_col is None:
        raise ValueError("News events içinde zaman kolonu yok.")

    news[news_dt_col] = pd.to_datetime(news[news_dt_col])
    news = news.sort_values(news_dt_col).reset_index(drop=True)

    news_last = news.iloc[[-1]].copy()

    # X_full (scaler input)
    missing_news_feats = [f for f in news_feature_list if f not in news_last.columns]
    if missing_news_feats:
        raise ValueError(
            f"News event datasında eksik feature var (ilk 10): {missing_news_feats[:10]}"
        )

    X_news_full = news_last[news_feature_list].copy()
    X_news_scaled = scale_for_model(X_news_full, news_scaler, news_model_n, news_scaler_n)

    news_proba = news_model.predict_proba(X_news_scaled)[0]
    news_class_id = int(news_proba.argmax())
    news_label = LABEL_MAP_NEWS.get(news_class_id, "CHOP")
    news_conf = float(news_proba[news_class_id])

    fut_pips_mid = float(news_last.get(PIP_COL, np.nan))

    # ---------- 3) KOM BİN E  ----------
    # Kombine confidence text
    if wave_conf >= 0.90 and news_conf >= 0.70:
        combo_conf_text = "YÜKSEK"
    elif wave_conf >= 0.80 and news_conf >= 0.55:
        combo_conf_text = "ORTA"
    else:
        combo_conf_text = "DÜŞÜK"

    # Kombine target pips: wave P50 yönlü
    combined_target_pips = meta_p50_dir
    combined_p90_pips = meta_p90_dir

    # Panel text (Türkçe)
    # dalga süresi saat olarak
    wave_hours = wave_duration_bars * 0.5  # 30m bar → 0.5 saat

    # Yön text
    wave_dir_text = "YUKARI TREND" if wave_class_id == 1 else (
        "AŞAĞI TREND" if wave_class_id == 2 else "YATAY/CHOP"
    )

    panel_text = []
    panel_text.append(f"Sinyal (Wave): {wave_label} ({wave_dir_text})")
    panel_text.append(
        f"Tahmini pip hedefi (P50): {combined_target_pips:+.0f} pip "
        f"(P90: {combined_p90_pips:+.0f} pip)"
    )
    panel_text.append(
        f"Tahmini dalga süresi: ~{wave_duration_bars:.0f} bar (≈ {wave_hours:.1f} saat)"
    )
    panel_text.append(f"Wave yön güveni (model proba): {wave_conf*100:.1f}%")

    panel_text.append("")
    panel_text.append(
        f"Haber etkisi (mid horizon): {news_label} (güven: {news_conf*100:.1f}%)"
    )
    if not np.isnan(fut_pips_mid):
        panel_text.append(
            f"Haber bazlı tipik hareket: gerçekleşen mid-horizon ≈ {fut_pips_mid:+.1f} pip"
        )
    panel_text.append("")
    panel_text.append(f"Kombine güven yorumu: {combo_conf_text}")
    panel_text.append(f"- Fiyat dalga modeli: {wave_dir_text}")
    panel_text.append(f"- Haber etkisi: {news_label}")

    panel_text_str = "\n".join(panel_text)

    card = {
        "wave": {
            "class_id": wave_class_id,
            "label": wave_label,
            "proba": wave_conf,
            "raw_strength_pips": wave_strength_pips,
            "raw_duration_bars": wave_duration_bars,
            "meta_p50": abs(meta_p50),
            "meta_p90": abs(meta_p90),
        },
        "news": {
            "class_id": news_class_id,
            "label": news_label,
            "confidence": news_conf,
            "fut_pips_mid": fut_pips_mid,
        },
        "combined": {
            "target_pips": combined_target_pips,
            "p90_pips": combined_p90_pips,
            "confidence_text": combo_conf_text,
            "panel_text": panel_text_str,
        },
    }

    print(panel_text_str)
    print("\n--- RAW CARD JSON ---")
    print(card)

    return card


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Aşama 1: High-confidence reporting (hiçbir şeyi kısaltmıyoruz)
    events, news_model, news_scaler, news_feature_list, news_model_n, news_scaler_n = (
        run_high_confidence_analysis()
    )

    # Aşama 2: Signal card demo (son bar + son haber)
    card = build_nasdaq_signal_card(
        events=events,
        news_model=news_model,
        news_scaler=news_scaler,
        news_feature_list=news_feature_list,
        news_model_n=news_model_n,
        news_scaler_n=news_scaler_n,
    )