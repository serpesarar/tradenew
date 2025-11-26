import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import json

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
EVENTS_PARQUET = "./staging/nasdaq_event_outcomes_v1.parquet"  # Training v2 ile aynı
MODEL_PATH = "./models/nasdaq_event_outcome_xgb_v2.pkl"
SCALER_PATH = "./models/nasdaq_event_outcome_scaler_v2.pkl"
FEATURE_LIST_PATH = "./models/nasdaq_event_outcome_features_v2.json"
LABEL_ENCODERS_PATH = "./models/nasdaq_event_outcome_label_encoders_v2.pkl"
MASTER_PARQUET = "./staging/nasdaq_master_features_v1.parquet"
OUTPUT_PARQUET = "./staging/nasdaq_event_outcomes_with_preds_v2.parquet"
OUTPUT_CSV = "./staging/nasdaq_event_outcomes_with_preds_v2.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("event_outcome_inference_batch_v2")

# Model sınıf id'leri için fallback label mapping
CLASS_ID_TO_LABEL = {
    0: "CHOP",
    1: "UP",
    2: "DOWN",
}


def load_model_and_meta():
    logger.info("🧠 v2 model + scaler + feature_list + label_encoders yükleniyor...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_LIST_PATH, "r") as f:
        feature_list = json.load(f)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    logger.info("   ✅ Model: %s", MODEL_PATH)
    logger.info("   ✅ Scaler: %s", SCALER_PATH)
    logger.info("   ✅ Feature count (v2): %d", len(feature_list))
    logger.info("   ✅ Label encoders (v2) yüklendi.")
    return model, scaler, feature_list, label_encoders


def decode_class_label(class_id, label_encoders) -> str:
    """
    Model sınıf id'sini string label'a çevir.
    LabelEncoder varsa onu kullan, yoksa fallback dictionary'e bak.
    """
    future_le = label_encoders.get("future_dir") if label_encoders else None

    if future_le is not None:
        try:
            decoded = future_le.inverse_transform([class_id])[0]
            return str(decoded)
        except Exception:
            pass

    try:
        return CLASS_ID_TO_LABEL[int(class_id)]
    except (ValueError, KeyError, TypeError):
        return str(class_id)


def main():
    logger.info("=" * 78)
    logger.info("🚀 NASDAQ EVENT OUTCOME INFERENCE BATCH v2 BAŞLIYOR")
    logger.info("=" * 78)

    logger.info("📥 Event outcomes yükleniyor: %s", EVENTS_PARQUET)
    events_df = pd.read_parquet(EVENTS_PARQUET)
    logger.info("   ✅ Event dataset shape: %s", events_df.shape)

    logger.info("📥 Master features yükleniyor: %s", MASTER_PARQUET)
    master_df = pd.read_parquet(MASTER_PARQUET)
    logger.info("   ✅ Master df shape: %s", master_df.shape)

    logger.info("🔗 Events + master merge ediliyor (training v2 ile aynı mantık)...")
    df = events_df.merge(master_df, on="timestamp", how="left", suffixes=("", "_m"))
    logger.info("   ✅ merged shape: %s", df.shape)

    # 🔽🔽🔽 Wave feature engineering (training v2 ile aynı mantık) 🔽🔽🔽
    logger.info("🌊 Wave feature engineering (_m suffix kolonları oluşturuluyor)...")
    
    # Timestamp'e göre sırala
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Wave base kolonları
    wave_base_cols = [
        "signal_wave",
        "wave_strength_pips",
        "wave_duration_bars",
        "up_move_pips",
        "down_move_pips",
        "up_duration_bars",
        "down_duration_bars",
    ]
    
    # Her wave kolonu için _m versiyonunu oluştur (groupby event_type ile rolling mean)
    for col in wave_base_cols:
        if col in df.columns:
            # event_type'a göre groupby yapıp rolling mean al
            df[f"{col}_m"] = (
                df[col]
                .groupby(df["event_type"])
                .transform(lambda x: x.rolling(50, min_periods=5).mean())
            )
            logger.info(f"   ✅ {col}_m oluşturuldu")
        else:
            logger.warning(f"   ⚠️ {col} kolonu bulunamadı, {col}_m oluşturulamadı")
    
    # Wave kolonlarını debug için logla
    wave_cols_debug = [c for c in df.columns if "wave_" in c or "signal_wave" in c]
    logger.info(f"   🔍 Wave ilgili kolonlar (toplam {len(wave_cols_debug)}): {wave_cols_debug[:30]}")
    
    model, scaler, feature_list, label_encoders = load_model_and_meta()

    # DEBUG: Hangi feature’lar eksik, önce loglayalım
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        logger.error("❌ df içinde eksik feature kolonları var. İlk 50 tanesi: %s", missing[:50])
        raise ValueError("df, feature_list ile uyumlu değil (eksik kolonlar var).")
    
    X = df[feature_list].astype(float).copy()
    
    logger.info("🧠 v2 model ile inference başlıyor (tüm eventler)...")
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    classes = model.classes_
    class_labels = [decode_class_label(c, label_encoders).upper() for c in classes]
    
    # proba kolonları: p_CHOP, p_DOWN, p_UP
    proba_df = pd.DataFrame(proba, columns=[f"p_{label}" for label in class_labels])
    
    # UP/DOWN/CHOP isimlerini normalize et
    def safe_get(col, default=0.0):
        return proba_df[col] if col in proba_df.columns else default
    
    df["p_chop"] = safe_get("p_CHOP")
    df["p_up"] = safe_get("p_UP")
    df["p_down"] = safe_get("p_DOWN")
    
    pred_idx = proba_df.values.argmax(axis=1)
    pred_class = classes[pred_idx]
    df["pred_class"] = pred_class
    df["pred_label"] = [class_labels[i] for i in pred_idx]

    df["max_prob"] = proba_df.max(axis=1)

    # Recommendation:
    def decide_reco(row):
        if row["max_prob"] < 0.60:
            return "PASS"
        if row["pred_label"] == "UP":
            return "LONG"
        if row["pred_label"] == "DOWN":
            return "SHORT"
        return "PASS"

    df["recommendation"] = df.apply(decide_reco, axis=1)

    logger.info("   ✅ Inference bitti. shape=%s", df.shape)
    logger.info(
        "   ✅ Tahmin label dağılımı (v2):\n%s",
        df["pred_label"].value_counts(dropna=False),
    )

    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("💾 Kaydedildi (Parquet): %s", OUTPUT_PARQUET)
    logger.info("💾 Kaydedildi (CSV)    : %s", OUTPUT_CSV)
    logger.info("=" * 78)
    logger.info("✅ NASDAQ EVENT OUTCOME INFERENCE BATCH v2 TAMAMLANDI")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()