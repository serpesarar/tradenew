import logging
import joblib
from typing import Dict, Any

import numpy as np
import pandas as pd

# ============================================================
# PATH AYARLARI – TRAIN ARTEFACTLERLE UYUMLU
# ============================================================

MODEL_PATH = "./models/nasdaq_event_outcome_xgb_v1.pkl"
SCALER_PATH = "./models/nasdaq_event_outcome_scaler_v1.pkl"
FEATURES_PATH = "./models/nasdaq_event_outcome_features_v1.pkl"
ENCODERS_PATH = "./models/nasdaq_event_outcome_encoders_v1.pkl"

# 0/1/2 → label mapping
CLASS_ID_TO_LABEL = {
    0: "CHOP",
    1: "UP",
    2: "DOWN",
}

# Recommendation için threshold
THRESHOLD = 0.6

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# YARDIMCI: SAFE LABEL ENCODE
# ============================================================

def _safe_label_encode(le, value) -> int:
    """
    LabelEncoder için güvenli encode:
      - None / NaN → "UNKNOWN"
      - Eğitimde görmediği kategori gelirse:
          - Eğer 'UNKNOWN' varsa ona map et
          - Yoksa classes_[0]'a map et
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        v = "UNKNOWN"
    else:
        v = str(value)

    if v not in le.classes_:
        if "UNKNOWN" in le.classes_:
            v = "UNKNOWN"
        else:
            v = le.classes_[0]

    return le.transform([v])[0]


# ============================================================
# ENGINE YÜKLEME
# ============================================================

def load_event_outcome_engine() -> Dict[str, Any]:
    """
    XGBoost event outcome model + scaler + encoders + feature_names
    hepsini yükler ve dict döner.
    
    NOT: Dosyalar joblib ile kaydedildiği için joblib.load() kullanıyoruz.
    """
    logger.info("🧠 Event outcome engine yükleniyor...")

    model = joblib.load(MODEL_PATH)
    logger.info("   ✅ Model yüklendi: %s", MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)
    logger.info("   ✅ Scaler yüklendi: %s", SCALER_PATH)

    feature_names = joblib.load(FEATURES_PATH)
    logger.info("   ✅ Feature list yüklendi (%d feature)", len(feature_names))

    encoders = joblib.load(ENCODERS_PATH)
    logger.info("   ✅ Label encoders yüklendi (%d kolon)", len(encoders))

    engine = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "encoders": encoders,
    }

    logger.info("🧠 Event outcome engine hazır.")
    return engine


# ============================================================
# FEATURE HAZIRLAMA (BATCH)
# ============================================================

def prepare_features_for_model(
    df_raw: pd.DataFrame,
    feature_names,
    encoders: Dict[str, Any],
) -> pd.DataFrame:
    """
    df_raw: TRAIN sırasında kullandığın event dataset ile
            kolon isimleri uyumlu DataFrame olmalı
            (event_type, entry_price, signal_wave vs.)

    Çıktı: modelin beklediği numeric X (n_samples x n_features)
    """

    df = df_raw.copy()

    # 1) String / kategorik kolonları encode et
    for col, le in encoders.items():
        if col not in df.columns:
            logger.warning("   ⚠️ encode edilecek kolon eksik: %s – UNKNOWN ile dolduruluyor", col)
            df[col] = "UNKNOWN"

        df[col] = df[col].apply(lambda v: _safe_label_encode(le, v))

    # 2) Eksik feature varsa 0 ile doldur
    for col in feature_names:
        if col not in df.columns:
            logger.warning("   ⚠️ feature eksik: %s – 0.0 ile dolduruluyor", col)
            df[col] = 0.0

    # 3) Sadece modelin beklediği kolonları al
    X = df[feature_names].copy()

    # 4) Tip güvenliği
    for col in X.columns:
        if X[col].dtype == "O":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X


# ============================================================
# ANA FONKSİYON: BATCH INFERENCE
# ============================================================

def run_event_outcome_inference(
    df_events: pd.DataFrame,
    engine: Dict[str, Any],
) -> pd.DataFrame:
    """
    🔹 Tek ana fonksiyonun bu:
       - df_events: event satırlarını içeren DataFrame
         (TRAIN'de kullandığın kolon yapısıyla uyumlu)
       - engine: load_event_outcome_engine() çıktısı

       Çıktı: df_events + prediction sütunları
    """
    model = engine["model"]
    scaler = engine["scaler"]
    feature_names = engine["feature_names"]
    encoders = engine["encoders"]

    if df_events.empty:
        logger.warning("⚠️ Uyarı: df_events boş, inference yapılmadı.")
        return df_events.copy()

    # 1) Feature hazırlığı
    X = prepare_features_for_model(df_events, feature_names, encoders)

    # 2) Scale
    X_scaled = scaler.transform(X)

    # 3) Probabilistic prediction
    proba = model.predict_proba(X_scaled)  # shape: (n_samples, 3)

    p_chop = proba[:, 0].astype(float)
    p_up = proba[:, 1].astype(float)
    p_down = proba[:, 2].astype(float)

    pred_classes = proba.argmax(axis=1).astype(int)
    max_probs = proba.max(axis=1).astype(float)
    pred_labels = [CLASS_ID_TO_LABEL.get(int(c), str(c)) for c in pred_classes]

    # 4) Çıktı DataFrame
    df_out = df_events.copy()
    df_out["p_chop"] = p_chop
    df_out["p_up"] = p_up
    df_out["p_down"] = p_down
    df_out["pred_class"] = pred_classes
    df_out["pred_label"] = pred_labels
    df_out["max_prob"] = max_probs

    # Varsayım: buraya kadar şunlar var:
    # df_preds["p_chop"], df_preds["p_up"], df_preds["p_down"]
    # df_preds["pred_class"], df_preds["pred_label"], df_preds["max_prob"]

    # --- RECOMMENDATION LOGIC (THRESHOLD=0.6) ---
    # kural:
    # 1) max_prob < 0.6  → PASS
    # 2) pred_label == CHOP → PASS
    # 3) pred_label == UP   → LONG
    # 4) pred_label == DOWN → SHORT

    cond_pass = (df_out["max_prob"] < THRESHOLD) | (df_out["pred_label"] == "CHOP")

    df_out["recommendation"] = np.where(
        cond_pass,
        "PASS",
        np.where(
            df_out["pred_label"] == "UP",
            "LONG",
            np.where(df_out["pred_label"] == "DOWN", "SHORT", "PASS"),
        ),
    )

    return df_out


# ============================================================
# SELF-TEST (opsiyonel)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("🔬 SELF-TEST: Event outcomes'tan mini inference denemesi")

    try:
        df_events = pd.read_parquet("./staging/nasdaq_event_outcomes_v1.parquet")
    except Exception as e:
        logger.error("Event outcomes parquet okunamadı: %s", e)
        raise SystemExit(1)

    if df_events.empty:
        logger.error("Event outcomes boş, test iptal.")
        raise SystemExit(1)

    # küçük bir subset alalım
    df_sample = df_events.head(5).copy()

    engine = load_event_outcome_engine()
    df_pred = run_event_outcome_inference(df_sample, engine)

    logger.info("✅ Inference sonucu kolonlar: %s", df_pred.columns.tolist())
    logger.info("✅ İlk satır prediction: %s", df_pred.iloc[0][["p_chop", "p_up", "p_down", "pred_label", "recommendation"]])