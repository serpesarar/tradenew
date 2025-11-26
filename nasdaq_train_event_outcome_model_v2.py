import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EVENT_OUTCOMES_PATH = "./staging/nasdaq_event_outcomes_v1.parquet"
MASTER_FEATURES_PATH = "./staging/nasdaq_master_features_v1.parquet"

MODEL_PATH = "./models/nasdaq_event_outcome_xgb_v2.pkl"
SCALER_PATH = "./models/nasdaq_event_outcome_scaler_v2.pkl"
FEATURE_LIST_PATH = "./models/nasdaq_event_outcome_features_v2.json"
LABEL_ENCODERS_PATH = "./models/nasdaq_event_outcome_label_encoders_v2.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2  # time-split yerine basit holdout; zaman bazlı istersek aşağıda değiştiririz.

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_event_outcome_v2")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 78)
    logger.info("🚀 EVENT OUTCOME MODEL v2 TRAINING BAŞLIYOR")
    logger.info("=" * 78)
    logger.info("📥 Event outcomes yükleniyor: %s", EVENT_OUTCOMES_PATH)
    events = pd.read_parquet(EVENT_OUTCOMES_PATH)
    logger.info("   ✅ events shape: %s", events.shape)

    logger.info("📥 Master features yükleniyor: %s", MASTER_FEATURES_PATH)
    master = pd.read_parquet(MASTER_FEATURES_PATH)
    logger.info("   ✅ master shape: %s", master.shape)

    # timestamp’ı datetime yap
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    master["timestamp"] = pd.to_datetime(master["timestamp"])

    return events, master


def merge_events_with_features(events: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    Event outcomes ile master feature'ları timestamp üzerinden merge eder.
    (Her event için aynı barın tüm feature’larını ekliyoruz.)
    """
    logger.info("🔗 Events + master merge ediliyor (timestamp üzerinde)...")

    # master sadece feature tarafı (gereksiz kolon varsa atarız)
    merged = events.merge(master, on="timestamp", how="left", suffixes=("", "_m"))

    logger.info("   ✅ merged shape: %s", merged.shape)
    missing_feat_rows = merged.isna().all(axis=1).sum()
    if missing_feat_rows > 0:
        logger.warning("⚠️ %d satırda tüm feature'lar NaN (merge sonrası) – timestamp uyumsuzluğu olabilir.", missing_feat_rows)

    return merged


def prepare_features_and_labels(merged: pd.DataFrame):
    """
    v2 için feature matrisi (X) ve label (y) hazırlar.
    DOWN tarafını özellikle güçlendirmek için class ağırlıklarını burada hesaplayacağız.
    """
    df = merged.copy()

    # --- Label kolonunu belirle ---
    # future_dir hem numeric hem string olabilir; önce string label üretelim.
    if "future_dir_label" in df.columns:
        y_raw = df["future_dir_label"].astype(str)
    else:
        # future_dir numeric ise onu label'a map'leriz; değilse string kabul ederiz
        if "future_dir" not in df.columns:
            raise ValueError("❌ 'future_dir' veya 'future_dir_label' kolonu bulunamadı, label çıkaramıyorum.")

        if np.issubdtype(df["future_dir"].dtype, np.number):
            # 0/1/2 gibi ise kaba mapping
            mapping = {0: "CHOP", 1: "UP", 2: "DOWN"}
            y_raw = df["future_dir"].map(mapping).fillna("CHOP").astype(str)
        else:
            # Zaten string ise direkt al
            y_raw = df["future_dir"].astype(str)

    logger.info("   ✅ Label dağılımı (y_raw):\n%s", y_raw.value_counts(dropna=False))

    # --- Leakage ve meta kolonları exclude list ---
    meta_cols = [
        "timestamp",
    ]

    possible_leak_cols = [
        "future_dir",
        "future_dir_label",
        "tp_sl_result",
        "tp_sl_result_label",
        "max_up_move_pips",
        "max_down_move_pips",
        "p_chop",
        "p_up",
        "p_down",
        "pred_class",
        "pred_label",
        "max_prob",
        "recommendation",
    ]

    leak_cols = [c for c in possible_leak_cols if c in df.columns]

    # Event meta kolonu; ister feature yaparız ister yapmayız, şimdilik kullanmayalım.
    event_meta_cols = []
    for c in ["event_type", "signal_wave", "signal_wave_label", "tp_pips", "sl_pips", "entry_price"]:
        if c in df.columns:
            event_meta_cols.append(c)

    drop_from_features = set(meta_cols + leak_cols + event_meta_cols)

    # --- Feature kolonlarını otomatik seç (numeric + NaN olmayan yoğun kolonlar) ---
    candidate_features = []
    for c in df.columns:
        if c in drop_from_features:
            continue
        if df[c].dtype == "O":
            # object / string’leri şimdilik almıyoruz (v3’de target encoding vs yaparız)
            continue
        candidate_features.append(c)

    # Çok fazla kolon olabilir; ama problem değil, XGB bunları yer.
    logger.info("   ✅ Seçilen feature kolon sayısı: %d", len(candidate_features))

    X = df[candidate_features].copy()

    # Basit missing value imputation
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    label_encoders = {"future_dir_label": le, "class_names": list(le.classes_)}

    logger.info("   ✅ Label encoder classes: %s", le.classes_)
    return X, y, candidate_features, label_encoders


def train_model_v2(X: pd.DataFrame, y: np.ndarray, label_encoders: dict):
    """
    XGBClassifier v2 – DOWN sınıfını özellikle ağırlıklandırıyoruz.
    """
    # Time-base split de yapabiliriz ama basit train_test_split ile başlayalım
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Class weight – DOWN'u boost'la
    classes = label_encoders["class_names"]
    # Örn: CHOP:1, UP:1, DOWN:2.5
    base_weights = {cls: 1.0 for cls in classes}
    if "DOWN" in base_weights:
        base_weights["DOWN"] = 2.5

    logger.info("   ✅ Class weights (v2): %s", base_weights)

    # sample_weight vektörü
    inv_map = {cls: idx for idx, cls in enumerate(classes)}
    y_train_labels = np.array([classes[c] for c in y_train])
    sample_weight = np.array([base_weights[label] for label in y_train_labels])

    # Model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("🧠 Model eğitimi başlıyor (v2)...")
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    # Validation raporu
    y_pred = model.predict(X_valid_scaled)
    logger.info("📊 CONFUSION MATRIX (v2):\n%s", confusion_matrix(y_valid, y_pred))
    logger.info("📊 CLASSIFICATION REPORT (v2):\n%s", classification_report(y_valid, y_pred, target_names=classes))

    return model, scaler


def save_artifacts(model, scaler, feature_list, label_encoders):
    Path("./models").mkdir(exist_ok=True, parents=True)

    import joblib

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_list, f)

    logger.info("💾 Model kaydedildi: %s", MODEL_PATH)
    logger.info("💾 Scaler kaydedildi: %s", SCALER_PATH)
    logger.info("💾 Feature list kaydedildi: %s", FEATURE_LIST_PATH)
    logger.info("💾 Label encoders kaydedildi: %s", LABEL_ENCODERS_PATH)


def main():
    events, master = load_data()
    merged = merge_events_with_features(events, master)
    X, y, feature_list, label_encoders = prepare_features_and_labels(merged)
    model, scaler = train_model_v2(X, y, label_encoders)
    save_artifacts(model, scaler, feature_list, label_encoders)

    logger.info("=" * 78)
    logger.info("✅ EVENT OUTCOME MODEL v2 TRAINING TAMAMLANDI")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()