#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
from xgboost import XGBClassifier

# ================== CONFIG ==================
EVENT_PATH      = "./staging/nasdaq_event_outcomes_v1.parquet"

MODEL_PATH      = "./models/nasdaq_event_outcome_xgb_v1.pkl"
SCALER_PATH     = "./models/nasdaq_event_outcome_scaler_v1.pkl"
FEATS_PATH      = "./models/nasdaq_event_outcome_features_v1.pkl"
ENCODERS_PATH   = "./models/nasdaq_event_outcome_encoders_v1.pkl"
REPORT_PATH     = "./models/event_outcome_training_report_v1.txt"

TARGET_COL      = "future_dir"   # 0=CHOP, 1=UP, 2=DOWN

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./models/event_outcome_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ============================================


def load_events(path: str) -> pd.DataFrame:
    logger.info("📥 Event outcomes yükleniyor: %s", path)
    df = pd.read_parquet(path)
    logger.info("✅ Shape: %s", (df.shape,))
    return df


def build_event_rows(df: pd.DataFrame, price_col: str, event_cols):
    """
    df içinden event_cols==1 olan satırlardan event tablosu çıkar.

    Her event için:
      - O timestamp'teki TÜM feature'ları (merged satırı)
      - + event_type
      - + entry_price
    alır.

    NOT: Bu dataset büyük olacak (yüzbinlerce satır x binlerce feature),
    ama modelin gerçekten akıllanması için buna ihtiyacımız var.
    """
    logger.info("\n📌 Event satırları oluşturuluyor (full feature snapshot)...")
    rows = []

    for ev_col in event_cols:
        ev_mask = df[ev_col].fillna(0).astype(int) == 1
        idxs = np.where(ev_mask.values)[0]
        logger.info(f"   • {ev_col}: {len(idxs)} event")

        for idx in idxs:
            # O satırdaki TÜM kolonları al
            base_row = df.iloc[idx].to_dict()

            # Event bilgisini ekle (hangi trigger'dan geldiği)
            base_row["event_type"] = ev_col

            # Entry price'ı explicit kolon olarak ekle
            if price_col in df.columns:
                base_row["entry_price"] = df.iloc[idx][price_col]

            rows.append(base_row)

    events = pd.DataFrame(rows)
    logger.info(f"   ✅ Toplam event satırı: {len(events)}")
    return events


def prepare_dataset(df: pd.DataFrame):
    logger.info("\n🔧 Dataset hazırlanıyor...")

    df = df.copy()

    # Timestamp varsa zaman sırasına göre sırala
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Hedef kolon
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL ('{TARGET_COL}') dataset içinde yok!")

    # Label NaN olanları düş
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    logger.info("   Label NaN drop: %d -> %d", before, after)

    y = df[TARGET_COL].astype(int)

    # Leak olabilecek kolonlar (direkt future outcome)
    leak_like_cols = [
        TARGET_COL,
        "tp_sl_result",
        "max_up_move_pips",
        "max_down_move_pips",
    ]

    # ID / zaman kolonları
    drop_cols = ["timestamp"]

    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # Sadece zaman + doğrudan future outcome kolonlarını drop et
    all_drop = list(set(drop_cols + leak_like_cols))
    logger.info("   Toplam drop kolon sayısı: %d", len(all_drop))
    logger.info("   → Drop kolon örnekleri: %s", [c for c in all_drop if c in df.columns][:20])
    
    # X: zaman + target + explicit future outcome dışındaki her şey
    X = df.drop(columns=[c for c in all_drop if c in df.columns])

    # String kolonları tespit et (özellikle event_type gibi)
    string_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    logger.info("   String kolonlar: %d", len(string_cols))
    if string_cols:
        logger.info("   → %s", string_cols[:20])
    logger.info("   Numeric kolonlar: %d", len(numeric_cols))

    # String kolonları LabelEncoder ile encode et
    label_encoders = {}
    X_enc = X.copy()

    for col in string_cols:
        le = LabelEncoder()
        X_enc[col] = X_enc[col].fillna("UNKNOWN").astype(str)
        X_enc[col] = le.fit_transform(X_enc[col])
        label_encoders[col] = le
        logger.info("   ✅ %s encode edildi (%d sınıf)", col, len(le.classes_))

    # ❗ KRİTİK SATIR: Artık sadece float/int değil, bool'ları da alıyoruz
    # Yani: object/string hariç her şey feature
    X_final = X_enc.select_dtypes(exclude=["object", "string"])
    feature_names = list(X_final.columns)

    logger.info("   ✅ Toplam feature sayısı: %d", len(feature_names))
    logger.info("   ✅ İlk 20 feature: %s", feature_names[:20])

    # Zaman sırasına göre split (60/20/20)
    n = len(X_final)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_train = X_final.iloc[:train_end].reset_index(drop=True)
    y_train = y.iloc[:train_end].reset_index(drop=True)

    X_val = X_final.iloc[train_end:val_end].reset_index(drop=True)
    y_val = y.iloc[train_end:val_end].reset_index(drop=True)

    X_test = X_final.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)

    logger.info("   Split sizes: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        scaler, feature_names, label_encoders
    )
def train_model(X_train, y_train, X_val, y_val) -> XGBClassifier:
    logger.info("\n🚂 Model eğitimi başlıyor...")

    # Class weights (dataset dengesiz olabilir)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    logger.info("⚖️ Class weights: %s", class_weight_dict)

    sample_weight_train = pd.Series(y_train).map(class_weight_dict).values

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    logger.info("✅ Best iteration: %s", getattr(model, "best_iteration", None))
    logger.info("✅ Best score: %s", getattr(model, "best_score", None))
    return model


def evaluate_and_report(model, X_val, y_val, X_test, y_test, feature_names):
    logger.info("\n📈 Değerlendirme...")

    # Validation
    val_proba = model.predict_proba(X_val)
    val_logloss = log_loss(y_val, val_proba)
    logger.info("   Val LogLoss: %.4f", val_logloss)

    # Test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    test_logloss = log_loss(y_test, y_proba)

    logger.info("\n📊 Test LogLoss: %.4f", test_logloss)
    logger.info("\n" + classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    logger.info("📊 Confusion Matrix:\n%s", cm)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\n🔝 Top 20 Features:\n%s", importance_df.head(20))

    # Rapor kaydet
    Path("./models").mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("NASDAQ EVENT OUTCOME MODEL REPORT v1\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Validation LogLoss: {val_logloss:.4f}\n")
        f.write(f"Test LogLoss: {test_logloss:.4f}\n\n")
        f.write("Top 30 Features:\n")
        f.write(importance_df.head(30).to_string(index=False))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    logger.info("💾 Rapor kaydedildi: %s", REPORT_PATH)


def save_artifacts(model, scaler, feature_names, label_encoders):
    logger.info("\n💾 Artefact'lar kaydediliyor...")
    Path("./models").mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATS_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)

    logger.info("✅ Model: %s", MODEL_PATH)
    logger.info("✅ Scaler: %s", SCALER_PATH)
    logger.info("✅ Features: %s", FEATS_PATH)
    logger.info("✅ Encoders: %s", ENCODERS_PATH)


def main():
    warnings.filterwarnings("ignore")

    df = load_events(EVENT_PATH)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler, feature_names, label_encoders
    ) = prepare_dataset(df)

    model = train_model(X_train, y_train, X_val, y_val)
    evaluate_and_report(model, X_val, y_val, X_test, y_test, feature_names)
    save_artifacts(model, scaler, feature_names, label_encoders)

    logger.info("\n" + "=" * 80)
    logger.info("✅ EVENT OUTCOME PIPELINE TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()