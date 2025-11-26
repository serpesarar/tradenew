#!/usr/bin/env python3
"""
NASDAQ WAVE DIRECTION MODEL

Girdi:
    - ./staging/nasdaq_full_wave_strength.parquet

Çıktı:
    - ./models/nasdaq_wave_dir_xgb.pkl
    - ./models/nasdaq_wave_features.pkl

Not:
    - future_* ve fut_pips_* gibi kolonlar X'e alınmıyor (data leak önlemi)
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

MODELS_DIR = "./models"
DATA_PATH = "./staging/nasdaq_full_wave_strength.parquet"


def proper_time_series_split(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print(f"   📏 Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("=" * 80)
    print("🚀 NASDAQ WAVE DIRECTION MODEL TRAINING")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {DATA_PATH}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"📥 Veri yükleniyor: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"   ✅ Shape: {df.shape}")

    # target
    if "signal_wave" not in df.columns:
        raise ValueError("signal_wave kolonu bulunamadı.")

    # future / leak kolonlarını X'ten çıkar
    drop_cols = [c for c in df.columns
                 if c.startswith("future_")
                 or c.startswith("fut_pips_")
                 or c in ["wave_strength_pips", "wave_duration_bars"]]

    base_cols = ["timestamp", "signal_wave"]

    X = df.drop(columns=drop_cols + base_cols, errors="ignore")
    y = df["signal_wave"]

    print(f"   ✅ Feature sayısı (X): {X.shape[1]}")

    # time-series split
    X_train, X_val, X_test, y_train, y_val, y_test = proper_time_series_split(X, y)

    # scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        random_state=42
    )

    print("\n🏋️ Eğitim başlıyor (VAL set ile early stopping)...")
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        
        verbose=50
    )

    print("\n📈 TEST SET SONUÇLARI (signal_wave):")
    y_pred = model.predict(X_test_s)
    print(classification_report(
        y_test,
        y_pred,
        target_names=["CHOP", "LONG_WAVE", "SHORT_WAVE"]
    ))

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(20)
    print("\n🎯 En önemli 20 feature:")
    print(fi)

    # Kaydet
    import joblib
    model_path = os.path.join(MODELS_DIR, "nasdaq_wave_dir_xgb.pkl")
    feat_path = os.path.join(MODELS_DIR, "nasdaq_wave_features.pkl")
    scaler_path = os.path.join(MODELS_DIR, "nasdaq_wave_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), feat_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Model kaydedildi: {model_path}")
    print(f"💾 Feature list kaydedildi: {feat_path}")
    print(f"💾 Scaler kaydedildi: {scaler_path}")

    print("=" * 80)
    print("✅ NASDAQ wave direction modeli hazır.")
    print("=" * 80)


if __name__ == "__main__":
    main()