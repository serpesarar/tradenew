import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "nasdaq_training_dataset_v2.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "nasdaq_meta_lgb_v2.pkl")

print("📂 Çalışma klasörü:", BASE_DIR)
print("📥 Veri dosyası:", DATA_PATH)

# =========================
# 1) VERİYİ YÜKLE
# =========================
df = pd.read_parquet(DATA_PATH)
print("✅ Veri yüklendi, shape:", df.shape)

# =========================
# 2) TARGET KOLONUNU BUL
# =========================
candidate_targets = [
    "target_killer",
    "target_composite",
    "target_simple",
    "target_atr",
    "target_sharpe",
    "target",
    "y",
    "label",
]

TARGET_COL = None
for c in candidate_targets:
    if c in df.columns and df[c].nunique() >= 2:
        TARGET_COL = c
        break

if TARGET_COL is None:
    print("❌ Kullanılabilir target kolonu bulamadım.")
    print("Mevcut kolonlar:", df.columns.tolist())
    raise SystemExit(1)

print(f"🎯 Seçilen target kolon: {TARGET_COL}")
print("Target dağılımı:")
print(df[TARGET_COL].value_counts(dropna=False))

y_raw = df[TARGET_COL].copy()

# =========================
# 3) TARGET'I SAYISALA ÇEVİR
# =========================
# Eğer SHORT / NEUTRAL / LONG gibi string ise map’le
if y_raw.dtype == "object":
    uniques = sorted(y_raw.dropna().unique())
    print("🔤 String target sınıfları:", uniques)

    mapping_known = {
        "SHORT": -1,
        "NEUTRAL": 0,
        "LONG": 1,
        "SELL": -1,
        "BUY": 1,
        "FLAT": 0,
    }
    y = y_raw.map(mapping_known)

    # Hâlâ NaN kalan varsa, onları da kendimiz index bazlı mapleyelim
    if y.isna().any():
        remaining = y_raw[y.isna()].unique()
        print("⚠️ Map'lenemeyen sınıflar, sıralı index ile maplenecek:", remaining)
        extra_map = {cls: i for i, cls in enumerate(remaining)}
        y = y_raw.map({**mapping_known, **extra_map})
else:
    y = y_raw.astype(int)

print("🎯 Target numeric özet:")
print(y.value_counts())

# =========================
# 4) FEATURE MATRİSİNİ HAZIRLA
# =========================
drop_cols = set(candidate_targets + ["datetime", "date"])
drop_cols = [c for c in drop_cols if c in df.columns]

df_feat = df.drop(columns=drop_cols)

# Sadece sayısal kolonlar
num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
X = df_feat[num_cols].copy()

print("🔧 Feature sayısı:", X.shape[1])
print("Örnek feature kolonları:", num_cols[:20])

# Güvenlik amaçlı: inf / NaN temizliği
X = X.replace([np.inf, -np.inf], np.nan)
nan_before = X.isna().sum().sum()
if nan_before > 0:
    print(f"⚠️ Feature içinde {nan_before} NaN vardı, dolduruluyor (ffill → bfill → 0).")
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

# =========================
# 5) TRAIN / TEST AYIR
# =========================
n = len(df)
test_ratio = 0.2
test_size = int(n * test_ratio)
train_size = n - test_size

X_train = X.iloc[:train_size].to_numpy()
y_train = y.iloc[:train_size].to_numpy()
X_test  = X.iloc[train_size:].to_numpy()
y_test  = y.iloc[train_size:].to_numpy()

print(f"📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# =========================
# 6) MODELİ EĞİT
# =========================
classes = np.unique(y_train)
n_classes = len(classes)
print(f"🔢 Sınıf sayısı: {n_classes}, sınıflar: {classes}")

params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 200,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "n_jobs": -1,
}

print("⚙️ LightGBM parametreleri:")
for k, v in params.items():
    print(f"  {k}: {v}")

model = LGBMClassifier(**params)
print("🚀 Eğitim başlıyor...")
model.fit(X_train, y_train)

# =========================
# 7) TEST DEĞERLENDİRME
# =========================
print("\n🔮 Test setinde tahmin yapılıyor...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n✅ TEST ACCURACY: {:.4f} ({:.2f}%)".format(acc, acc * 100))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, labels=classes))

# =========================
# 8) MODELİ KAYDET
# =========================
bundle = {
    "model": model,
    "features": num_cols,
    "target_col": TARGET_COL,
    "classes": classes,
    "train_size": train_size,
    "test_size": test_size,
    "accuracy": float(acc),
}

joblib.dump(bundle, MODEL_PATH)
print("\n💾 Model kaydedildi:", MODEL_PATH)
