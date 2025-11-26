import os
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib

import lightgbm as lgb
import optuna

# ================== AYARLAR ==================
DATA_FILE = "nasdaq_training_dataset_v2.parquet"
TARGET_COL = "target_composite"   # şu an bunu kullandık
TEST_FRACTION = 0.2               # son %20 test
N_SPLITS = 5                      # TimeSeries CV
N_TRIALS = int(os.getenv("N_TRIALS", "30"))  # Optuna deneme sayısı

RANDOM_STATE = 42

print("📂 Çalışma klasörü:", os.getcwd())
print("📥 Veri yükleniyor:", DATA_FILE)

# ================== 1) VERİYİ YÜKLE ==================
df = pd.read_parquet(DATA_FILE)
print("✅ Veri shape:", df.shape)

if TARGET_COL not in df.columns:
    raise SystemExit(
        f"❌ Target kolon '{TARGET_COL}' bulunamadı. "
        f"İlk 50 kolon: {list(df.columns)[:50]}"
    )

# Zaman sıralı olsun
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Target
y_raw = df[TARGET_COL]
print("\n🎯 Target dağılımı (ham):")
print(y_raw.value_counts(dropna=False))

# NaN targetları at
mask_valid = ~y_raw.isna()
df = df[mask_valid].reset_index(drop=True)
y_raw = y_raw[mask_valid].reset_index(drop=True)

# Binary hale getir (0/1)
y = y_raw.astype(int).to_numpy()
classes = np.unique(y)
print("\n🎯 Target sınıfları:", classes)

# ================== 2) FEATURE MATRİSİ ==================
# Cevap sızıntısını temizle: target/future içeren her şeyi at
drop_cols = set()
drop_cols.add(TARGET_COL)

# Zaman / kimlik kolonları
for c in ["datetime", "date", "time", "symbol", "regime", "regime_detailed"]:
    if c in df.columns:
        drop_cols.add(c)

# Tüm 'target' ve 'future' içeren kolonları at
for c in df.columns:
    lc = c.lower()
    if "target" in lc or "future" in lc:
        drop_cols.add(c)
    # eski merge'lerden gelen *_y etiketleri varsa onları da at
    if c.endswith("_y"):
        drop_cols.add(c)

print("\n❌ Drop edilen kolonlar:")
for c in sorted(drop_cols):
    print("  -", c)

# X full
X_full = df.drop(columns=sorted(drop_cols))

# Sadece sayısal kolonlar
num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
X = X_full[num_cols].copy()

print(f"\n🔧 Feature sayısı: {X.shape[1]}")
print("Örnek feature kolonları:", num_cols[:20])

# Sonsuzları ve NaN'leri temizle
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(method="ffill").fillna(method="bfill")
X = X.fillna(0.0)

# ================== 3) TRAIN / TEST ZAMAN BÖLÜNMESİ ==================
n = len(X)
test_size = int(n * TEST_FRACTION)
if test_size < 1000:
    test_size = max(1000, int(n * 0.1))  # Çok küçük olmasın

train_size = n - test_size

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")
print("🔢 Train sınıfları:", np.unique(y_train, return_counts=True))

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# ================== 4) OPTUNA HEDEF FONKSİYONU ==================
def objective(trial: optuna.Trial) -> float:
    # Binary problem olduğu için 'binary' objective kullanıyoruz
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 512),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 400),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 700),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    acc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)

    mean_acc = float(np.mean(acc_scores))
    print(f"Trial {trial.number}: CV ACC={mean_acc:.4f}")
    return mean_acc
# ================== 5) OPTUNA ÇALIŞTIR ==================
print("\n══════════════════════════════════════════════════════════")
print("  🔧 OPTUNA ARAMASI BAŞLIYOR  (LightGBM NASDAQ v2)")
print("══════════════════════════════════════════════════════════")

study = optuna.create_study(
    direction="maximize",
    study_name="nasdaq_meta_v2_optuna",
)

study.optimize(objective, n_trials=N_TRIALS)

print("\n✅ Optuna bitti.")
print("🎯 En iyi skor (CV ACC):", study.best_value)
print("📦 En iyi parametrerler:")
for k, v in study.best_params.items():
    print(f"  - {k}: {v}")

# Final param set (LightGBM'in istediği sabitler ekleniyor)
best_params = study.best_params.copy()
best_params.update(
    {
        "objective": "multiclass",
        "num_class": 2,
        "metric": "multi_logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
)

# ================== 6) CV ENSEMBLE (5 ADET MODEL) ==================
print("\n══════════════════════════════════════════════════════════")
print("  🤖 CV ENSEMBLE EĞİTİMİ (5 LightGBM modeli)")
print("══════════════════════════════════════════════════════════")

models = []
fold_accs = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accs.append(acc)
    models.append(model)

    print(f"  Fold {fold} ACC: {acc:.4f}")

print("  Ortalama CV ACC (ensemble base): {:.4f}".format(float(np.mean(fold_accs))))

# ================== 7) TEST SET ÜZERİNDE ENSEMBLE PERFORMANSI ==================
print("\n🔮 Test setinde tahmin yapılıyor (ensemble ortalaması)...")

# Her modelden probability al, ortalamasını al
proba_list = [m.predict_proba(X_test) for m in models]
proba_ens = np.mean(proba_list, axis=0)
y_pred = np.argmax(proba_ens, axis=1)

test_acc = accuracy_score(y_test, y_pred)

print("\n✅ TEST ACCURACY: {:.4f} ({:.2f}%)".format(test_acc, test_acc * 100))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, labels=classes))

# ================== 8) MODELİ KAYDET ==================
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "nasdaq_meta_optuna_cv_v2.pkl")

artifact = {
    "models": models,
    "features": num_cols,          # hangi kolonlarla eğittik
    "best_params": best_params,
    "cv_accuracy": float(np.mean(fold_accs)),
    "test_accuracy": float(test_acc),
    "classes": classes,
    "timeframe": "30m",
    "version": "v2_optuna_cv",
    "trained_date": dt.datetime.now().isoformat(),
}

joblib.dump(artifact, model_path)

print("\n💾 Model kaydedildi:", model_path)
print("🎉 NASDAQ v2 Optuna + CV Ensemble eğitim tamamlandı!")