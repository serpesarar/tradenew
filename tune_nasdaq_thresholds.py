import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =============== AYARLAR ===============
DATA_FILE = "nasdaq_training_dataset_v2.parquet"
MODEL_PATH = "models/nasdaq_meta_optuna_cv_v2.pkl"
TARGET_COL = "target_composite"
TEST_FRACTION = 0.2  # son %20 test

os.chdir("/Users/melihcanodacioglu/Desktop/nasdaq")
print("📂 Çalışma klasörü:", os.getcwd())

# =============== 1) VERİYİ YÜKLE ===============
print("📥 Veri yükleniyor:", DATA_FILE)
df = pd.read_parquet(DATA_FILE)
print("✅ Veri shape:", df.shape)

if TARGET_COL not in df.columns:
    raise SystemExit(f"❌ Target kolon '{TARGET_COL}' yok. Kolonlar: {list(df.columns)[:30]}")

# Zaman sıralı olsun
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

y_raw = df[TARGET_COL]
mask_valid = ~y_raw.isna()
df = df[mask_valid].reset_index(drop=True)
y_raw = y_raw[mask_valid].reset_index(drop=True)

y = y_raw.astype(int).to_numpy()
print("\n🎯 Target dağılımı:")
print(pd.Series(y).value_counts())

# =============== 2) FEATURE MATRİSİ ===============
drop_cols = [TARGET_COL]
for c in ["datetime", "date", "symbol", "regime", "regime_detailed"]:
    if c in df.columns:
        drop_cols.append(c)

X = df.drop(columns=drop_cols)

# Sadece sayısal kolonlar
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[num_cols].copy()
print(f"\n🔧 Feature sayısı (ham): {X.shape[1]}")

# Temizlik
X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill()
X = X.fillna(0.0)

# =============== 3) TRAIN / TEST BÖL ===============
n = len(X)
test_size = int(n * TEST_FRACTION)
train_size = n - test_size

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# =============== 4) MODELİ YÜKLE ===============
print("\n📦 Model yükleniyor:", MODEL_PATH)
obj = joblib.load(MODEL_PATH)

if not isinstance(obj, dict) or "models" not in obj or "features" not in obj:
    raise SystemExit("❌ Beklenen model formatı dict{'models','features',...} değil.")

models = obj["models"]
feat_names = obj["features"]

print(f"🧠 Ensemble model sayısı: {len(models)}")
print(f"🧩 Modelin beklediği feature sayısı: {len(feat_names)}")

# Feature hizalama
missing = [f for f in feat_names if f not in X_test.columns]
extra   = [c for c in X_test.columns if c not in feat_names]

if missing:
    print("\n⚠️ CSV'de olmayan ama modelin beklediği kolonlar (örnek ilk 20):")
    for m in missing[:20]:
        print(" -", m)
    print("Bu durumda model tam olarak test edilemez.")
    raise SystemExit(1)

X_test_aligned = X_test[feat_names].copy()
print("✅ Feature hizalaması OK.")

# =============== 5) ENSEMBLE PROBABILITY ===============
print("\n🔮 Ensemble ile olasılık hesaplanıyor...")

proba_list = []
for i, m in enumerate(models, start=1):
    p = m.predict_proba(X_test_aligned)
    proba_list.append(p)
    print(f"  Model {i} için proba shape: {p.shape}")

proba_mean = np.mean(proba_list, axis=0)  # (n_samples, 2)

classes_model = models[0].classes_
# 1 sınıfının index'ini bul
idx1 = int(np.where(classes_model == 1)[0][0])
p1 = proba_mean[:, idx1]  # "güçlü bar" olasılığı

print("\nÖrnek ilk 5 p(1):", p1[:5])

# =============== 6) FARKLI THRESHOLD ANALİZİ ===============
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

print("\n================ THRESHOLD TABLOSU (sinyal = p(1) >= t) ================")
print(f"{'thr':>5} {'cov%':>7} {'#sig':>7} {'sig_win%':>10} {'recall%':>9} {'acc%':>8}")

for t in thresholds:
    pred = (p1 >= t).astype(int)

    acc = accuracy_score(y_test, pred)
    n_signals = int(pred.sum())
    coverage = n_signals / len(y_test)

    # Sinyal isabet oranı = precision (model 1 dediğinde, gerçekten 1 olma oranı)
    if n_signals > 0:
        sig_win = precision_score(y_test, pred, pos_label=1)
    else:
        sig_win = np.nan

    rec = recall_score(y_test, pred, pos_label=1)

    print(
        f"{t:5.2f} "
        f"{coverage*100:7.2f} "
        f"{n_signals:7d} "
        f"{(sig_win*100 if not np.isnan(sig_win) else 0):10.2f} "
        f"{rec*100:9.2f} "
        f"{acc*100:8.2f}"
    )

print("\nAçıklamalar:")
print("  - thr      : eşik (p(1) >= thr ise sinyal veriyoruz)")
print("  - cov%     : testte sinyal çıkan bar oranı")
print("  - #sig     : sinyal sayısı")
print("  - sig_win% : sinyal isabet oranı (precision, model 1 dediğinde gerçekten 1 olma oranı)")
print("  - recall%  : gerçek 1'lerin ne kadarını yakalıyor")
print("  - acc%     : toplam doğruluk")