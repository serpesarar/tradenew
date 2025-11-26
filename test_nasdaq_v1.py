import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========= 1) YOLLAR =========
BASE_DIR = "/Users/melihcanodacioglu/Desktop/nasdaq"
MODEL_PATH = "nasdaq_30m_ULTIMATE_v1.pkl"
DATA_PATH = "nasdaq.csv"

os.chdir(BASE_DIR)
print(f"📂 Çalışma klasörü: {os.getcwd()}")

# ========= 2) MODELİ YÜKLE =========
print("📦 Model yükleniyor...")
obj = joblib.load(MODEL_PATH)

print("Tip:", type(obj))
if not isinstance(obj, dict):
    print("❌ Bu script v1 (dict) model için yazıldı, ama dosya dict değil.")
    raise SystemExit(1)

print("\n🔑 Anahtarlar:", list(obj.keys()))

trained_acc = obj.get("accuracy", None)
trained_sharpe = obj.get("sharpe", None)
trained_dd = obj.get("max_dd", None)

print("\n📊 Modelin kendi kaydettiği metrikler:")
print(f"  - accuracy: {trained_acc}")
print(f"  - sharpe  : {trained_sharpe}")
print(f"  - max_dd  : {trained_dd}")

features = obj.get("features", None)
scaler = obj.get("scaler", None)
models_dict = obj.get("models", None)
weights_dict = obj.get("weights", {})

if features is None or models_dict is None:
    print("❌ 'features' veya 'models' anahtarı yok, bu dosyadan ensemble test yapamam.")
    raise SystemExit(1)

print(f"\n🎯 Feature sayısı (eğitimde kullanılan): {len(features)}")
print("🧩 Baz modeller:", list(models_dict.keys()))

# ========= 3) VERİYİ YÜKLE =========
if not os.path.exists(DATA_PATH):
    print(f"❌ Veri dosyasını bulamadım: {DATA_PATH}")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)
print("\n✅ Veri yüklendi, shape:", df.shape)

# Target kolon
if "target" in df.columns:
    TARGET_COL = "target"
else:
    print("❌ 'target' kolonu yok. Kolonlardan bazıları:", list(df.columns)[:50])
    raise SystemExit(1)

y = df[TARGET_COL].to_numpy()
print("🎯 Target dağılımı:\n", pd.Series(y).value_counts())

# ========= 4) FEATURE HİZALAMA =========
missing_feats = [f for f in features if f not in df.columns]
if missing_feats:
    print("\n❌ Bazı feature'lar CSV'de yok, bu yüzden modeli birebir test edemiyoruz.")
    print("Eksik feature sayısı:", len(missing_feats))
    print("Örnek eksikler (ilk 20):")
    for f in missing_feats[:20]:
        print(" -", f)
    print("\n➡️ Çözüm: Bu modeli eğittiğin eski training scriptini kullanarak tekrar test yapmak.")
    raise SystemExit(1)

X = df[features].copy()

# Sayısal olmayan kolon var mı?
non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
if non_numeric:
    print("\n❌ Aşağıdaki feature'lar sayısal değil, model bunları eğitimde nasıl encode ettiğini bilmiyoruz:")
    for c in non_numeric:
        print(f" - {c} | dtype={X[c].dtype}")
    print("\nBu yüzden buradan çıkan sonuç güvenilir olmaz, burada duruyorum.")
    raise SystemExit(1)

X_arr = X.to_numpy(dtype=float)

# ========= 5) SCALER UYGULA =========
if scaler is not None:
    try:
        X_arr = scaler.transform(X_arr)
    except Exception as e:
        print("\n❌ scaler.transform sırasında hata oldu:")
        print(e)
        print("\nMuhtemel sebep: sklearn versiyon farkı veya feature sırası/şekli.")
        print("Bu yüzden, bu ortamda yeniden test etmek sağlıklı değil.")
        raise SystemExit(1)

# ========= 6) TIME-BASED TRAIN/TEST SPLIT =========
n = len(X_arr)
test_size = int(n * 0.2)  # son %20 test
if test_size == 0:
    print("❌ Veri çok az, test seti oluşturamıyorum.")
    raise SystemExit(1)

X_train, X_test = X_arr[:-test_size], X_arr[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

print(f"\n📊 Train shape: {X_train.shape}  | Test shape: {X_test.shape}")

# ========= 7) ENSEMBLE TAHMİN =========
model_names = list(models_dict.keys())
w = np.array([float(weights_dict.get(name, 1.0)) for name in model_names], dtype=float)
if w.sum() <= 0:
    w[:] = 1.0
w = w / w.sum()

all_preds = []
used_weights = []
used_model_names = []

for i, name in enumerate(model_names):
    m = models_dict[name]
    if not hasattr(m, "predict"):
        print(f"⚠️ Model '{name}' predict metoduna sahip değil, atlıyorum.")
        continue
    print(f"🔮 {name} modelinden tahmin alınıyor...")
    preds = m.predict(X_test)
    all_preds.append(preds)
    used_weights.append(w[i])
    used_model_names.append(name)

if not all_preds:
    print("❌ Hiç bir modelden tahmin alamadım, burada duruyorum.")
    raise SystemExit(1)

all_preds = np.vstack(all_preds)  # shape = (#models, n_samples)
used_weights = np.array(used_weights, dtype=float)
used_weights = used_weights / used_weights.sum()

print("\n🧮 Ensemble (ağırlıklı oy) hesaplanıyor...")

n_models, n_samples = all_preds.shape
final_pred = []

for j in range(n_samples):
    votes = {}
    for k in range(n_models):
        lbl = all_preds[k, j]
        votes[lbl] = votes.get(lbl, 0.0) + used_weights[k]
    best_lbl = max(votes.items(), key=lambda x: x[1])[0]
    final_pred.append(best_lbl)

y_pred = np.array(final_pred)

# ========= 8) METRİKLER =========
acc = accuracy_score(y_test, y_pred)
print("\n✅ TEST ACCURACY (bu ortamda yeniden hesaplanan): {:.4f} ({:.2f}%)".format(acc, acc * 100))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))