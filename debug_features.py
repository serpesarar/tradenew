import pandas as pd
import joblib
import os

# Model ve veri yolları
MODEL_PATH = "models/nasdaq_meta_optuna_cv_v2.pkl"
DATA_PATH = "nasdaq_training_dataset_v2.parquet"

print("="*60)
print("🔍 FEATURE UYUŞMAZLIĞI ANALİZİ")
print("="*60)

# Veriyi yükle
if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Veri yüklendi. Kolon sayısı: {len(df.columns)}")
    print(f"Veri kolonları (ilk 30):\n{sorted(df.columns.tolist())[:30]}")
else:
    print(f"❌ Veri dosyası bulunamadı: {DATA_PATH}")
    exit()

# Modeli yükle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"\n✅ Model yüklendi. Anahtarlar: {list(model.keys())}")
    
    features = model.get('features', [])
    print(f"Model feature'ları (ilk 30):\n{sorted(features)[:30]}")
else:
    print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
    exit()

# Karşılaştırma
missing_in_data = set(features) - set(df.columns)
missing_in_model = set(df.columns) - set(features)

print("\n" + "="*60)
print("❌ EKSİK FEATURE'LAR (Modelde var, veride yok):")
print("="*60)
for f in sorted(missing_in_data):
    print(f"  - {f}")

print("\n" + "="*60)
print("⚠️  FAZLA FEATURE'LAR (Veride var, modelde yok):")
print("="*60)
for f in sorted(missing_in_model):
    print(f"  - {f}")

print("\n" + "="*60)
print(f"Özet: {len(missing_in_data)} eksik, {len(missing_in_model)} fazla feature var.")
print("="*60)