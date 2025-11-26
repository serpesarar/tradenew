import pandas as pd
import joblib

# Load
df = pd.read_parquet("nasdaq_training_dataset_v2.parquet")
model = joblib.load("models/nasdaq_meta_optuna_cv_v2.pkl")

# Get feature names
model_features = set(model['features'])
data_columns = set(df.columns)

# Check each feature one by one
print("="*80)
print("🔍 DETAYLI FEATURE KARŞILAŞTIRMASI")
print("="*80)

print(f"\nModel top feature: {sorted(list(model_features))[:10]}")
print(f"Data top columns: {sorted(list(data_columns))[:10]}")

print("\n" + "="*80)
print("❌ GERÇEK EKSİK FEATURE'LAR:")
print("="*80)
for i, f in enumerate(sorted(model_features - data_columns), 1):
    print(f"{i:4d}. {f}")

print("\n" + "="*80)
print("⚠️  VERİDE FAZLA OLAN KOLONLAR:")
print("="*80)
for i, f in enumerate(sorted(data_columns - model_features), 1):
    print(f"{i:4d}. {f}")

print(f"\nÖzet: {len(model_features - data_columns)} eksik, {len(data_columns - model_features)} fazla")