import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 30 dakikalık NASDAQ-100 verisi
url = "https://storage.googleapis.com/kagglesdsdata/datasets/7874928/13223215/30m_data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251120%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251120T213702Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=47a4b9137d49e40b15baf7d442e55d30200856754f07534e79a44b36110f3860fc80b2bb96c9b6ba1393b71c366f2bb455db7afb77cdcea57a36c97533b180bf59a0117f94055183b42fabd392b9923e9a64942e183d6dd7c9868dea9eb97bf3cfda5109291c08a608c6236ddf24403d5ec96fd3684dbe849ca66a00cc998e6e754792bbccceae0626f9bd7b587f2553443fd7dce82c72aaa57e9560b67b605feb43ca13a40ee2b8b038af06c0f1716c717a259302fd4537157041ed9bcbb0aacfc05754372918c8aac9233b42e28e2f785662b1d81e713f506686dcfc86db6e3aa48e7ca3d57f76644ecb4a4fed912aa5683fdbcf4002869db92580bfa5f22b"

print("📊 NASDAQ-100 30 dakikalık veri indiriliyor...")

# Veriyi oku (tab-separated)
df = pd.read_csv(url, sep='\t')

# Temel bilgileri göster
print(f"\n✅ Veri başarıyla indirildi!")
print(f"📈 Toplam satır: {len(df):,}")
print(f"📋 Sütunlar: {df.columns.tolist()}")

print(f"\n🔍 Veri Önizleme:")
print(df.head(10))

print(f"\n📊 Veri Tipleri:")
print(df.dtypes)

print(f"\n📉 İstatistikler:")
print(df.describe())

# Tarih sütununu bul ve analiz et
date_columns = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]

if date_columns:
    date_col = date_columns[0]
    print(f"\n📅 Tarih sütunu: '{date_col}'")
    print(f"İlk tarih: {df[date_col].iloc[0]}")
    print(f"Son tarih: {df[date_col].iloc[-1]}")
    
    # Datetime'a çevir (format: YYYY.MM.DD HH:MM:SS)
    df[date_col] = pd.to_datetime(df[date_col], format='%Y.%m.%d %H:%M:%S', errors='coerce')
    
    # Tarih aralığını hesapla
    date_range = (df[date_col].max() - df[date_col].min()).days
    print(f"Toplam gün: {date_range:,} gün ({date_range/365:.1f} yıl)")

# Kaydet
df.to_csv('nasdaq100_30min_full.csv', index=False)
print(f"\n💾 'nasdaq100_30min_full.csv' olarak kaydedildi!")

# Eksik veri kontrolü
print(f"\n⚠️ Eksik Veri Analizi:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✅ Eksik veri yok!")

