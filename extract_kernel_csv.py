#!/usr/bin/env python3
"""
Kaggle kernel'ındaki veriyi CSV olarak kaydetmek için script
"""
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

print("📊 NASDAQ verisi çekiliyor...")

# Veriyi çek (yfinance kullanarak)
ticker = yf.Ticker('^NDX')
df = ticker.history(start='2012-01-01', end='2021-10-21')

print(f"✅ Veri çekildi: {df.shape}")

# Orijinal veriyi kaydet
df.to_csv('nasdaq_kernel_original_data.csv')
print("💾 Orijinal veri 'nasdaq_kernel_original_data.csv' olarak kaydedildi")

# Close price'ı filtrele
data = df.filter(['Close'])

# Model tahminlerini yapmak için (kernel'daki gibi)
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Training data
train_data = scaled_data[0:training_data_len, :]

# Test data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60: i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Not: Model olmadan tahmin yapamayız, ama veriyi kaydedebiliriz
# Train ve validation verilerini kaydet
train = data[:training_data_len]
valid = data[training_data_len:]

# Train verisini kaydet
train.to_csv('nasdaq_kernel_train_data.csv')
print("💾 Train verisi 'nasdaq_kernel_train_data.csv' olarak kaydedildi")

# Validation verisini kaydet
valid.to_csv('nasdaq_kernel_valid_data.csv')
print("💾 Validation verisi 'nasdaq_kernel_valid_data.csv' olarak kaydedildi")

# Tüm veriyi birleştirip kaydet
full_data = pd.DataFrame({
    'Date': df.index,
    'Open': df['Open'],
    'High': df['High'],
    'Low': df['Low'],
    'Close': df['Close'],
    'Volume': df['Volume']
})

# Adj Close varsa ekle
if 'Dividends' in df.columns:
    full_data['Dividends'] = df['Dividends']
if 'Stock Splits' in df.columns:
    full_data['Stock Splits'] = df['Stock Splits']

full_data.to_csv('nasdaq_kernel_full_data.csv', index=False)
print("💾 Tam veri 'nasdaq_kernel_full_data.csv' olarak kaydedildi")

print("\n✅ Tüm CSV dosyaları kaydedildi!")

