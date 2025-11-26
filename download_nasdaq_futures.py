import yfinance as yf
import pandas as pd

# NASDAQ 100 E-mini Futures
ticker = "NQ=F"  # Continuous futures contract

df = yf.download(ticker, start='2010-01-01', interval='1d')

# MultiIndex sütunlarını düzleştir ve yeniden adlandır
if df.columns.nlevels > 1:
    df.columns = df.columns.droplevel(0)
df.reset_index(inplace=True)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df.to_csv('nasdaq100_futures_2010_2024.csv', index=False)

print(df.head())
print(f"\nToplam satır: {len(df)}")
print(f"Tarih aralığı: {df['Date'].min()} - {df['Date'].max()}")
print(f"\nVolume örneği: {df['Volume'].iloc[0]}")

