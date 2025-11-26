import yfinance as yf
import pandas as pd

# NASDAQ 100 Index (^NDX)
# 2010-2024 arası tüm veriler

# GÜNLÜK VERİ
df_daily = yf.download('^NDX', start='2010-01-01', end='2024-12-31', interval='1d')
# MultiIndex sütunlarını düzleştir ve yeniden adlandır
if df_daily.columns.nlevels > 1:
    df_daily.columns = df_daily.columns.droplevel(0)
df_daily.reset_index(inplace=True)
df_daily.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df_daily.to_csv('nasdaq100_2010_2024_daily.csv', index=False)

# HAFTALIK VERİ
df_weekly = yf.download('^NDX', start='2010-01-01', end='2024-12-31', interval='1wk')
if df_weekly.columns.nlevels > 1:
    df_weekly.columns = df_weekly.columns.droplevel(0)
df_weekly.reset_index(inplace=True)
df_weekly.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df_weekly.to_csv('nasdaq100_2010_2024_weekly.csv', index=False)

# AYLIK VERİ
df_monthly = yf.download('^NDX', start='2010-01-01', end='2024-12-31', interval='1mo')
if df_monthly.columns.nlevels > 1:
    df_monthly.columns = df_monthly.columns.droplevel(0)
df_monthly.reset_index(inplace=True)
df_monthly.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df_monthly.to_csv('nasdaq100_2010_2024_monthly.csv', index=False)

print(f"Günlük veri: {len(df_daily)} satır")
print(df_daily.head())

