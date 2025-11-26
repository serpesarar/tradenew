#!/usr/bin/env python3
"""
NQ_5Years_8_11_2024.csv  (5 dakikalık OHLCV)
   → 30m / 60m / 240m OHLCV
   → M30 + H1 + H4 tek tabloda merge
   → Her TF için temel indikatörler (RSI, MACD, EMA, SMA, Bollinger, ATR, volume features)
   → Çıktı: nasdaq_full_from5m_clean.parquet

Bu dosya tamamen 5m datasından üretilmiş temiz bir master dataset olacak.
Eski 51k'lık nasdaq_full_clean.parquet'ten bağımsız çalışıyor.
"""

import pandas as pd
import numpy as np

FIVE_MIN_PATH = "./NQ_5Years_8_11_2024.csv"   # Gerekirse ismi değiştir
OUT_PATH      = "./nasdaq_full_from5m_clean.parquet"


# ---------- 1) Yardımcı fonksiyonlar ----------

def resample_ohlcv(df, rule):
    """5m OHLCV'den istenen periyotlara (30min, 60min, 240min) resample."""
    agg = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    agg = agg.dropna(subset=["Open", "High", "Low", "Close"])
    return agg


def add_basic_indicators(df, tf_tag):
    """
    Tek bir timeframe için (M30 / H1 / H4) temel indikatörleri ekler.
    Kolon isimleri: Open_M30, High_M30, Close_M30, Volume_M30 gibi olmalı.
    tf_tag: 'M30', 'H1', 'H4'
    """
    o = df[f"Open_{tf_tag}"]
    h = df[f"High_{tf_tag}"]
    l = df[f"Low_{tf_tag}"]
    c = df[f"Close_{tf_tag}"]
    v = df[f"Volume_{tf_tag}"]

    out = df.copy()

    # --- RSI ---
    def rsi(series, window):
        diff = series.diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    out[f"rsi_14_{tf_tag}"] = rsi(c, 14)
    out[f"rsi_7_{tf_tag}"]  = rsi(c, 7)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal

    out[f"macd_line_{tf_tag}"]      = macd_line
    out[f"macd_signal_{tf_tag}"]    = signal
    out[f"macd_hist_{tf_tag}"]      = hist
    out[f"macd_hist_diff_{tf_tag}"] = hist.diff()

    # --- EMA / SMA ---
    out[f"ema_20_{tf_tag}"]  = c.ewm(span=20, adjust=False).mean()
    out[f"ema_50_{tf_tag}"]  = c.ewm(span=50, adjust=False).mean()
    out[f"ema_200_{tf_tag}"] = c.ewm(span=200, adjust=False).mean()

    out[f"sma_20_{tf_tag}"]  = c.rolling(window=20).mean()
    out[f"sma_50_{tf_tag}"]  = c.rolling(window=50).mean()
    out[f"sma_200_{tf_tag}"] = c.rolling(window=200).mean()

    # --- Bollinger (20) ---
    ma20  = c.rolling(window=20).mean()
    std20 = c.rolling(window=20).std()

    out[f"boll_upper_{tf_tag}"]  = ma20 + 2 * std20
    out[f"boll_lower_{tf_tag}"]  = ma20 - 2 * std20
    out[f"boll_middle_{tf_tag}"] = ma20
    out[f"boll_width_{tf_tag}"]  = (out[f"boll_upper_{tf_tag}"] - out[f"boll_lower_{tf_tag}"]) / c
    out[f"boll_zscore_{tf_tag}"] = (c - ma20) / std20.replace(0, np.nan)

    # --- ATR(14) ---
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()

    out[f"atr_14_{tf_tag}"] = atr14
    out[f"atr_pct_{tf_tag}"] = atr14 / c

    # --- OBV / Volume ---
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    out[f"obv_{tf_tag}"] = obv

    vol_ma20 = v.rolling(window=20).mean()
    vol_std20 = v.rolling(window=20).std()

    out[f"volume_ma_{tf_tag}"]      = vol_ma20
    out[f"volume_ratio_{tf_tag}"]   = v / vol_ma20
    out[f"volume_zscore_{tf_tag}"]  = (v - vol_ma20) / vol_std20.replace(0, np.nan)
    out[f"volume_roc_{tf_tag}"]     = v.pct_change()
    out[f"volume_oscillator_{tf_tag}"] = (
        v.ewm(span=12, adjust=False).mean() - v.ewm(span=26, adjust=False).mean()
    )

    # --- Volatilite ---
    out[f"volatility_{tf_tag}"] = c.pct_change().rolling(window=20).std()

    return out


# ---------- 2) 5m datasını oku ve resample et ----------

print("=" * 80)
print("🚀 NASDAQ 5m → M30/H1/H4 MASTER DATASET")
print("=" * 80)
print(f"📥 5m data okunuyor: {FIVE_MIN_PATH}")

df5 = pd.read_csv(FIVE_MIN_PATH)

# Tip isimleri: Time, Open, High, Low, Close, Volume
time_col = "Time"
df5[time_col] = pd.to_datetime(df5[time_col])
df5 = df5.sort_values(time_col).set_index(time_col)

print(f"   ✅ 5m shape: {df5.shape}")

# Resample
m30 = resample_ohlcv(df5, "30min")
h1  = resample_ohlcv(df5, "60min")
h4  = resample_ohlcv(df5, "240min")

print(f"   ✅ M30 shape: {m30.shape}")
print(f"   ✅ H1  shape: {h1.shape}")
print(f"   ✅ H4  shape: {h4.shape}")
print(f"   🕒 M30 range: {m30.index.min()} → {m30.index.max()}")

# ---------- 3) Kolon isimlerini M30/H1/H4 yap ----------

m30_df = m30.rename(columns={
    "Open": "Open_M30",
    "High": "High_M30",
    "Low": "Low_M30",
    "Close": "Close_M30",
    "Volume": "Volume_M30",
})
h1_df = h1.rename(columns={
    "Open": "Open_H1",
    "High": "High_H1",
    "Low": "Low_H1",
    "Close": "Close_H1",
    "Volume": "Volume_H1",
})
h4_df = h4.rename(columns={
    "Open": "Open_H4",
    "High": "High_H4",
    "Low": "Low_H4",
    "Close": "Close_H4",
    "Volume": "Volume_H4",
})

# Index → timestamp
m30_df["timestamp"] = m30_df.index
h1_df["timestamp"]  = h1_df.index
h4_df["timestamp"]  = h4_df.index

# ---------- 4) M30 merkezli merge_asof: M30 + H1 + H4 ----------

base = m30_df.sort_values("timestamp").reset_index(drop=True)
h1_df = h1_df.sort_values("timestamp").reset_index(drop=True)
h4_df = h4_df.sort_values("timestamp").reset_index(drop=True)

print("\n🔗 M30 + H1 merge_asof...")
merged = pd.merge_asof(
    base,
    h1_df,
    on="timestamp",
    direction="backward",
)

print(f"   ✅ M30+H1 shape: {merged.shape}")

print("\n🔗 (M30+H1) + H4 merge_asof...")
merged = pd.merge_asof(
    merged.sort_values("timestamp"),
    h4_df.sort_values("timestamp"),
    on="timestamp",
    direction="backward",
)

print(f"   ✅ Final merged shape (raw): {merged.shape}")

merged = merged.set_index("timestamp")

# ---------- 5) Her timeframe için indikatörleri ekle ----------

print("\n🔧 M30 indikatörleri ekleniyor...")
merged = add_basic_indicators(merged, "M30")

print("🔧 H1 indikatörleri ekleniyor...")
merged = add_basic_indicators(merged, "H1")

print("🔧 H4 indikatörleri ekleniyor...")
merged = add_basic_indicators(merged, "H4")

# Temizlik: very early NaN'ler (indikatör başlangıçları)
before = merged.shape[0]
merged = merged.dropna().copy()
after = merged.shape[0]
print(f"\n🧼 NaN drop (indikatör başlangıçları): {before - after} satır atıldı.")
print(f"   ✅ Son shape: {merged.shape}")

# Index'i kolon yap
merged = merged.reset_index().rename(columns={"timestamp": "datetime"})

print("\n🪪 İlk 3 satır:")
print(merged.head(3))

# ---------- 6) Kaydet ----------

print(f"\n💾 Kaydediliyor: {OUT_PATH}")
merged.to_parquet(OUT_PATH, index=False)

print("=" * 80)
print("✅ BİTTİ: nasdaq_full_from5m_clean.parquet hazır (M30 merkezli, H1/H4 + indikatörler)")
print("=" * 80)