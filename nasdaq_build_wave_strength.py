#!/usr/bin/env python3
"""
NASDAQ WAVE STRENGTH & DURATION

Girdi:
    - ./staging/nasdaq_full_wave.parquet  (signal_wave + future_* + up/down_move_pips)

Çıktı:
    - ./staging/nasdaq_full_wave_strength.parquet

Yapılanlar:
    - Farklı horizon'lar için fut_pips_10 / 20 / 40 / 80 hesaplanıyor
    - wave_strength_pips: ilgili wave yönünde max pip
    - wave_duration_bars: o max pip'in kaçıncı barda geldiği
"""

import os
import numpy as np
import pandas as pd

IN_PATH = "./staging/nasdaq_full_wave.parquet"
OUT_PATH = "./staging/nasdaq_full_wave_strength.parquet"

PIP_SIZE = 1.0
HORIZONS = [10, 20, 40, 80]


def main():
    print("=" * 80)
    print("🚀 NASDAQ WAVE STRENGTH & DURATION")
    print("=" * 80)

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {IN_PATH}")

    os.makedirs("staging", exist_ok=True)

    print(f"📥 Veri yükleniyor: {IN_PATH}")
    df = pd.read_parquet(IN_PATH)
    print(f"   ✅ Shape: {df.shape}")

    if "close_M30" not in df.columns or "signal_wave" not in df.columns:
        raise ValueError("close_M30 veya signal_wave kolonları eksik.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    close = df["close_M30"].astype(float)

    # 1) Farklı horizon'lar için future pip hareketi
    print("\n🔧 future pip hareketleri hesaplanıyor...")
    for h in HORIZONS:
        fut_close = close.rolling(window=h, min_periods=1).max().shift(-h + 1)
        up_pips = (fut_close - close) / PIP_SIZE
        df[f"fut_pips_up_{h}"] = up_pips

        fut_close_min = close.rolling(window=h, min_periods=1).min().shift(-h + 1)
        down_pips = (fut_close_min - close) / PIP_SIZE
        df[f"fut_pips_down_{h}"] = down_pips

    # Son büyük horizon kadar son satırları at
    max_h = max(HORIZONS)
    before = len(df)
    df = df.iloc[:-max_h].copy()
    print(f"   NaN drop (son {max_h} bar): {before - len(df)} satır")

    # 2) wave_strength_pips & wave_duration_bars
    print("\n🔧 wave_strength_pips & wave_duration_bars hesaplanıyor...")

    signal = df["signal_wave"].values
    n = len(df)
    strength = np.zeros(n, dtype=float)
    duration = np.zeros(n, dtype=float)

    # tek bir ana horizon üzerinden güç & süre:
    MAIN_H = 80  # 80 barlık dalga içinde maksimum hareketi ölçüyoruz
    fut_max = close.rolling(window=MAIN_H, min_periods=1).max().shift(-MAIN_H + 1)
    fut_min = close.rolling(window=MAIN_H, min_periods=1).min().shift(-MAIN_H + 1)

    up_all = (fut_max - close) / PIP_SIZE
    down_all = (fut_min - close) / PIP_SIZE

    # duration için:
    # basitçe: gelecekteki window içinde max/min'in kaç bar uzakta olduğunu tahmin eden yaklaşım
    for i in range(n):
        if i + MAIN_H >= len(close):
            strength[i] = 0.0
            duration[i] = 0.0
            continue

        window = close.iloc[i : i + MAIN_H].values
        cur = window[0]

        if signal[i] == 1:  # LONG_WAVE
            idx = window.argmax()
            pip_move = (window[idx] - cur) / PIP_SIZE
            strength[i] = pip_move
            duration[i] = idx
        elif signal[i] == 2:  # SHORT_WAVE
            idx = window.argmin()
            pip_move = (window[idx] - cur) / PIP_SIZE
            strength[i] = pip_move
            duration[i] = idx
        else:  # CHOP
            strength[i] = 0.0
            duration[i] = 0.0

    df["wave_strength_pips"] = strength
    df["wave_duration_bars"] = duration

    # NaN / uç değer temizliği
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    print("\n📊 Örnek satırlar:")
    print(df[["timestamp", "signal_wave", "wave_strength_pips", "wave_duration_bars"]].head())

    print(f"\n💾 Kaydediliyor: {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)

    print("=" * 80)
    print("✅ BİTTİ: NASDAQ wave strength datası hazır.")
    print("=" * 80)


if __name__ == "__main__":
    main()