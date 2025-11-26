#!/usr/bin/env python3
import pandas as pd
import numpy as np

IN_PATH = "./staging/nasdaq_full_wave_v3.parquet"
OUT_PATH = "./staging/nasdaq_full_wave_patterns_v1.parquet"

def add_candlestick_patterns(df, prefix="M30"):
    o = df[f"Open_{prefix}"]
    h = df[f"High_{prefix}"]
    l = df[f"Low_{prefix}"]
    c = df[f"Close_{prefix}"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = (h - c).where(c >= o, h - o)
    lower = (o - l).where(c >= o, c - l)

    df[f"body_{prefix}"] = body
    df[f"range_{prefix}"] = rng
    df[f"upper_wick_{prefix}"] = upper
    df[f"lower_wick_{prefix}"] = lower

    # basic direction
    df[f"is_bull_{prefix}"] = (c > o).astype(int)
    df[f"is_bear_{prefix}"] = (c < o).astype(int)

    # doji (gövde / range çok küçük)
    df[f"is_doji_{prefix}"] = (
        (body / rng <= 0.1) & rng.notna()
    ).astype(int)

    # hammer / inverted hammer
    df[f"is_hammer_{prefix}"] = (
        (lower >= 2 * body) & (upper <= body)
    ).astype(int)
    df[f"is_inv_hammer_{prefix}"] = (
        (upper >= 2 * body) & (lower <= body)
    ).astype(int)

    # shooting star (bearish pin)
    df[f"is_shooting_star_{prefix}"] = (
        (upper >= 2 * body) & (c < o)
    ).astype(int)

    # engulfing patterns (previous bar)
    df[f"is_bull_engulf_{prefix}"] = (
        (c > o) &
        (df[f"is_bear_{prefix}"].shift(1) == 1) &
        (c >= df[f"Open_{prefix}"].shift(1)) &
        (o <= df[f"Close_{prefix}"].shift(1))
    ).astype(int)

    df[f"is_bear_engulf_{prefix}"] = (
        (c < o) &
        (df[f"is_bull_{prefix}"].shift(1) == 1) &
        (c <= df[f"Open_{prefix}"].shift(1)) &
        (o >= df[f"Close_{prefix}"].shift(1))
    ).astype(int)

    # inside bar / outside bar
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    df[f"is_inside_bar_{prefix}"] = (
        (h <= prev_h) & (l >= prev_l)
    ).astype(int)

    df[f"is_outside_bar_{prefix}"] = (
        (h >= prev_h) & (l <= prev_l)
    ).astype(int)

    return df

def add_swings_and_trend(df):
    h = df["High_M30"]
    l = df["Low_M30"]

    # 5-bar swing high/low (centered)
    swing_high = (
        (h.shift(2) < h.shift(1)) &
        (h.shift(1) < h) &
        (h > h.shift(-1)) &
        (h.shift(-1) > h.shift(-2))
    )
    swing_low = (
        (l.shift(2) > l.shift(1)) &
        (l.shift(1) > l) &
        (l < l.shift(-1)) &
        (l.shift(-1) < l.shift(-2))
    )
    df["swing_high_M30"] = swing_high.fillna(False).astype(int)
    df["swing_low_M30"] = swing_low.fillna(False).astype(int)

    # last & previous swing prices
    df["last_swing_high_price"] = np.where(df["swing_high_M30"] == 1, h, np.nan)
    df["last_swing_low_price"] = np.where(df["swing_low_M30"] == 1, l, np.nan)
    df["last_swing_high_price"] = df["last_swing_high_price"].ffill()
    df["last_swing_low_price"] = df["last_swing_low_price"].ffill()

    df["prev_swing_high_price"] = df["last_swing_high_price"].shift(1)
    df["prev_swing_low_price"] = df["last_swing_low_price"].shift(1)

    df["higher_high_M30"] = (
        (df["swing_high_M30"] == 1) &
        (df["prev_swing_high_price"].notna()) &
        (h > df["prev_swing_high_price"])
    ).astype(int)

    df["lower_high_M30"] = (
        (df["swing_high_M30"] == 1) &
        (df["prev_swing_high_price"].notna()) &
        (h < df["prev_swing_high_price"])
    ).astype(int)

    df["higher_low_M30"] = (
        (df["swing_low_M30"] == 1) &
        (df["prev_swing_low_price"].notna()) &
        (l > df["prev_swing_low_price"])
    ).astype(int)

    df["lower_low_M30"] = (
        (df["swing_low_M30"] == 1) &
        (df["prev_swing_low_price"].notna()) &
        (l < df["prev_swing_low_price"])
    ).astype(int)

    # simple HH/HL uptrend, LH/LL downtrend flags
    df["structure_up_M30"] = (
        (df["higher_high_M30"] == 1) | (df["higher_low_M30"] == 1)
    ).astype(int)
    df["structure_down_M30"] = (
        (df["lower_high_M30"] == 1) | (df["lower_low_M30"] == 1)
    ).astype(int)

    return df

def add_trend_channels(df, window=50, prefix="M30"):
    c = df[f"Close_{prefix}"].astype(float)
    mid = c.rolling(window).mean()
    std = c.rolling(window).std()

    df[f"chan_mid_{prefix}"] = mid
    df[f"chan_upper_{prefix}"] = mid + 2 * std
    df[f"chan_lower_{prefix}"] = mid - 2 * std
    df[f"chan_width_{prefix}"] = df[f"chan_upper_{prefix}"] - df[f"chan_lower_{prefix}"]

    # position inside channel
    df[f"chan_pos_{prefix}"] = (
        (c - df[f"chan_mid_{prefix}"]) / df[f"chan_width_{prefix}"].replace(0, np.nan)
    )

    # approx slope over 'window' bars
    df[f"chan_slope_{prefix}"] = (
        df[f"chan_mid_{prefix}"] - df[f"chan_mid_{prefix}"].shift(window)
    ) / float(window)

    df[f"chan_is_up_{prefix}"] = (df[f"chan_slope_{prefix}"] > 0).astype(int)
    df[f"chan_is_down_{prefix}"] = (df[f"chan_slope_{prefix}"] < 0).astype(int)

    # price near channel borders
    df[f"near_upper_chan_{prefix}"] = (df[f"chan_pos_{prefix}"] >= 0.5).astype(int)
    df[f"near_lower_chan_{prefix}"] = (df[f"chan_pos_{prefix}"] <= -0.5).astype(int)

    # tight range flag (narrow channel)
    df[f"is_range_{prefix}"] = (
        (df[f"chan_width_{prefix}"] / c.replace(0, np.nan) < 0.01)
    ).astype(int)

    return df

def main():
    print("=" * 80)
    print("🚀 NASDAQ PATTERN + TREND CHANNEL ENGINE (M30 merkezli)")
    print("=" * 80)
    print(f"📥 Veri yükleniyor: {IN_PATH}")
    df = pd.read_parquet(IN_PATH)

    # timestamp/datetime
    dt_col = None
    for cand in ["timestamp", "datetime", "time"]:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("timestamp/datetime kolonu bulunamadı.")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    if dt_col != "timestamp":
        df = df.rename(columns={dt_col: "timestamp"})

    print(f"   ✅ Shape (giriş): {df.shape}")

    # basic column check
    needed = ["Open_M30", "High_M30", "Low_M30", "Close_M30"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen M30 kolonları eksik: {missing}")

    # 1) Candlestick formasyonları
    print("\n🔧 Candlestick formasyonları hesaplanıyor (M30)...")
    df = add_candlestick_patterns(df, prefix="M30")

    # 2) Swing high/low + HH/HL/LH/LL market structure
    print("🔧 Swing high/low + market structure (HH/HL/LH/LL) hesaplanıyor...")
    df = add_swings_and_trend(df)

    # 3) Trend channel + range/touch bilgileri
    print("🔧 Trend kanalları hesaplanıyor (rolling 50 bar)...")
    df = add_trend_channels(df, window=50, prefix="M30")

    print("\n📊 Özet:")
    print(f"   ✅ Çıkış shape: {df.shape}")
    new_cols = [c for c in df.columns if any(
        tag in c for tag in [
            "body_M30", "range_M30", "is_doji_M30", "is_hammer_M30",
            "is_bull_engulf_M30", "swing_high_M30", "higher_high_M30",
            "chan_mid_M30", "chan_is_up_M30", "is_range_M30"
        ]
    )]
    print(f"   ➕ Eklenen feature sayısı (subset): {len(new_cols)}")
    print(f"   Örnek yeni kolonlar: {new_cols[:15]}")

    print(f"\n💾 Kaydediliyor: {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("✅ NASDAQ PATTERN + TREND CHANNEL ENGINE BİTTİ")
    print("=" * 80)

if __name__ == "__main__":
    main()