import pandas as pd
import numpy as np

DATA_FILE = "nasdaq_training_dataset_v2.parquet"
TIME_COL = "datetime"
CLOSE_COL = "close"   # NASDAQ M30 close kolonu

# === Bizim kullandığımız destek/direnç fonksiyonunun aynısı ===
def find_zones_price_action(
    df: pd.DataFrame,
    price_col: str,
    last_price: float,
    side: str,
    window: int = 600,          # biraz daha geniş tuttum
    zone_half_width: float = 50.0,  # toplam 100 puanlık bölge
    min_touches: int = 3,
    top_k: int = 3,
):
    # Son window barı al
    df = df.tail(window).copy()
    prices = df[price_col].to_numpy(dtype=float)

    levels = []
    # Basit yaklaşım: local min/max + price bins
    for i in range(2, len(prices) - 2):
        p = prices[i]
        left = prices[i-2:i]
        right = prices[i+1:i+3]

        if side == "support":
            if p <= min(left) and p <= min(right):
                levels.append(p)
        else:
            if p >= max(left) and p >= max(right):
                levels.append(p)

    if not levels:
        return []

    levels = sorted(levels)

    clusters = []
    for level in levels:
        if not clusters:
            clusters.append(
                {"level": level, "count": 1, "touches": 0}
            )
        else:
            last = clusters[-1]
            if abs(level - last["level"]) <= zone_half_width:
                new_count = last["count"] + 1
                last["level"] = (last["level"] * last["count"] + level) / new_count
                last["count"] = new_count
            else:
                clusters.append({"level": level, "count": 1, "touches": 0})

    prices_all = prices
    for c in clusters:
        lv = c["level"]
        low = lv - zone_half_width
        high = lv + zone_half_width
        mask = (prices_all >= low) & (prices_all <= high)
        c["touches"] = int(mask.sum())

    clusters = [c for c in clusters if c["touches"] >= min_touches]
    clusters.sort(key=lambda x: (-c["touches"],))

    return clusters[:top_k]

# === ANA TEST ===
print("📂 Veri yükleniyor:", DATA_FILE)
df = pd.read_parquet(DATA_FILE)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

df_plot = df[["datetime", CLOSE_COL]].copy()
last_price = float(df_plot[CLOSE_COL].iloc[-1])

print(f"✅ Veri shape: {df_plot.shape}")
print(f"📌 Son fiyat: {last_price:.2f}")

supports = find_zones_price_action(df_plot, CLOSE_COL, last_price, side="support")
resistances = find_zones_price_action(df_plot, CLOSE_COL, last_price, side="resistance")

print("\n=== DESTEKLER ===")
if not supports:
    print("❌ Hiç destek bulunamadı.")
else:
    for i, s in enumerate(supports, 1):
        dist = last_price - s["level"]
        print(f"S{i}: seviye={s['level']:.2f}, touches={s['touches']}, "
              f"mesafe=+{dist:.2f} puan (fiyat üstünde)")

print("\n=== DİRENÇLER ===")
if not resistances:
    print("❌ Hiç direnç bulunamadı.")
else:
    for i, r in enumerate(resistances, 1):
        dist = r["level"] - last_price
        print(f"R{i}: seviye={r['level']:.2f}, touches={r['touches']}, "
              f"mesafe=+{dist:.2f} puan (fiyat üstünde)")