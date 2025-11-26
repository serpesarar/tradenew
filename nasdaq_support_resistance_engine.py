#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- PARAMETRELER ----------------
INPUT_PATH  = "./staging/nasdaq_full_wave_patterns_v1.parquet"
OUTPUT_PATH = "./staging/nasdaq_full_wave_patterns_sr_v1.parquet"

PRICE_COL_HIGH = "High_M30"
PRICE_COL_LOW  = "Low_M30"
PRICE_COL_CLOSE = "Close_M30"

ZONE_WIDTH   = 50.0   # bir zonun toplam genişliği (point/pip)
MIN_TOUCHES  = 5      # bir zonun "ciddi" sayılması için min dokunuş
SWING_LEFT   = 2      # swing high/low tespiti için sol/sağ bar sayısı
SWING_RIGHT  = 2

print("=" * 80)
print("🚀 NASDAQ SUPPORT / RESISTANCE ENGINE (M30 merkezli)")
print("=" * 80)
print(f"📥 Veri yükleniyor: {INPUT_PATH}")

if not Path(INPUT_PATH).exists():
    raise FileNotFoundError(f"Girdi dosyası yok: {INPUT_PATH}")

df = pd.read_parquet(INPUT_PATH)
print(f"   ✅ Shape (giriş): {df.shape}")

# ---------------- GÜVENLİ KONTROLLER ----------------
for col in [PRICE_COL_HIGH, PRICE_COL_LOW, PRICE_COL_CLOSE]:
    if col not in df.columns:
        raise ValueError(f"Gerekli kolon yok: {col}")

# timestamp / datetime
dt_col = None
for c in ["timestamp", "datetime", "time"]:
    if c in df.columns:
        dt_col = c
        break
if dt_col is None:
    raise ValueError("timestamp/datetime kolonu bulunamadı.")

df[dt_col] = pd.to_datetime(df[dt_col])
df = df.sort_values(dt_col).reset_index(drop=True)

# ====================================================
# 1) SWING HIGH / SWING LOW TESPİTİ
# ====================================================
print("\n🔧 Swing high / swing low tespit ediliyor...")

high = df[PRICE_COL_HIGH]
low  = df[PRICE_COL_LOW]

# swing high: bulunduğu barın high'ı hem sol hem sağındaki barlardan yüksek
is_swing_high = (
    (high.shift(1).notna()) &
    (high.shift(-1).notna())
)

for i in range(1, SWING_LEFT + 1):
    is_swing_high &= high > high.shift(i)
for i in range(1, SWING_RIGHT + 1):
    is_swing_high &= high >= high.shift(-i)

# swing low: bulunduğu barın low'u hem sol hem sağındakilerden düşük
is_swing_low = (
    (low.shift(1).notna()) &
    (low.shift(-1).notna())
)

for i in range(1, SWING_LEFT + 1):
    is_swing_low &= low < low.shift(i)
for i in range(1, SWING_RIGHT + 1):
    is_swing_low &= low <= low.shift(-i)

df["swing_high_price"] = np.where(is_swing_high, high, np.nan)
df["swing_low_price"]  = np.where(is_swing_low, low, np.nan)

print(f"   ✅ Swing high sayısı: {int(is_swing_high.sum())}")
print(f"   ✅ Swing low  sayısı: {int(is_swing_low.sum())}")

# ====================================================
# 2) SWING NOKTALARINI ZONLARA CLUSTER ETME
# ====================================================
def build_zones(prices: pd.Series, zone_width: float, min_touches: int):
    """
    prices: swing high veya swing low fiyatları (NaN hariç)
    zone_width: zon genişliği (ör: 50 point)
    min_touches: bir zonun anlamlı sayılması için min dokunuş sayısı
    """
    prices = prices.dropna().sort_values().values
    if len(prices) == 0:
        return []

    zones = []
    current_center = prices[0]
    current_points = [prices[0]]

    for p in prices[1:]:
        if abs(p - current_center) <= zone_width:
            # aynı zon
            current_points.append(p)
            current_center = np.mean(current_points)
        else:
            # mevcut zonu kapat, yeni zon aç
            zones.append({
                "center": current_center,
                "touches": len(current_points)
            })
            current_center = p
            current_points = [p]

    # son zonu da ekle
    zones.append({
        "center": current_center,
        "touches": len(current_points)
    })

    # min_touches filtrele
    zones = [z for z in zones if z["touches"] >= min_touches]
    return zones

print("\n🔧 Resistance zonları (swing high) hesaplanıyor...")
res_zones = build_zones(df["swing_high_price"], ZONE_WIDTH, MIN_TOUCHES)
print(f"   ✅ Resistance zon sayısı: {len(res_zones)}")

print("🔧 Support zonları (swing low) hesaplanıyor...")
sup_zones = build_zones(df["swing_low_price"], ZONE_WIDTH, MIN_TOUCHES)
print(f"   ✅ Support zon sayısı: {len(sup_zones)}")

# zonları DataFrame'e çevir (ileride istersen ayrı kaydedebilirsin)
res_df = pd.DataFrame(res_zones)
sup_df = pd.DataFrame(sup_zones)
res_df["type"] = "RESISTANCE"
sup_df["type"] = "SUPPORT"

# ====================================================
# 3) HER BAR İÇİN EN YAKIN SUPPORT / RESISTANCE
# ====================================================
print("\n📐 Her bar için en yakın support/resistance zonu hesaplanıyor...")

close = df[PRICE_COL_CLOSE].values

sup_centers = np.array([z["center"] for z in sup_zones]) if len(sup_zones) > 0 else np.array([])
sup_touches = np.array([z["touches"] for z in sup_zones]) if len(sup_zones) > 0 else np.array([])

res_centers = np.array([z["center"] for z in res_zones]) if len(res_zones) > 0 else np.array([])
res_touches = np.array([z["touches"] for z in res_zones]) if len(res_zones) > 0 else np.array([])

support_price = np.full(len(df), np.nan)
support_strength = np.full(len(df), np.nan)
support_dist = np.full(len(df), np.nan)
support_flag = np.zeros(len(df), dtype=int)

resistance_price = np.full(len(df), np.nan)
resistance_strength = np.full(len(df), np.nan)
resistance_dist = np.full(len(df), np.nan)
resistance_flag = np.zeros(len(df), dtype=int)

for i, price in enumerate(close):
    # SUPPORT → fiyatın altındaki en yakın zon
    if sup_centers.size > 0:
        diffs_sup = price - sup_centers
        # sadece altındaki zonlar
        valid_sup = np.where(diffs_sup >= 0)[0]
        if len(valid_sup) > 0:
            idx = valid_sup[np.argmin(diffs_sup[valid_sup])]
            support_price[i] = sup_centers[idx]
            support_strength[i] = sup_touches[idx]
            support_dist[i] = diffs_sup[idx]
            if diffs_sup[idx] <= ZONE_WIDTH:
                support_flag[i] = 1

    # RESISTANCE → fiyatın üstündeki en yakın zon
    if res_centers.size > 0:
        diffs_res = res_centers - price
        valid_res = np.where(diffs_res >= 0)[0]
        if len(valid_res) > 0:
            idx = valid_res[np.argmin(diffs_res[valid_res])]
            resistance_price[i] = res_centers[idx]
            resistance_strength[i] = res_touches[idx]
            resistance_dist[i] = diffs_res[idx]
            if diffs_res[idx] <= ZONE_WIDTH:
                resistance_flag[i] = 1

df["sr_support_price"]       = support_price
df["sr_support_strength"]    = support_strength
df["sr_support_distance"]    = support_dist
df["sr_near_support"]        = support_flag  # 1 = 50 point içinde

df["sr_resistance_price"]    = resistance_price
df["sr_resistance_strength"] = resistance_strength
df["sr_resistance_distance"] = resistance_dist
df["sr_near_resistance"]     = resistance_flag  # 1 = 50 point içinde

# ====================================================
# 4) TEMİZLİK & KAYIT
# ====================================================
# Artık swing_* kolonları istersen bırakabiliriz, ileride işine yarar.
print("\n📊 Özet:")
print(f"   ✅ Çıkış shape: {df.shape}")
print("   Örnek yeni kolonlar: ",
      ["sr_support_price", "sr_support_strength",
       "sr_support_distance", "sr_near_support",
       "sr_resistance_price", "sr_resistance_strength",
       "sr_resistance_distance", "sr_near_resistance"])

Path("./staging").mkdir(exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\n💾 Kaydediliyor: {OUTPUT_PATH}")

print("\n" + "=" * 80)
print("✅ NASDAQ SUPPORT / RESISTANCE ENGINE BİTTİ")
print("=" * 80)