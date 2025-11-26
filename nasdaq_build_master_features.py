#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = "./staging/nasdaq_full_wave_patterns_sr_v1.parquet"
OUT_PATH = "./staging/nasdaq_master_features_v1.parquet"

print("=" * 80)
print("🚀 NASDAQ MASTER FEATURE SET BUILDER (v1)")
print("=" * 80)
print(f"📥 Input: {IN_PATH}")

if not Path(IN_PATH).exists():
    raise FileNotFoundError(f"Girdi dosyası yok: {IN_PATH}")

df = pd.read_parquet(IN_PATH)
print(f"   ✅ Input shape: {df.shape}")

# ---------------------------------------------------------------------
# 1) META & LABEL kolonları ayır
# ---------------------------------------------------------------------
meta_cols = ["timestamp"]
label_cols = [
    "signal_wave", "signal_wave_label",
    "wave_strength_pips", "wave_duration_bars",
    "up_move_pips", "down_move_pips",
    "up_duration_bars", "down_duration_bars",
]

# varsa kesişim al
meta_cols = [c for c in meta_cols if c in df.columns]
label_cols = [c for c in label_cols if c in df.columns]

# Numeric kolonlar
numeric_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c])
]

# Base feature set (label hariç numeric)
base_feats = [
    c for c in numeric_cols
    if c not in label_cols
]

print(f"   🔢 Numeric kolon sayısı: {len(numeric_cols)}")
print(f"   🔢 Base feature sayısı (label hariç): {len(base_feats)}")

# ---------------------------------------------------------------------
# 2) Feature grupları tanımla (prefix / isim bazlı)
# ---------------------------------------------------------------------
def has_any_prefix(col, prefixes):
    return any(col.startswith(p) for p in prefixes)

core_price_feats = [
    c for c in base_feats
    if (
        c in ["Close", "Volume", "ema20", "ema50", "ema200",
              "atr14", "vol_20", "vol_60", "rsi14", "ret_20_z", "regime_id"]
        or c.endswith("_M30")
    )
]

pattern_feats = [
    c for c in base_feats
    if (
        c.startswith("body_") or c.startswith("range_")
        or c.startswith("is_")
        or c.startswith("swing_")
        or c.startswith("chan_")
    )
]

sr_feats = [
    c for c in base_feats
    if c.startswith("sr_")
]

# Geri kalan (diğer timeframe/macro vs.)
other_feats = [
    c for c in base_feats
    if c not in core_price_feats
    and c not in pattern_feats
    and c not in sr_feats
]

print("\n📂 Feature grupları:")
print(f"   • core_price_feats: {len(core_price_feats)}")
print(f"   • pattern_feats:    {len(pattern_feats)}")
print(f"   • sr_feats:         {len(sr_feats)}")
print(f"   • other_feats:      {len(other_feats)}")

# ---------------------------------------------------------------------
# 3) Master DF başlat: meta + label + base feature'lar
# ---------------------------------------------------------------------
cols_keep = meta_cols + base_feats + label_cols
cols_keep = list(dict.fromkeys(cols_keep))  # sırayı koru, duplicate temizle

master = df[cols_keep].copy()
start_cols = master.shape[1]
print(f"\n   ✅ Master başlangıç kolon sayısı: {start_cols}")

# ---------------------------------------------------------------------
# 4) Helper: grup içi tüm çiftli çarpımlar
# ---------------------------------------------------------------------
def add_pairwise_products(df, feats, prefix):
    """feats listesindeki kolonlar için a*b ürün featureları ekler."""
    feats = [c for c in feats if c in df.columns]
    n = len(feats)
    added = 0
    for i in range(n):
        c1 = feats[i]
        for j in range(i, n):
            c2 = feats[j]
            new_name = f"{prefix}{c1}__x__{c2}"
            df[new_name] = df[c1] * df[c2]
            added += 1
    return added

# ---------------------------------------------------------------------
# 5) 1) Tüm base feature'ların karesini ekle
# ---------------------------------------------------------------------
print("\n🔧 1) Square feature'lar (x^2) ekleniyor...")
sq_added = 0
for c in base_feats:
    new_name = f"{c}__sq"
    master[new_name] = master[c] * master[c]
    sq_added += 1

print(f"   ➕ Eklenen square feature sayısı: {sq_added}")

# ---------------------------------------------------------------------
# 6) 2) Core price group için tüm pairwise çarpımlar
# ---------------------------------------------------------------------
print("\n🔧 2) Core price pairwise ürünleri ekleniyor...")
core_added = add_pairwise_products(master, core_price_feats, prefix="core__")
print(f"   ➕ Eklenen core_price pairwise sayısı: {core_added}")

# ---------------------------------------------------------------------
# 7) 3) Pattern group pairwise çarpımlar
# ---------------------------------------------------------------------
print("\n🔧 3) Pattern pairwise ürünleri ekleniyor...")
pattern_added = add_pairwise_products(master, pattern_feats, prefix="pat__")
print(f"   ➕ Eklenen pattern pairwise sayısı: {pattern_added}")

# ---------------------------------------------------------------------
# 8) 4) Support/Resistance pairwise çarpımlar
# ---------------------------------------------------------------------
print("\n🔧 4) Support/Resistance pairwise ürünleri ekleniyor...")
sr_added = add_pairwise_products(master, sr_feats, prefix="sr__")
print(f"   ➕ Eklenen SR pairwise sayısı: {sr_added}")

# ---------------------------------------------------------------------
# 9) 5) Özel “cross-group” etkileşimler (trend x regime x SR)
# ---------------------------------------------------------------------
print("\n🔧 5) Manuel önemli cross-group etkileşimler ekleniyor...")

def safe_add(name, expr):
    master[name] = expr

# Bazı temel kolonlar varsa:
has_col = master.columns

# Trend vs regime
if "chan_is_up_M30" in has_col and "regime_id" in has_col:
    safe_add("cross_trend_regime",
             master["chan_is_up_M30"] * master["regime_id"])

# Trend vs daily momentum
if "chan_is_up_M30" in has_col and "ret_20_z" in has_col:
    safe_add("cross_trend_ret20z",
             master["chan_is_up_M30"] * master["ret_20_z"])

# SR yakınlık vs trend
if "sr_near_support" in has_col and "chan_is_up_M30" in has_col:
    safe_add("cross_near_support_trend",
             master["sr_near_support"] * master["chan_is_up_M30"])

if "sr_near_resistance" in has_col and "chan_is_up_M30" in has_col:
    safe_add("cross_near_resistance_trend",
             master["sr_near_resistance"] * master["chan_is_up_M30"])

# Volatilite vs SR uzaklık
if "sr_support_distance" in has_col and "vol_20" in has_col:
    safe_add("cross_supportdist_vol20",
             master["sr_support_distance"] * master["vol_20"])

if "sr_resistance_distance" in has_col and "vol_20" in has_col:
    safe_add("cross_resistancedist_vol20",
             master["sr_resistance_distance"] * master["vol_20"])

# RSI vs trend
if "rsi14" in has_col and "chan_is_up_M30" in has_col:
    safe_add("cross_rsi_trend",
             master["rsi14"] * master["chan_is_up_M30"])

# Daily ema kaymaları vs channel mid
if "chan_mid_M30" in has_col and "ema20" in has_col:
    safe_add("cross_chanmid_ema20",
             master["chan_mid_M30"] - master["ema20"])

if "chan_mid_M30" in has_col and "ema50" in has_col:
    safe_add("cross_chanmid_ema50",
             master["chan_mid_M30"] - master["ema50"])

# ---------------------------------------------------------------------
# 10) Son kontroller
# ---------------------------------------------------------------------
end_cols = master.shape[1]
print("\n📊 ÖZET")
print(f"   Başlangıç kolon sayısı: {start_cols}")
print(f"   Son kolon sayısı:       {end_cols}")
print(f"   Eklenen kolon sayısı:   {end_cols - start_cols}")

# NaN / Inf kontrol
nan_cols = master.columns[master.isna().any()].tolist()
inf_cols = []
for c in master.columns:
    if pd.api.types.is_numeric_dtype(master[c]):
        if np.isinf(master[c]).any():
            inf_cols.append(c)

print("\n🧼 NaN / Inf kontrolü:")
print(f"   NaN içeren kolon sayısı: {len(nan_cols)}")
print(f"   Inf içeren kolon sayısı: {len(inf_cols)}")

if nan_cols:
    print(f"   ⚠️ İlk 10 NaN kolon: {nan_cols[:10]}")
    # basit çözüm: ffill + bfill + 0
    master[nan_cols] = master[nan_cols].fillna(0.0)

if inf_cols:
    print(f"   ⚠️ İlk 10 Inf kolon: {inf_cols[:10]}")
    for c in inf_cols:
        master[c].replace([np.inf, -np.inf], 0.0, inplace=True)

# Örnek kolon isimleri
print("\n🧾 Örnek kolonlar:")
print("   İlk 10:", list(master.columns[:10]))
print("   Son 10:", list(master.columns[-10:]))

# ---------------------------------------------------------------------
# 11) Kaydet
# ---------------------------------------------------------------------
Path("./staging").mkdir(exist_ok=True)
master.to_parquet(OUT_PATH, index=False)
print(f"\n💾 Kaydedildi: {OUT_PATH}")

print("\n" + "=" * 80)
print("✅ NASDAQ MASTER FEATURE SET BUILDER BİTTİ (v1)")
print("=" * 80)