import os
import pandas as pd
import numpy as np

BASE = os.getcwd()
print(f"📂 Çalışma klasörü: {BASE}")

files = {
    "M30": "nasdaq.csv",
    "H1" : "nasdaq60.csv",
    "H4" : "nasdaq240.csv",
}

dfs = {}
cols_map = {}

print("\n================ DOSYALARI YÜKLE =================")
for name, fname in files.items():
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"❌ {name}: {fname} bulunamadı!")
        continue

    df = pd.read_csv(path)
    dfs[name] = df
    cols_map[name] = set(df.columns)

    print(f"\n=== {name}: {fname} ===")
    print("Shape:", df.shape)
    print("Toplam kolon sayısı:", len(df.columns))

if len(dfs) < 3:
    print("\n⚠️ Tüm timeframe'ler yüklenemedi, lütfen yukarıdaki eksik dosyaları düzelt.")
    raise SystemExit(1)

# =============== KOLON KARŞILAŞTIRMASI =================
print("\n================ KOLON KIYASLAMA =================")

all_cols = set.union(*cols_map.values())

for name in ["M30", "H1", "H4"]:
    others = [cname for cname in cols_map.keys() if cname != name]
    other_union = set.union(*(cols_map[o] for o in others))
    only_here = cols_map[name] - other_union
    print(f"\n🔍 Sadece {name} içinde olup diğerlerinde OLMAYAN kolonlar ({len(only_here)} adet):")
    if only_here:
        for c in sorted(only_here):
            print("  ", c)
    else:
        print("  (yok)")

# Ortak kolonlar
common_cols = set.intersection(*cols_map.values())
print(f"\n✅ Üç timeframe'de ORTAK olan kolon sayısı: {len(common_cols)}")

# =============== M30 SAYISAL ÖZET =================
m30 = dfs["M30"]

print("\n================ M30 (nasdaq.csv) SAYISAL ÖZET =================")
num_cols = m30.select_dtypes(include=[np.number]).columns.tolist()
print("Sayısal kolon sayısı:", len(num_cols))

if num_cols:
    desc = m30[num_cols].describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["unique"] = [m30[c].nunique() for c in desc.index]
    print("\nM30 sayısal kolon özet (ilk 40 kolon):")
    print(desc.head(40))

    const_cols = [c for c in num_cols if m30[c].nunique() == 1]
    if const_cols:
        print("\n⚠️ M30'da değeri HEP aynı olan (bilgi taşımayan) sayısal kolonlar:")
        for c in const_cols[:50]:
            print("  ", c)
        if len(const_cols) > 50:
            print(f"  ... (toplam {len(const_cols)} sabit kolon)")
    else:
        print("\n✅ M30'da tamamen sabit sayısal kolon yok.")

print("\n✅ Kıyaslama tamamlandı.")
