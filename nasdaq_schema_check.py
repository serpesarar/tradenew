#!/usr/bin/env python3
"""
NASDAQ RAW -> CLEAN MULTI-TF BASE PARQUET

Kullanılan dosyalar:
    - nasdaq.csv        (ana timeframe, M30 varsayıyoruz)
    - nasdaq60.csv      (H1)
    - nasdaq240.csv     (H4)

Çıktı:
    - nasdaq_full_base_M30_H1_H4.parquet

Yapılanlar:
    - Her dosyada datetime kolonu otomatik bulunur
    - Tekrarlayan timestamp'lar gruplanıp son satır alınır
    - target* ve future* kolonları DROPlanır (data leak önlemi)
    - Tüm numeric kolonlar float'a çevrilir
    - M30/H1/H4 kolonları suffix ile ayrılır: _M30, _H1, _H4
    - Üç timeframe timestamp üzerinden merge edilir
"""

import pandas as pd
import numpy as np
import os

BASE_FILE = "nasdaq.csv"
H1_FILE = "nasdaq60.csv"
H4_FILE = "nasdaq240.csv"

OUT_PATH = "nasdaq_full_base_M30_H1_H4.parquet"


# ------------------ yardımcı fonksiyonlar ------------------ #

def detect_datetime_col(df):
    """Datetime benzeri kolonu otomatik bul."""
    candidates = [c for c in df.columns
                  if any(k in c.lower() for k in ["datetime", "timestamp", "time", "date"])]
    if not candidates:
        raise ValueError("Datetime benzeri kolon bulunamadı. Lütfen manuel ayarla.")
    
    # Tercih sırası: datetime > timestamp > time > date
    priority = ["datetime", "timestamp", "time", "date"]
    for p in priority:
        for c in candidates:
            if c.lower() == p:
                return c
    return candidates[0]


def sanitize_df(df, timeframe_label):
    """
    - Datetime kolonu bul, 'timestamp' olarak yeniden adlandır.
    - target* ve future* kolonlarını sil.
    - Duplicate timestamp varsa son satırı bırak.
    - Numeric kolonları float'a çevir.
    """
    print(f"\n🧼 [{timeframe_label}] Temizlik başlıyor, shape={df.shape}")
    
    # 1) datetime / timestamp
    dt_col = detect_datetime_col(df)
    print(f"   🕒 Datetime kolonu: {dt_col}")
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col)
    df = df.rename(columns={dt_col: "timestamp"})
    
    # 2) duplicate timestamp temizliği
    dup_count = df.duplicated("timestamp").sum()
    if dup_count > 0:
        print(f"   ⚠️ {dup_count} adet aynı timestamp var → son satır bırakılıyor.")
        df = df.drop_duplicates("timestamp", keep="last")
    
    # 3) future / target kolonlarını sil (data leak önlemi)
    leak_cols = [c for c in df.columns
                 if ("target" in c.lower()) or ("future" in c.lower())]
    if leak_cols:
        print(f"   ⚠️ Data leak riskli kolonlar DROPlanıyor ({len(leak_cols)}):")
        print("      ", leak_cols[:10], "..." if len(leak_cols) > 10 else "")
        df = df.drop(columns=leak_cols)
    
    # 4) numeric konversiyon
    non_ts_cols = [c for c in df.columns if c != "timestamp"]
    for c in non_ts_cols:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before_na = len(df)
    df = df.dropna(subset=["timestamp"])
    print(f"   ✅ Temizlik sonrası shape={df.shape} (timestamp NaN drop={before_na - len(df)})")
    
    return df


def add_suffix(df, suffix):
    """timestamp dışındaki tüm kolonlara suffix ekle."""
    rename_map = {}
    for c in df.columns:
        if c == "timestamp":
            continue
        rename_map[c] = f"{c}{suffix}"
    return df.rename(columns=rename_map)


# ------------------ ana pipeline ------------------ #

def main():
    print("=" * 80)
    print("🚀 NASDAQ RAW -> MULTI-TIMEFRAME BASE PARQUET")
    print("=" * 80)
    
    # 1) Dosyaları oku
    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"{BASE_FILE} bulunamadı.")
    if not os.path.exists(H1_FILE):
        raise FileNotFoundError(f"{H1_FILE} bulunamadı.")
    if not os.path.exists(H4_FILE):
        raise FileNotFoundError(f"{H4_FILE} bulunamadı.")
    
    print(f"📥 {BASE_FILE} okunuyor...")
    df_base = pd.read_csv(BASE_FILE)
    print(f"   ✅ base shape: {df_base.shape}")
    
    print(f"📥 {H1_FILE} okunuyor...")
    df_h1 = pd.read_csv(H1_FILE)
    print(f"   ✅ H1 shape:   {df_h1.shape}")
    
    print(f"📥 {H4_FILE} okunuyor...")
    df_h4 = pd.read_csv(H4_FILE)
    print(f"   ✅ H4 shape:   {df_h4.shape}")
    
    # 2) Temizlik
    df_base = sanitize_df(df_base, "M30 (base)")
    df_h1 = sanitize_df(df_h1, "H1")
    df_h4 = sanitize_df(df_h4, "H4")
    
    # 3) Suffix ekle
    df_base = add_suffix(df_base, "_M30")
    df_h1 = add_suffix(df_h1, "_H1")
    df_h4 = add_suffix(df_h4, "_H4")
    
    # 4) Merge (timestamp üzerinden asof merge)
    # base: referans zaman serisi
    df_base = df_base.sort_values("timestamp")
    df_h1 = df_h1.sort_values("timestamp")
    df_h4 = df_h4.sort_values("timestamp")
    
    print("\n🔗 M30 + H1 merge_asof...")
    merged = pd.merge_asof(
        df_base,
        df_h1,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1H")
    )
    
    print("🔗 (M30+H1) + H4 merge_asof...")
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        df_h4,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("4H")
    )
    
    # 5) Son temizlik
    before_drop = len(merged)
    # OLHV ve volume M30 zorunlu
    required = ["open_M30", "high_M30", "low_M30", "close_M30", "volume_M30"]
    for col in required:
        if col not in merged.columns:
            raise ValueError(f"Zorunlu kolon eksik: {col}. Lütfen nasdaq.csv içeriğini kontrol et.")
    
    merged = merged.dropna(subset=required)
    print(f"\n🧽 Gerekli kolonlarda NaN drop: {before_drop - len(merged)} satır")
    
    # 6) Numeric type zorlaması (XGBoost uyumu için)
    for c in merged.columns:
        if c == "timestamp":
            continue
        if merged[c].dtype == "object":
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    
    print("\n📊 SON ÖZET")
    print(f"   Shape: {merged.shape}")
    print("   İlk 3 satır:")
    print(merged.head(3))
    
    # 7) Kaydet
    merged.to_parquet(OUT_PATH, index=False)
    print("\n💾 Kaydedildi:", OUT_PATH)
    print("=" * 80)
    print("✅ NASDAQ MULTI-TF BASE PARQUET HAZIR")
    print("=" * 80)


if __name__ == "__main__":
    main()