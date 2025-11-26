#!/usr/bin/env python3
import pandas as pd

PAIRS = [
    ("nasdaq.csv",   "nasdaq_5m_M30_part.csv", "nasdaq_M30_merged.csv"),
    ("nasdaq60.csv", "nasdaq_5m_H1_part.csv",  "nasdaq60_merged.csv"),
    ("nasdaq240.csv","nasdaq_5m_H4_part.csv",  "nasdaq240_merged.csv"),
]

def merge_two(old_path, new_path, out_path):
    print(f"\n🔗 Merge: {old_path} + {new_path} -> {out_path}")

    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)

    # datetime kolonunu datetime'a çevir
    for df in (df_old, df_new):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError(f"{old_path} / {new_path} içinde datetime benzeri kolon yok.")

    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # timestamp'a göre uniq + sıralama
    df_all = (
        df_all
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df_all.to_csv(out_path, index=False)
    print(f"   ✅ Son shape: {df_all.shape}")

def main():
    for old_path, part_path, out_path in PAIRS:
        merge_two(old_path, part_path, out_path)

    print("\n✅ Tüm raw dosyalar 5 dakikalık veri ile birleştirildi.")
    print("   Sonraki adım: nasdaq_build_synthetic_timestamps.py içinde")
    print("   DATA_PATH'leri bu *_merged.csv dosyalarına çevirip")
    print("   pipeline'ı baştan çalıştırmak.")

if __name__ == "__main__":
    main()