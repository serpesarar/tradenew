#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SEARCH_DIRS = [
    Path("."),            # current repo
    Path("./staging"),
    Path("./data"),
    Path("./fundamental"),
    Path("./news"),
]

FILE_EXTS = [".parquet", ".csv", ".feather"]


def find_datasets():
    files = []
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for ext in FILE_EXTS:
            files.extend(base.rglob(f"*{ext}"))
    # Benzersiz ve sıralı
    return sorted(set(files))


def summarize_dataframe(df: pd.DataFrame, max_cols: int = 40):
    info = {}
    info["shape"] = df.shape
    info["columns"] = list(df.columns)
    info["dtypes"] = df.dtypes.astype(str).to_dict()
    # Basit istatistikler
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    info["numeric_cols"] = numeric_cols
    return info


def main():
    files = find_datasets()
    if not files:
        print("⚠️ Hiç dataset bulunamadı. SEARCH_DIRS içini kontrol et.")
        return

    print("=" * 120)
    print(f"📂 Bulunan dataset sayısı: {len(files)}")
    print("=" * 120)

    for path in files:
        print("\n" + "-" * 120)
        print(f"📄 Dosya: {path}")
        print("-" * 120)

        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".feather":
                df = pd.read_feather(path)
            else:
                print(f"  ❌ Desteklenmeyen format: {path.suffix}")
                continue
        except Exception as e:
            print(f"  ❌ Yükleme hatası: {e}")
            continue

        info = summarize_dataframe(df)

        print(f"  ✅ Shape: {info['shape'][0]:,} satır x {info['shape'][1]} kolon")
        print(f"  ✅ İlk 10 kolon ismi:")
        for c in info["columns"][:10]:
            print(f"     - {c}")

        if info["shape"][1] > 10:
            print(f"  ℹ️  Toplam kolon sayısı: {info['shape'][1]} (tam liste aşağıda)")
        print("\n  📋 Tüm kolon isimleri:")
        for c in info["columns"]:
            print(f"     - {c}")

        print("\n  🧬 dtypes:")
        for c, t in info["dtypes"].items():
            print(f"     - {c}: {t}")

        if info["numeric_cols"]:
            print("\n  📊 Örnek numeric kolonlar:")
            for c in info["numeric_cols"][:10]:
                print(f"     - {c}")
        else:
            print("\n  ℹ️ Numeric kolon yok gibi görünüyor.")

        print("\n  👀 İlk 5 satır:")
        print(df.head(5))

    print("\n" + "=" * 120)
    print("✅ Dataset inspection tamamlandı.")
    print("=" * 120)


if __name__ == "__main__":
    main()