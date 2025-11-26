import os
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# 1) Basit yardımcı fonksiyonlar
# -------------------------------
def read_first_existing(candidates, parse_dt=True, dt_col="datetime"):
    """
    Verilen yol adaylarından ilk bulduğunu okur.
    dt_col varsa datetime'e çevirir.
    """
    for rel in candidates:
        path = os.path.join(BASE_DIR, rel)
        if os.path.exists(path):
            print(f"📥 OK: {path}")
            df = pd.read_csv(path)
            if parse_dt and dt_col in df.columns:
                df[dt_col] = pd.to_datetime(df[dt_col])
            return df
    raise FileNotFoundError(f"Şu dosyalardan hiçbiri yok: {candidates}")


def add_prefix_timeframe(df, prefix, drop_targets=True):
    """
    H1 / H4 gibi timeframe'leri M30'a yapıştırmadan önce
    kolon isimlerine prefix ekler: H1_open, H4_macd_line gibi.
    """
    drop_list = set([
        "target", "target_simple", "target_sharpe", "target_atr",
        "target_composite", "target_old", "target_killer",
    ])

    cols = []
    for c in df.columns:
        if c == "datetime":
            cols.append(c)
        else:
            if drop_targets and c in drop_list:
                continue
            cols.append(c)

    out = df[cols].copy()
    new_cols = []
    for c in out.columns:
        if c == "datetime":
            new_cols.append("datetime")
        else:
            new_cols.append(f"{prefix}_{c}")
    out.columns = new_cols
    return out


def build_fundamental_from_folder(folder="fundamental"):
    """
    fundamental/ altındaki TÜM CSV dosyalarını toplar,
    tarih bazında tek geniş tabloya birleştirir.
    Her dosya: date + <dosyaadi>__kolon şeklinde gelir.
    """
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"ℹ️ '{folder}' klasörü yok, fundamental data atlanıyor.")
        return None

    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"ℹ️ '{folder}' içinde csv yok, fundamental data atlanıyor.")
        return None

    base = None
    print(f"📰 Fundamental klasörü bulundu, {len(csv_files)} dosya var.")
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"  → işleniyor: {path}  (prefix='{name}__')")

        df = pd.read_csv(path)

        # Tarih kolonu bulmaya çalış
        dt_col = None
        for cand in ["datetime", "date", "Date", "DATE", "time", "Time"]:
            if cand in df.columns:
                dt_col = cand
                break
        if dt_col is None:
            # fallback: ilk kolon tarih varsay
            dt_col = df.columns[0]

        df[dt_col] = pd.to_datetime(df[dt_col])
        df["date"] = df[dt_col].dt.date

        value_cols = [c for c in df.columns if c not in [dt_col, "date"]]
        if not value_cols:
            continue

        small = df[["date"] + value_cols].copy()
        small.columns = ["date"] + [f"{name}__{c}" for c in value_cols]

        if base is None:
            base = small
        else:
            base = base.merge(small, on="date", how="outer")

    if base is not None:
        base = base.sort_values("date").reset_index(drop=True)
        print("✅ Fundamental birleşimi tamam, shape:", base.shape)
    return base


# -------------------------------
# 2) ANA BUILD FONKSİYONU
# -------------------------------
def build():
    print("📂 Çalışma klasörü:", BASE_DIR)

    # 2.1) M30, H1, H4 verilerini yükle
    print("\n=== M30 (ana seri) ===")
    m30 = read_first_existing(["nasdaq.csv", "nasdaq30.csv"])
    m30 = m30.sort_values("datetime").reset_index(drop=True)
    print("M30 shape:", m30.shape)

    print("\n=== H1 (60m) ===")
    h1 = read_first_existing(["nasdaq60.csv", "nasdaq_60.csv"])
    h1 = h1.sort_values("datetime").reset_index(drop=True)
    print("H1 shape:", h1.shape)

    print("\n=== H4 (240m) ===")
    h4 = read_first_existing(["nasdaq240.csv", "nasdaq_240.csv"])
    h4 = h4.sort_values("datetime").reset_index(drop=True)
    print("H4 shape:", h4.shape)

    # 2.2) H1 / H4'ü prefix ile hazırlayıp asof merge
    h1p = add_prefix_timeframe(h1, "H1")
    h4p = add_prefix_timeframe(h4, "H4")

    merged = m30.copy()
    print("\n🔗 H1 asof merge (backward)...")
    merged = pd.merge_asof(
        merged.sort_values("datetime"),
        h1p.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    print("  → H1 merge sonrası shape:", merged.shape)

    print("🔗 H4 asof merge (backward)...")
    merged = pd.merge_asof(
        merged.sort_values("datetime"),
        h4p.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    print("  → H4 merge sonrası shape:", merged.shape)

    # 2.3) Fundamental data ekle
    print("\n=== Fundamental ekleme ===")
    # 1) Eğer hazır 'fundamental_timeseries.csv' varsa önce onu dene
    fund_ts_path = os.path.join(BASE_DIR, "fundamental_timeseries.csv")
    fund_ts = None

    if os.path.exists(fund_ts_path):
        print(f"📥 fundamental_timeseries.csv bulundu: {fund_ts_path}")
        fund_ts = pd.read_csv(fund_ts_path)
        # tarih normalize
        if "date" not in fund_ts.columns:
            fund_ts.rename(columns={fund_ts.columns[0]: "date"}, inplace=True)
        fund_ts["date"] = pd.to_datetime(fund_ts["date"]).dt.date
        fund_ts = fund_ts.sort_values("date").reset_index(drop=True)
        print("  → fundamental_timeseries shape:", fund_ts.shape)
    else:
        # 2) Yoksa fundamental klasöründen topla
        fund_ts = build_fundamental_from_folder("fundamental")

    if fund_ts is not None:
        merged["date"] = merged["datetime"].dt.date
        print("🔗 Fundamental merge (date bazlı, left join)...")
        merged = merged.merge(fund_ts, on="date", how="left")
        print("  → Fundamental sonrası shape:", merged.shape)
    else:
        print("ℹ️ Fundamental veri eklenmedi (dosya/klasör bulunamadı).")

    # 2.4) Makro regime dosyası varsa ekle
    regimes_path = os.path.join(BASE_DIR, "fundamental", "macro_regimes.csv")
    if os.path.exists(regimes_path):
        print(f"\n📥 macro_regimes bulundu: {regimes_path}")
        reg = pd.read_csv(regimes_path)
        if "date" not in reg.columns:
            reg.rename(columns={reg.columns[0]: "date"}, inplace=True)
        reg["date"] = pd.to_datetime(reg["date"]).dt.date

        if "date" not in merged.columns:
            merged["date"] = merged["datetime"].dt.date

        print("🔗 macro_regimes merge...")
        merged = merged.merge(reg, on="date", how="left", suffixes=("", "_regime"))
        print("  → macro_regimes sonrası shape:", merged.shape)
    else:
        print("ℹ️ macro_regimes.csv bulunamadı, atlıyorum.")

    # 2.5) Temizlik: inf/NaN doldurma
    print("\n🧹 Sayısal kolon temizliği (inf/NaN)...")
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].replace([np.inf, -np.inf], np.nan)

    # çok fazla NaN varsa önce ileri/geri doldur, sonra kalanları 0’a çek
    merged[num_cols] = (
        merged[num_cols]
        .fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(0.0)
    )

    merged = merged.sort_values("datetime").reset_index(drop=True)

    # 2.6) Kaydet
    out_parquet = os.path.join(BASE_DIR, "nasdaq_training_dataset_v2.parquet")
    out_sample = os.path.join(BASE_DIR, "nasdaq_training_dataset_v2_sample.csv")

    merged.to_parquet(out_parquet, index=False)
    merged.head(5000).to_csv(out_sample, index=False)

    print("\n✅ Build NASDAQ v2 TAMAMLANDI")
    print("  →", out_parquet, " | shape:", merged.shape)
    print("  →", out_sample, "  | (ilk 5000 satır)")

if __name__ == "__main__":
    build()
