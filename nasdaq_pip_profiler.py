import os
import numpy as np
import pandas as pd

# ==================== AYARLAR ====================
DATA_FILE = "nasdaq_training_dataset_v2.parquet"
TIME_COL = "datetime"
CLOSE_COL = "close"

# Kaç bar sonrası hareketi ölçelim?
HORIZONS = [4, 8, 16]  # istersen [2,4,8,16,32] yaparız

TRAIN_FRACTION = 0.8  # %80 train, %20 test

# İnceleyeceğimiz olaylar (event)
# col: kolon adı, direction: "up" (long bekleriz) veya "down" (short bekleriz)
EVENT_DEFS = [
    {"name": "near_support",          "col": "near_support",          "direction": "up"},
    {"name": "near_resistance",       "col": "near_resistance",       "direction": "down"},
    {"name": "near_major_support",    "col": "near_major_support",    "direction": "up"},
    {"name": "near_major_resistance", "col": "near_major_resistance", "direction": "down"},
    {"name": "volume_spike",          "col": "volume_spike",          "direction": "either"},
]

# ==================== YARDIMCI FONKSİYON ====================
def summarize_event(df, event_name, col, direction, horizons):
    """
    df: inceleyeceğimiz segment (train ya da test)
    event_name: örn "near_support"
    col: kolon adı
    direction: "up", "down" veya "either"
    horizons: liste [4,8,16,...]
    """
    if col not in df.columns:
        print(f"  ⚠️ Kolon yok: {col}, event '{event_name}' atlanıyor.")
        return

    mask_evt = df[col] == 1
    df_evt = df[mask_evt].copy()

    if df_evt.empty:
        print(f"  ⚠️ Event '{event_name}' hiç oluşmamış.")
        return

    print(f"\n  ▶ Event: {event_name}  (kolon: {col}, yön: {direction})")
    print(f"    Olay sayısı: {len(df_evt)}")

    for h in horizons:
        fwd_col = f"fwd_ret_{h}"
        if fwd_col not in df_evt.columns:
            print(f"    - h={h}: '{fwd_col}' yok, atlıyorum.")
            continue

        ret = df_evt[fwd_col].values
        # NaN temizle
        ret = ret[~np.isnan(ret)]
        if len(ret) == 0:
            print(f"    - h={h}: geçerli veri yok.")
            continue

        # Beklenen yön pozitif olsun diye "gain" tanımlıyoruz:
        if direction == "up":
            gain = ret
        elif direction == "down":
            gain = -ret  # aşağı hareket bizim lehimize
        else:
            gain = ret  # "either" ise işaretine bakmadan istatistik bakarız

        # Win = beklediğimiz yönde mi?
        if direction in ("up", "down"):
            win_mask = gain > 0
            win_rate = 100.0 * win_mask.mean()
        else:
            # "either" ise win anlamlı değil, NaN yapalım
            win_rate = np.nan

        median_gain = np.median(gain)
        p25 = np.percentile(gain, 25)
        p75 = np.percentile(gain, 75)
        mean_gain = np.mean(gain)
        max_gain = np.max(gain)
        min_gain = np.min(gain)

        print(f"    - h={h} bar sonrası:")
        print(f"        Örnek sayısı: {len(gain)}")
        if direction in ("up", "down"):
            print(f"        Win rate (beklenen yön): {win_rate:.1f}%")
        print(f"        Median: {median_gain:.2f} puan")
        print(f"        25% - 75%: [{p25:.2f}, {p75:.2f}] puan")
        print(f"        Ortalama: {mean_gain:.2f}  | Min: {min_gain:.2f}  | Max: {max_gain:.2f}")


# ==================== ANA AKIŞ ====================
if __name__ == "__main__":
    print("📂 Çalışma klasörü:", os.getcwd())
    print("📥 Veri yükleniyor:", DATA_FILE)

    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"❌ {DATA_FILE} bulunamadı.")

    df = pd.read_parquet(DATA_FILE)
    print("✅ Veri shape:", df.shape)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    if CLOSE_COL not in df.columns:
        raise SystemExit(f"❌ '{CLOSE_COL}' kolonu yok. Kolonlar: {list(df.columns)[:30]}")

    # ---------- İLERİ GETİRİ HESAPLARI ----------
    max_h = max(HORIZONS)
    for h in HORIZONS:
        df[f"fwd_ret_{h}"] = df[CLOSE_COL].shift(-h) - df[CLOSE_COL]

    # Son max_h barın forward'ı NaN olacak, at
    df = df.iloc[:-max_h].reset_index(drop=True)

    n = len(df)
    train_n = int(n * TRAIN_FRACTION)

    df_train = df.iloc[:train_n].copy()
    df_test  = df.iloc[train_n:].copy()

    print(f"\n📊 Train boyutu: {df_train.shape}, Test boyutu: {df_test.shape}")
    print(f"   Train aralığı: {df_train[TIME_COL].min()} → {df_train[TIME_COL].max()}")
    print(f"   Test  aralığı: {df_test[TIME_COL].min()} → {df_test[TIME_COL].max()}")

    print("\n================= TRAIN SET ANALİZİ (öğrenme) =================")
    for evt in EVENT_DEFS:
        summarize_event(df_train, evt["name"], evt["col"], evt["direction"], HORIZONS)

    print("\n================= TEST SET ANALİZİ (doğrulama) =================")
    for evt in EVENT_DEFS:
        summarize_event(df_test, evt["name"], evt["col"], evt["direction"], HORIZONS)

    print("\n✅ Pip / puan profili analizi tamamlandı.")