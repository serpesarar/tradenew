#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# ================== CONFIG ==================

MASTER_PATH      = "./staging/nasdaq_master_features_v1.parquet"
WAVE_PATH        = "./staging/nasdaq_full_wave_v3.parquet"
PATTERN_SR_PATH  = "./staging/nasdaq_full_wave_patterns_sr_v1.parquet"

# Timestamp & fiyat kolon adayları
TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "time"]
PRICE_CANDIDATES     = ["Close_M30", "close_M30", "close", "Close"]

# Pip/point ayarı (NAS100)
PIP_MULTIPLIER = 1.0

# Event sonrası horizon & TP/SL
FUTURE_H_BARS = 50   # event'ten sonra 50 bar
TP_PIPS       = 100
SL_PIPS       = 50

OUTPUT_PATH   = "./staging/nasdaq_event_outcomes_v1.parquet"

# ==========================================================

def find_timestamp_col(df, where:str):
    cols = df.columns
    for c in TIMESTAMP_CANDIDATES:
        if c in cols:
            print(f"✅ {where} timestamp kolonu: {c}")
            return c
    # fallback: içinde "time" geçen ilk kolon
    for c in cols:
        if "time" in c.lower() or "date" in c.lower():
            print(f"✅ {where} timestamp kolonu (fallback): {c}")
            return c
    raise ValueError(f"{where}: timestamp kolonu bulunamadı. Lütfen timestamp ismini kodda güncelle.")

def find_price_col(df):
    cols = df.columns
    # öncelikle aday listeden dene
    for c in PRICE_CANDIDATES:
        if c in cols:
            print(f"✅ PRICE kolonu: {c}")
            return c
    # fallback: içinde "close" geçen ilk numeric kolon
    for c in cols:
        if "close" in c.lower() and np.issubdtype(df[c].dtype, np.number):
            print(f"✅ PRICE kolonu (fallback): {c}")
            return c
    # son fallback: ilk numeric kolon (çok ekstrem durum)
    num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    if num_cols:
        print(f"⚠️ PRICE bulunamadı, fallback olarak ilk numeric kolon kullanılıyor: {num_cols[0]}")
        return num_cols[0]
    raise ValueError("PRICE kolonu bulunamadı. Lütfen fiyat kolonunun adını kodda elle belirt.")

def load_and_merge():
    print("📥 MASTER yükleniyor:", MASTER_PATH)
    master = pd.read_parquet(MASTER_PATH)

    print("📥 WAVE yükleniyor:", WAVE_PATH)
    wave = pd.read_parquet(WAVE_PATH)

    print("📥 PATTERN/SR yükleniyor:", PATTERN_SR_PATH)
    patt = pd.read_parquet(PATTERN_SR_PATH)

    # Timestamp kolonlarını bul
    ts_master = find_timestamp_col(master, "MASTER")
    ts_wave   = find_timestamp_col(wave, "WAVE")
    ts_patt   = find_timestamp_col(patt, "PATTERN/SR")

    # Ortak isim: "timestamp"
    master = master.rename(columns={ts_master: "timestamp"})
    wave   = wave.rename(columns={ts_wave: "timestamp"})
    patt   = patt.rename(columns={ts_patt: "timestamp"})

    # Timestamp normalize
    for name, df in [("MASTER", master), ("WAVE", wave), ("PATT", patt)]:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Wave'ten alacağımız ana kolonlar
    wave_cols = [
        "signal_wave",
        "signal_wave_label",
        "up_move_pips",
        "down_move_pips",
        "up_duration_bars",
        "down_duration_bars",
        "wave_strength_pips",
        "wave_duration_bars",
    ]
    wave_cols_present = ["timestamp"] + [c for c in wave_cols if c in wave.columns]

    # Merge: master + wave + pattern/SR
    merged = master.merge(
        wave[wave_cols_present],
        on="timestamp",
        how="left"
    )
    merged = merged.merge(
        patt,
        on="timestamp",
        how="left"
    )

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    print(f"✅ Merged shape: {merged.shape}")

    # 🔥 ÖNEMLİ: FİYAT KOLONUNU MERGED ÜZERİNDE TEKRAR BUL
    price_col = find_price_col(merged)

    return merged, price_col
def detect_event_columns(df: pd.DataFrame):
    """
    Event kolonlarını otomatik bul:
      - dtype: bool veya az unique'li integer (0/1)
      - ismi pattern / sr / support / resistance / channel / swing vb. içeriyor
    """
    print("\n🔍 Event kolonları tespit ediliyor...")

    keywords = [
        "pattern", "harmonic", "butterfly", "gartley", "abcd",
        "sr_", "support", "resistance", "zone", "level_touch",
        "channel", "trendline", "supply", "demand",
        "is_doji", "is_hammer", "engulf", "swing_high", "swing_low",
        "breakout", "break_down", "break_up"
    ]

    event_cols = []
    for c in df.columns:
        if c == "timestamp":
            continue
        col = df[c]
        # Sadece 0/1 / bool / az unique'li kolonlara bak
        if col.dtype == bool or (
            np.issubdtype(col.dtype, np.integer) and col.nunique() <= 5
        ):
            lc = c.lower()
            if any(k in lc for k in keywords):
                event_cols.append(c)

    # SR near flag’leri özellikle ekle
    for extra in ["sr_near_support", "sr_near_resistance"]:
        if extra in df.columns and extra not in event_cols:
            event_cols.append(extra)

    print(f"   ✅ Bulunan event kolon sayısı: {len(event_cols)}")
    if event_cols:
        print("   📋 Örnek:", event_cols[:20])
    else:
        print("   ⚠️ Event kolonu bulunamadı. Gerekirse keywords listesine bak.")
    return event_cols

def build_event_rows(df: pd.DataFrame, price_col: str, event_cols):
    """
    df içinden event_cols==1 olan satırlardan event tablosu çıkar.
    Her event tipi için ayrı satır üretir.
    """
    print("\n📌 Event satırları oluşturuluyor...")
    rows = []

    for ev_col in event_cols:
        ev_mask = df[ev_col].fillna(0).astype(int) == 1
        idxs = np.where(ev_mask.values)[0]
        print(f"   • {ev_col}: {len(idxs)} event")

        for idx in idxs:
            row = {
                "timestamp": df.at[idx, "timestamp"],
                "event_type": ev_col,
                "entry_price": df.at[idx, price_col],
            }

            # Wave context (varsa)
            for wc in [
                "signal_wave",
                "signal_wave_label",
                "wave_strength_pips",
                "wave_duration_bars",
                "up_move_pips",
                "down_move_pips",
                "up_duration_bars",
                "down_duration_bars",
            ]:
                if wc in df.columns:
                    row[wc] = df.at[idx, wc]

            # SR context (varsa)
            for sc in [
                "sr_support_price",
                "sr_support_strength",
                "sr_support_distance",
                "sr_near_support",
                "sr_resistance_price",
                "sr_resistance_strength",
                "sr_resistance_distance",
                "sr_near_resistance",
            ]:
                if sc in df.columns:
                    row[sc] = df.at[idx, sc]

            rows.append(row)

    events = pd.DataFrame(rows)
    print(f"   ✅ Toplam event satırı: {len(events)}")
    return events

def label_event_outcomes(df: pd.DataFrame, events: pd.DataFrame, price_col: str):
    """
    Her event için future path'e bakıp outcome label'ları üret:
      - max_up_move_pips
      - max_down_move_pips
      - tp_sl_result (1=TP, 0=SL, 2=NONE)
      - future_dir (0=CHOP, 1=UP, 2=DOWN)
    """
    print("\n🎯 Event outcome label'ları hesaplanıyor...")

    df = df.reset_index(drop=True).copy()
    events = events.copy()

    price_arr = df[price_col].values
    ts_arr = df["timestamp"].values
    idx_map = {t: i for i, t in enumerate(ts_arr)}

    labeled_rows = []

    for _, ev in events.iterrows():
        t_ev = ev["timestamp"]
        idx = idx_map.get(t_ev, None)
        if idx is None:
            continue

        entry_price = price_arr[idx]
        tp_level = entry_price + TP_PIPS / PIP_MULTIPLIER
        sl_level = entry_price - SL_PIPS / PIP_MULTIPLIER

        j_end = min(len(df) - 1, idx + FUTURE_H_BARS)
        future_slice = price_arr[idx+1:j_end+1]
        if len(future_slice) == 0:
            continue

        # Max up/down move
        max_up = (future_slice - entry_price).max() * PIP_MULTIPLIER
        max_down = (future_slice - entry_price).min() * PIP_MULTIPLIER

        # TP / SL hit sırası
        hit_tp = np.where(future_slice >= tp_level)[0]
        hit_sl = np.where(future_slice <= sl_level)[0]

        if len(hit_tp) == 0 and len(hit_sl) == 0:
            tp_sl_result = 2  # NONE
        elif len(hit_tp) > 0 and len(hit_sl) == 0:
            tp_sl_result = 1  # TP
        elif len(hit_sl) > 0 and len(hit_tp) == 0:
            tp_sl_result = 0  # SL
        else:
            tp_sl_result = 1 if hit_tp[0] < hit_sl[0] else 0

        # Final direction (UP / DOWN / CHOP)
        final_price = future_slice[-1]
        if final_price > entry_price + 5 / PIP_MULTIPLIER:
            future_dir = 1  # UP
        elif final_price < entry_price - 5 / PIP_MULTIPLIER:
            future_dir = 2  # DOWN
        else:
            future_dir = 0  # CHOP

        row = ev.to_dict()
        row.update({
            "max_up_move_pips": max_up,
            "max_down_move_pips": max_down,
            "tp_sl_result": tp_sl_result,
            "future_dir": future_dir,
        })
        labeled_rows.append(row)

    labeled = pd.DataFrame(labeled_rows)
    print(f"   ✅ Label'lanmış event sayısı: {len(labeled)}")
    return labeled

def main():
    merged, price_col = load_and_merge()
    event_cols = detect_event_columns(merged)

    if not event_cols:
        raise SystemExit("❌ Event kolonları bulunamadı. detect_event_columns içindeki keywords'i genişletmen gerekebilir.")

    events = build_event_rows(merged, price_col, event_cols)
    labeled = label_event_outcomes(merged, events, price_col)

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_path, index=False)

    print(f"\n💾 Kaydedildi: {out_path}")
    print("✅ Bitti.")

if __name__ == "__main__":
    main()