import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# PATHLER
# --------------------------------------------------------------------
MASTER_PATH = "./staging/nasdaq_master_features_v1.parquet"
SHORT_PLAYBOOK_JSON = "./models/setup_playbook_short_v1.json"

OUTPUT_PARQUET = "./staging/nasdaq_playbook_short_signals_from_master_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_playbook_short_signals_from_master_v1.csv"

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("signal_engine_short")


def load_master() -> pd.DataFrame:
    logger.info("=" * 79)
    logger.info("🚀 NASDAQ SIGNAL ENGINE (SHORT PLAYBOOK v1) BAŞLIYOR")
    logger.info("=" * 79)
    logger.info("📥 MASTER yükleniyor: %s", MASTER_PATH)
    df = pd.read_parquet(MASTER_PATH)
    logger.info("   ✅ MASTER shape: %s", df.shape)
    return df


def load_short_playbook() -> pd.DataFrame:
    logger.info("📥 Short playbook JSON yükleniyor: %s", SHORT_PLAYBOOK_JSON)
    with open(SHORT_PLAYBOOK_JSON, "r") as f:
        pb = json.load(f)

    # Dict -> DataFrame
    pb_df = pd.DataFrame(pb).T  # key'ler index, her biri bir setup
    logger.info("   ✅ Short playbook entry sayısı: %d", len(pb_df))

    # Beklenen kolonlar (build_short_v1 scriptinden gelenler)
    # event_type, direction, n_events, pct_up, pct_down, down_bias,
    # avg_max_up_pips, avg_max_down_pips, suggested_tp_pips, suggested_sl_pips, rr
    needed = [
        "event_type",
        "direction",
        "n_events",
        "pct_up",
        "pct_down",
        "down_bias",
        "avg_max_up_pips",
        "avg_max_down_pips",
        "suggested_tp_pips",
        "suggested_sl_pips",
        "rr",
    ]
    missing = [c for c in needed if c not in pb_df.columns]
    if missing:
        raise ValueError(f"❌ Short playbook JSON'da eksik kolonlar var: {missing}")

    return pb_df


def build_short_signals(master_df: pd.DataFrame, pb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Long signal engine'deki mantığın SHORT versiyonu:

    - MASTER tarafında her pattern (event_type) ayrı bir kolon (0/1 flag).
    - Short playbook'taki her event_type için:
        * master_df[event_type] == 1 olan satırları al
        * event_type kolonunu string olarak doldur
        * direction = SHORT
        * TP/SL pips = playbook değerleri
        * TP/SL fiyatı SHORT mantığıyla hesapla:
              TP = entry_price - tp_pips
              SL = entry_price + sl_pips
    """

    logger.info("🔧 SHORT sinyaller master datasından pattern kolonları ile üretiliyor...")

    df = master_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

       # Entry price kolonu seçimi (fiyat kaynağını otomatik bul)
       # Entry price kolonu seçimi (fiyat kaynağını otomatik bul - case-insensitive)
    PRICE_CANDIDATES_LOWER = [
        "entry_price",
        "close",
        "close_m30",
        "close_m15",
        "close_m5",
        "px_close",
    ]

    price_col = None
    for c in df.columns:
        if c.lower() in PRICE_CANDIDATES_LOWER:
            price_col = c
            break

    if price_col is None:
        # Hata durumunda kolonları loga basalım ki beraber bakabilelim
        cols_preview = ", ".join(list(df.columns)[:50])
        raise ValueError(
            "❌ MASTER'da entry_price/close tarzı bir fiyat kolonu bulunamadı. "
            f"İlk kolonlar: {cols_preview}"
        )

    logger.info("   ℹ️ Entry price için kullanılan kolon: %s", price_col)
    df["entry_price"] = df[price_col].astype(float)

    all_signals = []

    # Playbook'taki her SHORT pattern için master'dan event çıkar
    for row in pb_df.itertuples():
        etype = row.event_type

        if etype not in df.columns:
            logger.warning("⚠️ Master'da bu pattern için kolon yok, atlanıyor: %s", etype)
            continue

        # Bu pattern'in tetiklendiği satırlar
        sub = df[df[etype] == 1].copy()
        if sub.empty:
            logger.warning("ℹ️ Bu pattern için event bulunamadı: %s", etype)
            continue

        logger.info("   → Pattern %s için %d event bulundu.", etype, len(sub))

        # Temel setup bilgileri
        sub["event_type"] = etype
        sub["direction"] = "SHORT"

        tp_pips = float(row.suggested_tp_pips)
        sl_pips = float(row.suggested_sl_pips)

        sub["tp_pips"] = tp_pips
        sub["sl_pips"] = sl_pips

        entry = sub["entry_price"].astype(float)

        # 🔻 SHORT için TP/SL fiyatı
        sub["tp_price"] = entry - tp_pips
        sub["sl_price"] = entry + sl_pips

        # Playbook meta
        sub["rr"] = float(row.rr)
        sub["n_events_pb"] = int(row.n_events)
        sub["pct_up_pb"] = float(row.pct_up)
        sub["pct_down_pb"] = float(row.pct_down)
        sub["down_bias_pb"] = float(row.down_bias)
        sub["avg_max_up_pips_pb"] = float(row.avg_max_up_pips)
        sub["avg_max_down_pips_pb"] = float(row.avg_max_down_pips)

        all_signals.append(sub)

    if not all_signals:
        logger.warning("⚠️ Hiç SHORT sinyal üretilemedi, playbook pattern'ları master'da yok olabilir.")
        return pd.DataFrame()

    merged = pd.concat(all_signals, ignore_index=True)
    logger.info("   ✅ Üretilen SHORT sinyal sayısı: %d", len(merged))

    # ---------------------------------------------------------
    # Context kolon alias'ları (master'da varsa kopyala)
    # ---------------------------------------------------------
    def copy_if_exists(src_col: str, dst_col: str):
        if src_col in merged.columns:
            merged[dst_col] = merged[src_col]

    copy_if_exists("signal_wave", "signal_wave_at_entry")
    copy_if_exists("signal_wave_label", "signal_wave_label_at_entry")
    copy_if_exists("wave_strength_pips", "wave_strength_pips_at_entry")
    copy_if_exists("wave_duration_bars", "wave_duration_bars_at_entry")
    copy_if_exists("up_move_pips", "up_move_pips_at_entry")
    copy_if_exists("down_move_pips", "down_move_pips_at_entry")
    copy_if_exists("up_duration_bars", "up_duration_bars_at_entry")
    copy_if_exists("down_duration_bars", "down_duration_bars_at_entry")

    copy_if_exists("chan_is_up_M30", "chan_is_up_M30_at_entry")
    copy_if_exists("chan_is_down_M30", "chan_is_down_M30_at_entry")
    copy_if_exists("is_range_M30", "is_range_M30_at_entry")
    copy_if_exists("near_lower_chan_M30", "near_lower_chan_M30_at_entry")
    copy_if_exists("near_upper_chan_M30", "near_upper_chan_M30_at_entry")

    copy_if_exists("sr_support_price_at_entry", "sr_support_price_at_entry")
    copy_if_exists("sr_support_strength_at_entry", "sr_support_strength_at_entry")
    copy_if_exists("sr_support_distance_pips_at_entry", "sr_support_distance_pips_at_entry")
    copy_if_exists("sr_resistance_price_at_entry", "sr_resistance_price_at_entry")
    copy_if_exists("sr_resistance_strength_at_entry", "sr_resistance_strength_at_entry")
    copy_if_exists("sr_resistance_distance_pips_at_entry", "sr_resistance_distance_pips_at_entry")
    copy_if_exists("sr_near_support_at_entry", "sr_near_support_at_entry")
    copy_if_exists("sr_near_resistance_at_entry", "sr_near_resistance_at_entry")
    copy_if_exists("nearest_sr_type", "nearest_sr_type")
    copy_if_exists("nearest_sr_dist_pips", "nearest_sr_dist_pips")

    # Çıkış kolonlarını long sinyallere mümkün olduğunca paralel tutalım
    cols = [
        "timestamp",
        "event_type",
        "direction",
        "entry_price",
        "tp_pips",
        "sl_pips",
        "tp_price",
        "sl_price",
        "rr",
        "n_events_pb",
        "pct_up_pb",
        "pct_down_pb",
        "down_bias_pb",
        "avg_max_up_pips_pb",
        "avg_max_down_pips_pb",
        "sr_support_price_at_entry",
        "sr_support_strength_at_entry",
        "sr_support_distance_pips_at_entry",
        "sr_resistance_price_at_entry",
        "sr_resistance_strength_at_entry",
        "sr_resistance_distance_pips_at_entry",
        "sr_near_support_at_entry",
        "sr_near_resistance_at_entry",
        "nearest_sr_type",
        "nearest_sr_dist_pips",
        "signal_wave_at_entry",
        "signal_wave_label_at_entry",
        "wave_strength_pips_at_entry",
        "wave_duration_bars_at_entry",
        "up_move_pips_at_entry",
        "down_move_pips_at_entry",
        "up_duration_bars_at_entry",
        "down_duration_bars_at_entry",
        "chan_is_up_M30_at_entry",
        "chan_is_down_M30_at_entry",
        "is_range_M30_at_entry",
        "near_lower_chan_M30_at_entry",
        "near_upper_chan_M30_at_entry",
    ]

    final_cols = [c for c in cols if c in merged.columns]
    signals = merged[final_cols].copy()

    logger.info("   ✅ Çıkış kolon sayısı: %d", len(signals.columns))
    return signals

def save_signals(df: pd.DataFrame) -> None:
    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("💾 Short sinyaller (Parquet): %s", OUTPUT_PARQUET)
    logger.info("💾 Short sinyaller (CSV)    : %s", OUTPUT_CSV)


def main():
    master_df = load_master()
    short_pb_df = load_short_playbook()
    short_signals = build_short_signals(master_df, short_pb_df)
    save_signals(short_signals)

    logger.info("=" * 79)
    logger.info("✅ NASDAQ SIGNAL ENGINE (SHORT PLAYBOOK v1) TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()