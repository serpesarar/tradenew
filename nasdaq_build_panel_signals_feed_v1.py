import logging
from pathlib import Path

import pandas as pd

UNIFIED_PATH = "./staging/nasdaq_fake_live_unified_signals_with_model_v1.parquet"

PANEL_PARQUET = "./staging/nasdaq_panel_signals_feed_v1.parquet"
PANEL_CSV = "./staging/nasdaq_panel_signals_feed_v1.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("panel_feed_builder")


def main():
    logger.info("=" * 79)
    logger.info("🚀 BUILD PANEL SIGNALS FEED v1 BAŞLIYOR")
    logger.info("=" * 79)

    df = pd.read_parquet(UNIFIED_PATH)
    logger.info("📥 Unified fake-live df yüklendi. shape=%s", df.shape)

    # Sadece gerçek trade alınan satırlar (LONG / SHORT)
    if "final_action" not in df.columns:
        raise ValueError("❌ 'final_action' kolonu yok, unified pipeline doğru mu çalıştı?")

    trade_mask = df["final_action"].isin(["LONG", "SHORT"])
    panel_df = df[trade_mask].copy()
    logger.info("✅ Sadece trade satırları filtrelendi. shape=%s", panel_df.shape)

    # side kolonu unified df'de yok, kendimiz üretelim
    # LONG → LONG, SHORT → SHORT (istersen BUY/SELL'e de çevirebiliriz)
    if "side" not in panel_df.columns:
        panel_df["side"] = panel_df["final_action"].map({"LONG": "LONG", "SHORT": "SHORT"})

    # Kolon subset'i – panel için kritik olanlar
    desired_cols = [
        # Zaman ve yön
        "timestamp",
        "side",            # LONG / SHORT
        "final_action",    # LONG / SHORT
        "direction",       # direction kolonu varsa
        "event_type",

        # Fiyat & TP/SL
        "entry_price",
        "tp_pips",
        "sl_pips",
        "rr",

        # Pattern / playbook istatistikleri
        "n_events",
        "pct_up",
        "pct_down",
        "up_bias",
        "avg_max_up_pips",
        "avg_max_down_pips",

        # Model tahminleri
        "p_chop",
        "p_up",
        "p_down",
        "pred_label",
        "max_prob",
        "recommendation",
        "future_dir_label",
        "tp_sl_result_label",

        # Gerçek outcome (backtest için)
        "future_dir",
        "tp_sl_result",
        "max_up_move_pips",
        "max_down_move_pips",

        # SR bilgisi
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

        # Wave / context
        "signal_wave_at_entry",
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
        "regime_M30",
    ]

    # Sadece df'de gerçekten olan kolonları al (short df’de olmayan bir iki kolon olabilir diye)
    keep_cols = [c for c in desired_cols if c in panel_df.columns]
    missing = [c for c in desired_cols if c not in panel_df.columns]

    if missing:
        logger.warning("⚠️ Panel feed'de olmayan kolonlar var, atlanıyor: %s", missing)

    panel_df = panel_df[keep_cols].copy()

    # Zamanı garantiye al
    if "timestamp" in panel_df.columns:
        panel_df["timestamp"] = pd.to_datetime(panel_df["timestamp"])

    # Zaman + side'a göre sort
    panel_df = panel_df.sort_values(["timestamp", "side"]).reset_index(drop=True)

    # Basit bir özet (side yerine final_action'dan sayalım)
    n_long = (panel_df["final_action"] == "LONG").sum() if "final_action" in panel_df.columns else 0
    n_short = (panel_df["final_action"] == "SHORT").sum() if "final_action" in panel_df.columns else 0
    logger.info("📊 Panel feed dağılımı:")
    logger.info("   • Toplam trade satırı: %d", len(panel_df))
    logger.info("   • LONG trades        : %d", n_long)
    logger.info("   • SHORT trades       : %d", n_short)

    Path(PANEL_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(PANEL_PARQUET, index=False)
    panel_df.to_csv(PANEL_CSV, index=False)

    logger.info("💾 Panel feed (Parquet): %s", PANEL_PARQUET)
    logger.info("💾 Panel feed (CSV)    : %s", PANEL_CSV)
    logger.info("=" * 79)
    logger.info("✅ BUILD PANEL SIGNALS FEED v1 TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()