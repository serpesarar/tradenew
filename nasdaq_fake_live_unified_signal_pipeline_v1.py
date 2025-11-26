import logging
from pathlib import Path

import pandas as pd

# Paths
LONG_FAKE_LIVE_PATH = "./staging/nasdaq_fake_live_signals_with_model_v1.parquet"
SHORT_FAKE_LIVE_PATH = "./staging/nasdaq_fake_live_short_signals_with_model_v1.parquet"

OUT_PARQUET = "./staging/nasdaq_fake_live_unified_signals_with_model_v1.parquet"
OUT_CSV = "./staging/nasdaq_fake_live_unified_signals_with_model_v1.csv"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_live_unified")


def load_long_short() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 78)
    logger.info("🚀 NASDAQ FAKE-LIVE UNIFIED SIGNAL PIPELINE v1 BAŞLIYOR")
    logger.info("=" * 78)

    logger.info("📥 LONG fake-live sinyaller yükleniyor: %s", LONG_FAKE_LIVE_PATH)
    long_df = pd.read_parquet(LONG_FAKE_LIVE_PATH)
    logger.info("   ✅ LONG fake-live sinyaller yüklendi. shape=%s", long_df.shape)

    logger.info("📥 SHORT fake-live sinyaller yükleniyor: %s", SHORT_FAKE_LIVE_PATH)
    short_df = pd.read_parquet(SHORT_FAKE_LIVE_PATH)
    logger.info("   ✅ SHORT fake-live sinyaller yüklendi. shape=%s", short_df.shape)

    return long_df, short_df


def unify_long_short(long_df: pd.DataFrame, short_df: pd.DataFrame) -> pd.DataFrame:
    # Güvenlik için direction/final_action kolonlarını normalize edelim
    long_df = long_df.copy()
    short_df = short_df.copy()

    # LONG tarafında direction yoksa doldur
    if "direction" not in long_df.columns:
        long_df["direction"] = "LONG"
    else:
        long_df["direction"] = long_df["direction"].fillna("LONG")

    if "direction" not in short_df.columns:
        short_df["direction"] = "SHORT"
    else:
        short_df["direction"] = short_df["direction"].fillna("SHORT")

    # final_action her iki tarafta da olmalı; yoksa set et
    if "final_action" not in long_df.columns:
        long_df["final_action"] = "LONG"
    if "final_action" not in short_df.columns:
        short_df["final_action"] = "SHORT"

    # Kolon setlerini hizala (eksik kolonlar için NaN oluştur)
    all_cols = sorted(set(long_df.columns) | set(short_df.columns))
    long_df = long_df.reindex(columns=all_cols)
    short_df = short_df.reindex(columns=all_cols)

    unified = pd.concat([long_df, short_df], ignore_index=True)
    unified = unified.sort_values("timestamp").reset_index(drop=True)

    # 🔧 Tip fix: kategorik kolonları string'e zorla (pyarrow parquet hatasını engelle)
    cat_cols = [
        "pred_label",
        "recommendation",
        "final_action",
        "direction",
        "future_dir_label",
        "tp_sl_result",
        "tp_sl_result_label",
        "event_type",
        "nearest_sr_type",
        "regime_M30",
    ]
    for c in cat_cols:
        if c in unified.columns:
            unified[c] = unified[c].astype("string")

    # Bazı numerik kolonlar float olsun (olası karışıklıkları temizlemek için)
    num_cols = [
        "p_chop",
        "p_up",
        "p_down",
        "max_prob",
        "tp_pips",
        "sl_pips",
        "max_up_move_pips",
        "max_down_move_pips",
        "entry_price",
    ]
    for c in num_cols:
        if c in unified.columns:
            unified[c] = pd.to_numeric(unified[c], errors="coerce")

    logger.info("📊 Unified stream özet:")
    logger.info("   • Toplam satır          : %d", len(unified))
    for action in ["LONG", "SHORT", "PASS", "NO_PRED"]:
        cnt = (unified["final_action"] == action).sum() if "final_action" in unified.columns else 0
        logger.info("   • final_action=%-8s : %d", action, cnt)

    return unified


def save_unified(unified: pd.DataFrame) -> None:
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

    unified.to_parquet(OUT_PARQUET, index=False)
    unified.to_csv(OUT_CSV, index=False)

    logger.info("💾 Kaydedildi (Parquet): %s", OUT_PARQUET)
    logger.info("💾 Kaydedildi (CSV)    : %s", OUT_CSV)


def main():
    long_df, short_df = load_long_short()
    unified = unify_long_short(long_df, short_df)
    save_unified(unified)

    logger.info("=" * 78)
    logger.info("✅ NASDAQ FAKE-LIVE UNIFIED SIGNAL PIPELINE v1 TAMAMLANDI")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()