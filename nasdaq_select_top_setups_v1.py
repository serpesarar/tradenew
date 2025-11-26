#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ======================= CONFIG =======================
STATS_CSV_PATH = "./models/event_pattern_stats_all_v1.csv"

TOP_LONG_CSV_PATH = "./models/top_long_setups_v1.csv"
TOP_SHORT_CSV_PATH = "./models/top_short_setups_v1.csv"
REPORT_PATH = "./models/top_setups_report_v1.txt"

# 🔧 Threshold'lar – istersen bunlarla oynayabilirsin
MIN_EVENTS_LONG = 300
MIN_EVENTS_SHORT = 300

MIN_UP_RATE = 0.58       # long: en az %58 up
MIN_UP_BIAS = 0.10       # long: pct_up - pct_down en az %10
MIN_UP_ASYM_PIPS = 10.0  # long: avg_max_up_pips > |avg_max_down_pips| + 10

MIN_DOWN_RATE = 0.58       # short: en az %58 down
MIN_DOWN_BIAS = 0.10       # short: pct_down - pct_up en az %10
MIN_DOWN_ASYM_PIPS = 10.0  # short: |avg_max_down_pips| > avg_max_up_pips + 10

TOP_N_LONG = 50
TOP_N_SHORT = 50

# ======================= LOGGING =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./models/top_setups_v1.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


def load_stats(path: str) -> pd.DataFrame:
    """event_pattern_stats_all_v1.csv dosyasını yükle ve kolonları kontrol et."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stats CSV bulunamadı: {path}")

    logger.info("📥 Stats CSV yükleniyor: %s", path)
    df = pd.read_csv(path)

    required_cols = [
        "event_type", "n_events",
        "pct_chop", "pct_up", "pct_down",
        "avg_max_up_pips", "avg_max_down_pips"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}")

    logger.info("✅ Stats shape: %s", df.shape)
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bias ve asimetri metriklerini ekle."""
    df = df.copy()

    df["up_bias"] = df["pct_up"] - df["pct_down"]
    df["down_bias"] = df["pct_down"] - df["pct_up"]

    # Pip asimetrileri
    df["up_asym_pips"] = df["avg_max_up_pips"] - df["avg_max_down_pips"].abs()
    df["down_asym_pips"] = df["avg_max_down_pips"].abs() - df["avg_max_up_pips"]

    # Basit bir skor: bias * asimetri (istediğinde oynarsın)
    df["long_score"] = df["up_bias"] * df["up_asym_pips"]
    df["short_score"] = df["down_bias"] * df["down_asym_pips"]

    logger.info("✅ Türetilmiş metrikler eklendi (up_bias, down_bias, up_asym_pips, down_asym_pips)")
    return df


def select_top_long(df: pd.DataFrame) -> pd.DataFrame:
    """Top long setup'ları filtrele."""
    logger.info("🔍 Long setup filtresi uygulanıyor...")

    mask = (
        (df["n_events"] >= MIN_EVENTS_LONG) &
        (df["pct_up"] >= MIN_UP_RATE) &
        (df["up_bias"] >= MIN_UP_BIAS) &
        (df["up_asym_pips"] >= MIN_UP_ASYM_PIPS)
    )

    long_df = df[mask].copy()
    logger.info("   ✅ Long candidate sayısı (raw): %d", len(long_df))

    if long_df.empty:
        logger.warning("   ⚠️ Long setup bulunamadı, threshold'lar çok agresif olabilir.")
        return long_df

    # En iyi long setup'ları sırala
    long_df = long_df.sort_values(
        by=["long_score", "pct_up", "up_asym_pips", "n_events"],
        ascending=[False, False, False, False]
    )

    if len(long_df) > TOP_N_LONG:
        long_df = long_df.head(TOP_N_LONG)

    logger.info("   ✅ Long setup seçildi (final): %d", len(long_df))
    return long_df


def select_top_short(df: pd.DataFrame) -> pd.DataFrame:
    """Top short setup'ları filtrele."""
    logger.info("🔍 Short setup filtresi uygulanıyor...")

    mask = (
        (df["n_events"] >= MIN_EVENTS_SHORT) &
        (df["pct_down"] >= MIN_DOWN_RATE) &
        (df["down_bias"] >= MIN_DOWN_BIAS) &
        (df["down_asym_pips"] >= MIN_DOWN_ASYM_PIPS)
    )

    short_df = df[mask].copy()
    logger.info("   ✅ Short candidate sayısı (raw): %d", len(short_df))

    if short_df.empty:
        logger.warning("   ⚠️ Short setup bulunamadı, threshold'lar çok agresif olabilir.")
        return short_df

    # En iyi short setup'ları sırala
    short_df = short_df.sort_values(
        by=["short_score", "pct_down", "down_asym_pips", "n_events"],
        ascending=[False, False, False, False]
    )

    if len(short_df) > TOP_N_SHORT:
        short_df = short_df.head(TOP_N_SHORT)

    logger.info("   ✅ Short setup seçildi (final): %d", len(short_df))
    return short_df


def save_outputs(long_df: pd.DataFrame, short_df: pd.DataFrame):
    """CSV + text rapor kaydet."""
    Path("./models").mkdir(exist_ok=True)

    if not long_df.empty:
        long_df.to_csv(TOP_LONG_CSV_PATH, index=False)
        logger.info("💾 Long setup CSV: %s", TOP_LONG_CSV_PATH)
    else:
        logger.info("ℹ️ Long setup CSV yazılmadı (boş).")

    if not short_df.empty:
        short_df.to_csv(TOP_SHORT_CSV_PATH, index=False)
        logger.info("💾 Short setup CSV: %s", TOP_SHORT_CSV_PATH)
    else:
        logger.info("ℹ️ Short setup CSV yazılmadı (boş).")

    # Kısa text rapor
    with open(REPORT_PATH, "w") as f:
        f.write("TOP SETUPS REPORT v1\n")
        f.write("=" * 80 + "\n\n")

        f.write("LONG SETUPS (filtered)\n")
        f.write("-" * 80 + "\n")
        if long_df.empty:
            f.write("No long setups found with current thresholds.\n\n")
        else:
            f.write(long_df[[
                "event_type", "n_events",
                "pct_up", "pct_down", "up_bias",
                "avg_max_up_pips", "avg_max_down_pips",
                "up_asym_pips", "long_score"
            ]].to_string(index=False))
            f.write("\n\n")

        f.write("SHORT SETUPS (filtered)\n")
        f.write("-" * 80 + "\n")
        if short_df.empty:
            f.write("No short setups found with current thresholds.\n\n")
        else:
            f.write(short_df[[
                "event_type", "n_events",
                "pct_up", "pct_down", "down_bias",
                "avg_max_up_pips", "avg_max_down_pips",
                "down_asym_pips", "short_score"
            ]].to_string(index=False))
            f.write("\n")

    logger.info("💾 Text rapor kaydedildi: %s", REPORT_PATH)


def main():
    logger.info("=" * 80)
    logger.info("🚀 TOP SETUP SELECTION v1 BAŞLIYOR")
    logger.info("=" * 80)

    try:
        df = load_stats(STATS_CSV_PATH)
        df = add_derived_metrics(df)

        long_df = select_top_long(df)
        short_df = select_top_short(df)

        save_outputs(long_df, short_df)

        logger.info("=" * 80)
        logger.info("✅ TOP SETUP SELECTION v1 TAMAMLANDI")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception("❌ HATA: %s", e)


if __name__ == "__main__":
    main()