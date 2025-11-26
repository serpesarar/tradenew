#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ================== CONFIG ==================
EVENT_PATH = "./staging/nasdaq_event_outcomes_v1.parquet"

OUT_DIR = Path("./models")
OUT_DIR.mkdir(exist_ok=True)

STATS_ALL_CSV = OUT_DIR / "event_pattern_stats_all_v1.csv"
STATS_SUPPORT_CSV = OUT_DIR / "event_pattern_stats_support_v1.csv"
STATS_RESIST_CSV = OUT_DIR / "event_pattern_stats_resistance_v1.csv"
REPORT_TXT = OUT_DIR / "event_pattern_stats_report_v1.txt"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "event_pattern_stats_v1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ================== CORE FUNCTIONS ==================
def compute_stats_for_group(df: pd.DataFrame) -> dict:
    """
    Tek bir event_type grubu için istatistik hesaplar.
    df: sadece o event_type'a ait satırlar.
    """
    row = {}
    n = len(df)
    row["n_events"] = n

    # future_dir dağılımı (0=CHOP, 1=UP, 2=DOWN varsayımı)
    if "future_dir" in df.columns:
        vc = df["future_dir"].value_counts(normalize=True)
        row["pct_chop"] = float(vc.get(0, 0.0))
        row["pct_up"] = float(vc.get(1, 0.0))
        row["pct_down"] = float(vc.get(2, 0.0))
    else:
        row["pct_chop"] = row["pct_up"] = row["pct_down"] = np.nan

    # TP/SL sonucu varsa
    if "tp_sl_result" in df.columns:
        # Örn: 1 = TP, -1 = SL, 0 = none (senin label tanımına göre)
        vc_tp = df["tp_sl_result"].value_counts(normalize=True)
        row["pct_tp"] = float(vc_tp.get(1, 0.0))
        row["pct_sl"] = float(vc_tp.get(-1, 0.0))
        row["pct_tp_or_be"] = float(vc_tp.get(1, 0.0) + vc_tp.get(0, 0.0))
    else:
        row["pct_tp"] = row["pct_sl"] = row["pct_tp_or_be"] = np.nan

    # Move pips istatistikleri
    if "max_up_move_pips" in df.columns:
        row["avg_max_up_pips"] = float(df["max_up_move_pips"].mean())
        row["p90_max_up_pips"] = float(df["max_up_move_pips"].quantile(0.9))
    else:
        row["avg_max_up_pips"] = row["p90_max_up_pips"] = np.nan

    if "max_down_move_pips" in df.columns:
        row["avg_max_down_pips"] = float(df["max_down_move_pips"].mean())
        row["p90_max_down_pips"] = float(df["max_down_move_pips"].quantile(0.9))
    else:
        row["avg_max_down_pips"] = row["p90_max_down_pips"] = np.nan

    # Wave strength (bilgilendirici)
    if "wave_strength_pips" in df.columns:
        row["avg_wave_strength_pips"] = float(df["wave_strength_pips"].mean())
    else:
        row["avg_wave_strength_pips"] = np.nan

    # SR context oranları (varsa)
    if "sr_near_support" in df.columns:
        row["sr_near_support_rate"] = float(df["sr_near_support"].mean())
    else:
        row["sr_near_support_rate"] = np.nan

    if "sr_near_resistance" in df.columns:
        row["sr_near_resistance_rate"] = float(df["sr_near_resistance"].mean())
    else:
        row["sr_near_resistance_rate"] = np.nan

    # Channel / range context (varsa)
    for col in ["chan_is_up_M30", "chan_is_down_M30", "is_range_M30"]:
        if col in df.columns:
            row[f"{col}_rate"] = float(df[col].mean())
        else:
            row[f"{col}_rate"] = np.nan

    return row


def build_stats_table(events: pd.DataFrame, min_events: int = 50) -> pd.DataFrame:
    """
    Tüm event_type'lar için istatistik tablosu üretir.
    min_events: en az bu kadar event olan pattern'leri dikkate al.
    """
    logger.info("🔍 event_type bazında istatistikler hesaplanıyor...")
    if "event_type" not in events.columns:
        raise ValueError("event_type kolonu dataset'te yok!")

    groups = events.groupby("event_type")
    rows = []

    for ev_type, g in groups:
        n = len(g)
        if n < min_events:
            continue

        stats = compute_stats_for_group(g)
        stats["event_type"] = ev_type
        rows.append(stats)

    stats_df = pd.DataFrame(rows)

    if not stats_df.empty:
        stats_df = stats_df.sort_values("n_events", ascending=False)

    logger.info("✅ Toplam pattern (min_events=%d filtresinden geçen): %d", min_events, len(stats_df))
    return stats_df


def save_text_report(
    stats_all: pd.DataFrame,
    stats_support: pd.DataFrame | None,
    stats_resist: pd.DataFrame | None,
    path: Path
):
    """
    Özet text raporu kaydeder: top pattern'ler, UP/DOWN oranları vb.
    """
    with open(path, "w") as f:
        f.write("NASDAQ EVENT PATTERN STATS v1\n")
        f.write("=" * 80 + "\n\n")

        if stats_all is None or stats_all.empty:
            f.write("No stats available.\n")
            return

        # En çok görülen ilk 30 pattern
        f.write("TOP 30 PATTERNS BY COUNT (ALL EVENTS)\n")
        f.write("-" * 80 + "\n")
        cols_basic = [
            "event_type", "n_events",
            "pct_chop", "pct_up", "pct_down",
            "avg_max_up_pips", "avg_max_down_pips",
        ]
        f.write(
            stats_all[cols_basic]
            .head(30)
            .to_string(index=False)
        )
        f.write("\n\n")

        # UP oranı en yüksek pattern'ler (en az 200 event)
        if "pct_up" in stats_all.columns:
            f.write("TOP 30 PATTERNS BY UP RATE (n_events >= 200)\n")
            f.write("-" * 80 + "\n")
            top_up = (
                stats_all[stats_all["n_events"] >= 200]
                .sort_values("pct_up", ascending=False)
                .head(30)
            )
            f.write(
                top_up[cols_basic].to_string(index=False)
            )
            f.write("\n\n")

        # DOWN oranı en yüksek pattern'ler (en az 200 event)
        if "pct_down" in stats_all.columns:
            f.write("TOP 30 PATTERNS BY DOWN RATE (n_events >= 200)\n")
            f.write("-" * 80 + "\n")
            top_down = (
                stats_all[stats_all["n_events"] >= 200]
                .sort_values("pct_down", ascending=False)
                .head(30)
            )
            f.write(
                top_down[cols_basic].to_string(index=False)
            )
            f.write("\n\n")

        # Support-only özet
        if stats_support is not None and not stats_support.empty:
            f.write("TOP 20 PATTERNS NEAR SUPPORT (sr_near_support==1)\n")
            f.write("-" * 80 + "\n")
            f.write(
                stats_support[cols_basic]
                .head(20)
                .to_string(index=False)
            )
            f.write("\n\n")

        # Resistance-only özet
        if stats_resist is not None and not stats_resist.empty:
            f.write("TOP 20 PATTERNS NEAR RESISTANCE (sr_near_resistance==1)\n")
            f.write("-" * 80 + "\n")
            f.write(
                stats_resist[cols_basic]
                .head(20)
                .to_string(index=False)
            )
            f.write("\n\n")

    logger.info("💾 Text report kaydedildi: %s", path)


# ================== MAIN ==================
def main():
    logger.info("=" * 80)
    logger.info("🚀 EVENT PATTERN ANALYSIS v1 BAŞLIYOR")
    logger.info("=" * 80)

    # 1) Dataset yükle
    logger.info("📥 Event outcomes yükleniyor: %s", EVENT_PATH)
    events = pd.read_parquet(EVENT_PATH)
    logger.info("✅ Event dataset shape: %s", events.shape)

    # 2) Tüm eventler için istatistik
    stats_all = build_stats_table(events, min_events=50)
    stats_all.to_csv(STATS_ALL_CSV, index=False)
    logger.info("💾 Tüm event stats CSV: %s", STATS_ALL_CSV)

    # 3) Support-only (sr_near_support == 1) varsa
    stats_support = None
    if "sr_near_support" in events.columns:
        logger.info("📊 Support-only (sr_near_support == 1) analiz ediliyor...")
        support_df = events[events["sr_near_support"] == 1]
        logger.info("   Support event sayısı: %d", len(support_df))
        stats_support = build_stats_table(support_df, min_events=50)
        if not stats_support.empty:
            stats_support.to_csv(STATS_SUPPORT_CSV, index=False)
            logger.info("💾 Support stats CSV: %s", STATS_SUPPORT_CSV)
        else:
            logger.info("   (Support stats boş, CSV yazılmadı.)")

    # 4) Resistance-only (sr_near_resistance == 1) varsa
    stats_resist = None
    if "sr_near_resistance" in events.columns:
        logger.info("📊 Resistance-only (sr_near_resistance == 1) analiz ediliyor...")
        resist_df = events[events["sr_near_resistance"] == 1]
        logger.info("   Resistance event sayısı: %d", len(resist_df))
        stats_resist = build_stats_table(resist_df, min_events=50)
        if not stats_resist.empty:
            stats_resist.to_csv(STATS_RESIST_CSV, index=False)
            logger.info("💾 Resistance stats CSV: %s", STATS_RESIST_CSV)
        else:
            logger.info("   (Resistance stats boş, CSV yazılmadı.)")

    # 5) Text rapor
    save_text_report(stats_all, stats_support, stats_resist, REPORT_TXT)

    logger.info("=" * 80)
    logger.info("✅ EVENT PATTERN ANALYSIS v1 TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()