import logging
from datetime import timedelta

import numpy as np
import pandas as pd

INPUT_PATH = "./staging/nasdaq_event_outcomes_with_preds_v1.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nasdaq_event_outcome_timeslice_eval_v1")


def load_data() -> pd.DataFrame:
    logger.info("=" * 79)
    logger.info("🚀 NASDAQ EVENT OUTCOME TIME-SLICE EVAL v1 BAŞLIYOR")
    logger.info("=" * 79)

    logger.info("📥 Tahmin dataset yükleniyor: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    logger.info("   ✅ df shape: %s", df.shape)

    if "timestamp" not in df.columns:
        raise ValueError("❌ Dataset içinde 'timestamp' kolonu yok.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # future_dir_label yoksa future_dir'den üret
    if "future_dir_label" not in df.columns:
        def map_future(row):
            v = row.get("future_dir")
            # Zaten string ise direkt dön
            if isinstance(v, str):
                return v
            # Numeric encoding ise tahmini mapping
            if v == 1:
                return "UP"
            elif v == 2:
                return "DOWN"
            else:
                return "CHOP"

        df["future_dir_label"] = df.apply(map_future, axis=1)

    # pred_label yoksa pred_class vs. fallback
    if "pred_label" not in df.columns:
        raise ValueError("❌ Dataset içinde 'pred_label' kolonu yok, inference batch doğru çalışmış mı kontrol et.")

    return df


def compute_metrics(df: pd.DataFrame, name: str) -> None:
    """
    Belirli bir slice için temel classification metriklerini loglar.
    """
    if df.empty:
        logger.warning("⚠️ Slice '%s' için hiç satır yok.", name)
        return

    logger.info("-" * 79)
    logger.info("📊 SLICE: %s", name)
    logger.info("   • Satır sayısı: %d", len(df))

    y_true = df["future_dir_label"].astype(str)
    y_pred = df["pred_label"].astype(str)

    acc = (y_true == y_pred).mean()
    logger.info("   • Genel accuracy: %.3f", acc)

    # UP precision
    mask_pred_up = y_pred == "UP"
    if mask_pred_up.any():
        up_prec = (y_true[mask_pred_up] == "UP").mean()
    else:
        up_prec = np.nan

    # DOWN precision
    mask_pred_down = y_pred == "DOWN"
    if mask_pred_down.any():
        down_prec = (y_true[mask_pred_down] == "DOWN").mean()
    else:
        down_prec = np.nan

    logger.info("   • UP precision   : %.3f", up_prec)
    logger.info("   • DOWN precision : %.3f", down_prec)

    # High confidence subset (max_prob > 0.6)
    if "max_prob" in df.columns:
        high_conf = df[df["max_prob"] > 0.6]
        logger.info("   • High-conf satır sayısı (max_prob>0.6): %d", len(high_conf))
        if not high_conf.empty:
            hc_true = high_conf["future_dir_label"].astype(str)
            hc_pred = high_conf["pred_label"].astype(str)
            hc_acc = (hc_true == hc_pred).mean()

            hc_pred_up = hc_pred == "UP"
            hc_pred_down = hc_pred == "DOWN"

            hc_up_prec = (hc_true[hc_pred_up] == "UP").mean() if hc_pred_up.any() else np.nan
            hc_down_prec = (hc_true[hc_pred_down] == "DOWN").mean() if hc_pred_down.any() else np.nan

            logger.info("   • High-conf accuracy       : %.3f", hc_acc)
            logger.info("   • High-conf UP precision   : %.3f", hc_up_prec)
            logger.info("   • High-conf DOWN precision : %.3f", hc_down_prec)


def main():
    df = load_data()

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    logger.info("🕒 Tarih aralığı: %s  →  %s", min_ts, max_ts)

    # Son 1 yılı out-of-sample slice gibi ele al
    cutoff = max_ts - timedelta(days=365)
    logger.info("   • Cutoff (son 1 yıl): %s", cutoff)

    df_past = df[df["timestamp"] < cutoff].copy()
    df_last_year = df[df["timestamp"] >= cutoff].copy()

    # 1) Tüm dönem
    compute_metrics(df, "TÜM DÖNEM")

    # 2) Cutoff öncesi dönem
    compute_metrics(df_past, "CUTOFF ÖNCESİ (TRAIN TARZI)")

    # 3) Son 1 yıl
    compute_metrics(df_last_year, "SON 1 YIL (PSEUDO OOS)")

    logger.info("=" * 79)
    logger.info("✅ NASDAQ EVENT OUTCOME TIME-SLICE EVAL v1 TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()