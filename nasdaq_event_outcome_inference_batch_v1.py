import logging
import pandas as pd

from nasdaq_event_outcome_inference_v1 import (
    load_event_outcome_engine,
    run_event_outcome_inference,
)
import numpy as np

THRESHOLD = 0.60  # max_prob eşiği – 0.6 üstü ise trade, altı PASS
INPUT_PARQUET = "./staging/nasdaq_event_outcomes_v1.parquet"
OUTPUT_PARQUET = "./staging/nasdaq_event_outcomes_with_preds_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_event_outcomes_with_preds_v1.csv"

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("==============================================================================")
    logger.info("🚀 NASDAQ EVENT OUTCOME INFERENCE BATCH v1 BAŞLIYOR")
    logger.info("==============================================================================")

    # 1) Event dataset'i yükle
    logger.info("📥 Event outcomes yükleniyor: %s", INPUT_PARQUET)
    try:
        df_events = pd.read_parquet(INPUT_PARQUET)
    except Exception as e:
        logger.error("❌ Event outcomes okunamadı: %s", e)
        return

    logger.info("✅ Event dataset shape: %s", df_events.shape)

    if df_events.empty:
        logger.error("❌ Event dataset boş, inference iptal.")
        return

    # 2) Engine yükle
    engine = load_event_outcome_engine()

    # 3) Inference çalıştır
    logger.info("🧠 Inference başlıyor (tüm eventler)...")
    df_pred = run_event_outcome_inference(df_events, engine)
    logger.info("✅ Inference bitti. Shape: %s", df_pred.shape)

    # 4) Basit metrik: modelin UP dediği yerlerde gerçek UP oranı vs
    if "future_dir" in df_pred.columns:
        # future_dir: 0 / 1 / 2 ise
        total = len(df_pred)

        # en yüksek olasılık UP olanlar
        mask_up = df_pred["pred_label"] == "UP"
        mask_down = df_pred["pred_label"] == "DOWN"

        up_preds = df_pred[mask_up]
        down_preds = df_pred[mask_down]

        logger.info("📊 Toplam event sayısı: %d", total)
        logger.info("   → Model UP dediği event sayısı: %d", len(up_preds))
        logger.info("   → Model DOWN dediği event sayısı: %d", len(down_preds))

        # Gerçek label'lar future_dir kolonundan
        if not up_preds.empty:
            true_up_rate = (up_preds["future_dir"] == 1).mean()
            logger.info("   ✅ UP tahminlerinde gerçek UP oranı: %.3f", true_up_rate)

        if not down_preds.empty:
            true_down_rate = (down_preds["future_dir"] == 2).mean()
            logger.info("   ✅ DOWN tahminlerinde gerçek DOWN oranı: %.3f", true_down_rate)

        # Bir de confidence filtresi örneği (max_prob > 0.6)
        high_conf = df_pred["max_prob"] > 0.6
        df_hc = df_pred[high_conf]
        logger.info("   → high confidence (max_prob>0.6) event sayısı: %d", len(df_hc))

        if not df_hc.empty:
            hc_up = df_hc[df_hc["pred_label"] == "UP"]
            hc_down = df_hc[df_hc["pred_label"] == "DOWN"]

            if not hc_up.empty:
                hc_up_true = (hc_up["future_dir"] == 1).mean()
                logger.info("   ✅ HIGH CONF UP (max_prob>0.6) gerçek UP oranı: %.3f", hc_up_true)

            if not hc_down.empty:
                hc_down_true = (hc_down["future_dir"] == 2).mean()
                logger.info("   ✅ HIGH CONF DOWN (max_prob>0.6) gerçek DOWN oranı: %.3f", hc_down_true)

    # 5) Kaydet
    logger.info("💾 Kaydediliyor...")
    df_pred.to_parquet(OUTPUT_PARQUET, index=False)
    df_pred.to_csv(OUTPUT_CSV, index=False)
    logger.info("✅ Parquet: %s", OUTPUT_PARQUET)
    logger.info("✅ CSV    : %s", OUTPUT_CSV)

    logger.info("==============================================================================")
    logger.info("✅ NASDAQ EVENT OUTCOME INFERENCE BATCH v1 TAMAMLANDI")
    logger.info("==============================================================================")


if __name__ == "__main__":
    main()