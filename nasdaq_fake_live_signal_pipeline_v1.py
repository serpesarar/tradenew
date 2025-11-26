import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PLAYBOOK_SIGNALS_PATH = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"
EVENT_PREDS_PATH = "./staging/nasdaq_event_outcomes_with_preds_v2.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_signals_with_model_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_fake_live_signals_with_model_v1.csv"

HIGH_CONF_THRESHOLD = 0.60  # Şimdilik kullanmıyoruz ama ileride threshold tuning için kalabilir.

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_live_signal_pipeline")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_future_dir_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    future_dir_label yoksa, future_dir'dan türet.
    - Eğer future_dir numeric ise: 0 = CHOP, 1 = DOWN, 2 = UP (v2 label encoder gibi)
    - Eğer string ise: direkt uppercase.
    """
    df = df.copy()

    if "future_dir_label" in df.columns:
        return df

    if "future_dir" not in df.columns:
        df["future_dir_label"] = np.nan
        return df

    col = df["future_dir"]

    if np.issubdtype(col.dtype, np.number):
        mapping = {0: "CHOP", 1: "DOWN", 2: "UP"}
        df["future_dir_label"] = col.map(mapping).fillna("UNKNOWN")
    else:
        # string / kategorik ise
        df["future_dir_label"] = col.astype(str).str.upper()

    return df


def _ensure_tp_sl_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    tp_sl_result_label yoksa, tp_sl_result'tan türet.

    Varsayılan mapping (event outcome tarafındaki numeric kodlardan bağımsız, generic):
      - Numeric ise:
          v > 0  → "TP"
          v < 0  → "SL"
          v == 0 → "BE"
      - String ise:
          "TP" / "SL" / "BE" / vs. uppercase olarak alınır.

    Böylece:
      - Hem inference tarafında numeric kod gelse,
      - Hem de direkt "TP"/"SL"/"BE" stringleri gelse,
    unified + backtest tarafında hep aynı label kullanılır.
    """
    df = df.copy()

    if "tp_sl_result_label" in df.columns:
        return df

    if "tp_sl_result" not in df.columns:
        df["tp_sl_result_label"] = np.nan
        return df

    raw = df["tp_sl_result"]
    num = pd.to_numeric(raw, errors="coerce")

    if num.notna().any():
        # Numeric parse edebildiklerimiz için sign-based generic mapping
        def _map_num(v):
            if pd.isna(v):
                return np.nan
            if v > 0:
                return "TP"
            if v < 0:
                return "SL"
            return "BE"

        mapped_from_num = num.map(_map_num)

        # Parse edilemeyenler için string'e bak
        raw_str = raw.astype("string").str.upper()
        res = mapped_from_num.where(num.notna(), raw_str)
    else:
        # Hiç numeric parse yoksa, tamamen string kabul et
        res = raw.astype("string").str.upper().fillna("UNKNOWN")

    df["tp_sl_result_label"] = res
    return df


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Playbook sinyallerini ve model tahminlerini yükler."""
    logger.info("=" * 79)
    logger.info("🚀 NASDAQ FAKE-LIVE SIGNAL PIPELINE v1 BAŞLIYOR")
    logger.info("=" * 79)

    logger.info("📥 Playbook sinyalleri yükleniyor: %s", PLAYBOOK_SIGNALS_PATH)
    sig_df = pd.read_parquet(PLAYBOOK_SIGNALS_PATH)
    logger.info("   ✅ Sinyal df shape: %s", sig_df.shape)

    logger.info("📥 Model tahminleri yükleniyor: %s", EVENT_PREDS_PATH)
    preds_df = pd.read_parquet(EVENT_PREDS_PATH)
    logger.info("   ✅ Tahmin df shape: %s", preds_df.shape)

    return sig_df, preds_df


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
def merge_signals_with_preds(sig_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Playbook sinyallerini, event outcome tahminleriyle merge eder.
    Join key: [timestamp, event_type, entry_price]

    NOT:
    - Karar mantığı için zorunlu kolonlar:
        timestamp, event_type, entry_price, p_chop, p_up, p_down, pred_label, max_prob
    - Diğer kolonlar (future_dir, tp_sl_result, max_up_move_pips vs.) sadece log / backtest için.
    """
    logger.info("🔗 Playbook sinyalleri ile model tahminleri merge ediliyor...")

    # Zorunlu kolonlar
    required_cols = [
        "timestamp",
        "event_type",
        "entry_price",
        "p_chop",
        "p_up",
        "p_down",
        "pred_label",
        "max_prob",
    ]

    # Opsiyonel kolonlar
    optional_cols = [
        "pred_class",
        "recommendation",
        "future_dir",
        "future_dir_label",
        "tp_sl_result",
        "tp_sl_result_label",
        "max_up_move_pips",
        "max_down_move_pips",
    ]

    missing_req = [c for c in required_cols if c not in preds_df.columns]
    if missing_req:
        raise ValueError(f"❌ Tahmin datasetinde zorunlu kolonlar eksik: {missing_req}")

    available_optional = [c for c in optional_cols if c in preds_df.columns]
    keep_cols = required_cols + available_optional

    preds_small = preds_df[keep_cols].copy()

    # Tip uyumu (timestamp + entry_price)
    sig_df = sig_df.copy()
    preds_small = preds_small.copy()

    sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
    preds_small["timestamp"] = pd.to_datetime(preds_small["timestamp"])

    sig_df["entry_price"] = sig_df["entry_price"].astype(float)
    preds_small["entry_price"] = preds_small["entry_price"].astype(float)

    merged = sig_df.merge(
        preds_small,
        on=["timestamp", "event_type", "entry_price"],
        how="left",
        suffixes=("", "_preds"),
    )

    # future_dir_label yoksa future_dir'dan üret
    merged = _ensure_future_dir_label(merged)

    # tp_sl_result_label yoksa tp_sl_result'tan üret
    merged = _ensure_tp_sl_result_label(merged)

    missing_preds = merged["pred_label"].isna().sum()
    if missing_preds > 0:
        logger.warning(
            "⚠️ %d sinyalde eşleşen model tahmini bulunamadı (pred_label is NaN)",
            missing_preds,
        )
    else:
        logger.info("   ✅ Tüm sinyaller için model tahmini bulundu.")

    logger.info("   ✅ Merge sonrası shape: %s", merged.shape)
    return merged


# -----------------------------------------------------------------------------
# Decision Logic
# -----------------------------------------------------------------------------
def apply_fake_live_logic(merged: pd.DataFrame) -> pd.DataFrame:
    """
    LONG fake - live karar mantığı:

      - Model tahmini yoksa → NO_PRED
      - direction LONG / BUY değilse → PASS
      - p_up, p_down, p_chop NaN ise → PASS
      - p_up en büyük ise → LONG
      - Aksi halde → PASS

    Buradaki mantık "golden" versiyon: çok basit, saf model yön seçimi.
    """
    logger.info("🧠 Fake-live karar mantığı uygulanıyor...")

    df = merged.sort_values("timestamp").reset_index(drop=True).copy()

    def decide_row(row) -> str:
        # Model tahmini yoksa
        if pd.isna(row.get("pred_label")):
            return "NO_PRED"

        # Bu dosya LONG playbook setupları için; direction farklıysa trade alma
        if row.get("direction") not in ("LONG", "BUY", None):
            return "PASS"

        p_up = row.get("p_up", np.nan)
        p_down = row.get("p_down", np.nan)
        p_chop = row.get("p_chop", np.nan)

        # Eğer probabilitelerden biri bile yoksa, trade alma
        if pd.isna(p_up) or pd.isna(p_down) or pd.isna(p_chop):
            return "PASS"

        # 🔥 Golden mantık: UP en yüksekse LONG aç
        if (p_up >= p_down) and (p_up >= p_chop):
            return "LONG"
        else:
            return "PASS"

    df["final_action"] = df.apply(decide_row, axis=1)

    # Quick metrikler
    total = len(df)
    n_long = (df["final_action"] == "LONG").sum()
    n_pass = (df["final_action"] == "PASS").sum()
    n_nopred = (df["final_action"] == "NO_PRED").sum()

    logger.info("📊 Fake-live karar dağılımı:")
    logger.info("   • Toplam satır: %d", total)
    logger.info("   • LONG       : %d (%.2f%%)", n_long, 100 * n_long / total if total else 0)
    logger.info("   • PASS       : %d (%.2f%%)", n_pass, 100 * n_pass / total if total else 0)
    logger.info("   • NO_PRED    : %d (%.2f%%)", n_nopred, 100 * n_nopred / total if total else 0)

    # Performans log (kararı etkilemez)
    mask_long = df["final_action"] == "LONG"
    if mask_long.any():
        long_df = df[mask_long].copy()

        # 1) Directional win-rate
        if "future_dir_label" in long_df.columns:
            dir_col = long_df["future_dir_label"]
            dir_win_rate = (dir_col == "UP").mean()
        else:
            dir_col = long_df.get("future_dir")
            if dir_col is not None:
                if np.issubdtype(dir_col.dtype, np.number):
                    dir_win_rate = (dir_col == 2).mean()
                else:
                    dir_win_rate = (dir_col.astype(str).str.upper() == "UP").mean()
            else:
                dir_win_rate = float("nan")

        # 2) TP/SL/BE pips bazlı (varsa) – sadece log için
        required_cols = {"tp_pips", "sl_pips", "max_up_move_pips", "max_down_move_pips"}
        if required_cols.issubset(long_df.columns):
            tp_pips = long_df["tp_pips"].astype(float)
            sl_pips = long_df["sl_pips"].astype(float)
            max_up = long_df["max_up_move_pips"].astype(float)
            max_down = long_df["max_down_move_pips"].astype(float)

            hit_tp = max_up >= tp_pips
            hit_sl = (-max_down) >= sl_pips

            only_tp = hit_tp & ~hit_sl
            only_sl = hit_sl & ~hit_tp
            neither = ~(hit_tp | hit_sl)

            tp_rate = only_tp.mean()
            sl_rate = only_sl.mean()
            be_rate = neither.mean()
        else:
            logger.warning(
                "⚠️ LONG trades için TP/SL hesaplamak için gerekli kolonlar eksik: %s",
                required_cols - set(long_df.columns),
            )
            tp_rate = sl_rate = be_rate = float("nan")

        logger.info(
            "   ✅ LONG trades directional win-rate (future_dir==UP): %.3f",
            dir_win_rate,
        )
        logger.info(
            "   ✅ LONG trades TP%%: %.3f  SL%%: %.3f  BE%%: %.3f",
            tp_rate,
            sl_rate,
            be_rate,
        )
    else:
        logger.warning("⚠️ final_action == 'LONG' olan hiç satır yok, threshold çok agresif olabilir.")

    return df


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
def save_output(df: pd.DataFrame) -> None:
    """Sonucu parquet + csv olarak kaydeder."""
    Path(OUTPUT_PARQUET).parent.mkdir(parents = True, exist_ok = True)

    df.to_parquet(OUTPUT_PARQUET, index = False)
    df.to_csv(OUTPUT_CSV, index = False)

    logger.info("💾 Kaydedildi (Parquet): %s", OUTPUT_PARQUET)
    logger.info("💾 Kaydedildi (CSV)    : %s", OUTPUT_CSV)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    sig_df, preds_df = load_data()
    merged = merge_signals_with_preds(sig_df, preds_df)
    final_df = apply_fake_live_logic(merged)
    save_output(final_df)

    logger.info("=" * 79)
    logger.info("✅ NASDAQ FAKE-LIVE SIGNAL PIPELINE v1 TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()
