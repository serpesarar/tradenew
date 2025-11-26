import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Paths
PLAYBOOK_SHORT_SIGNALS_PATH = "./staging/nasdaq_playbook_short_signals_from_master_v1.parquet"
EVENT_PREDS_PATH = "./staging/nasdaq_event_outcomes_with_preds_v2.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_short_signals_with_model_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_fake_live_short_signals_with_model_v1.csv"

# -----------------------------------------------------------------------------
# Tunable structural filters (mirrors LONG side, inverted for shorts)
# -----------------------------------------------------------------------------
USE_PATTERN_WAVE_FILTER = True
ALLOWED_SHORT_WAVES: set[int] = {1, 2, 3}

USE_CHANNEL_FILTER = True

USE_SR_FILTER = True
MIN_SR_STRENGTH = 2.0  # sr_resistance_strength_at_entry minimum
MAX_SR_DISTANCE_PIPS = 50.0
MIN_SR_REACTIONS = 4

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_live_short_signal_pipeline")


# -----------------------------------------------------------------------------
# Column documentation (playbook context)
# -----------------------------------------------------------------------------
# Support/Resistance columns (at entry):
#   - sr_support_price_at_entry, sr_support_strength_at_entry, sr_support_distance_pips_at_entry
#   - sr_resistance_price_at_entry, sr_resistance_strength_at_entry, sr_resistance_distance_pips_at_entry
#   - sr_near_support_at_entry, sr_near_resistance_at_entry
#   - nearest_sr_type, nearest_sr_dist_pips
#   - optional reaction-count columns (e.g., sr_resistance_reaction_count_at_entry)
# Channel/regime columns (M30):
#   - chan_is_up_M30_at_entry, chan_is_down_M30_at_entry, is_range_M30_at_entry
#   - near_lower_chan_M30_at_entry, near_upper_chan_M30_at_entry
# Wave/leg columns:
#   - signal_wave_at_entry, wave_strength_pips_at_entry, wave_duration_bars_at_entry
#   - up_move_pips_at_entry, down_move_pips_at_entry, up_duration_bars_at_entry, down_duration_bars_at_entry


# -----------------------------------------------------------------------------
# Helpers shared with LONG side
# -----------------------------------------------------------------------------
def _ensure_future_dir_label(df: pd.DataFrame) -> pd.DataFrame:
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
        df["future_dir_label"] = col.astype(str).str.upper()

    return df


def _ensure_tp_sl_result_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tp_sl_result_label" in df.columns:
        return df

    if "tp_sl_result" not in df.columns:
        df["tp_sl_result_label"] = np.nan
        return df

    raw = df["tp_sl_result"]
    num = pd.to_numeric(raw, errors="coerce")

    if num.notna().any():
        def _map_num(v):
            if pd.isna(v):
                return np.nan
            if v > 0:
                return "TP"
            if v < 0:
                return "SL"
            return "BE"

        mapped_from_num = num.map(_map_num)
        raw_str = raw.astype("string").str.upper()
        res = mapped_from_num.where(num.notna(), raw_str)
    else:
        res = raw.astype("string").str.upper().fillna("UNKNOWN")

    df["tp_sl_result_label"] = res
    return df


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 79)
    logger.info("🚀 NASDAQ FAKE-LIVE SHORT SIGNAL PIPELINE v1 BAŞLIYOR")
    logger.info("=" * 79)

    logger.info("📥 Short playbook sinyalleri yükleniyor: %s", PLAYBOOK_SHORT_SIGNALS_PATH)
    sig_df = pd.read_parquet(PLAYBOOK_SHORT_SIGNALS_PATH)
    logger.info("   ✅ Short sinyal df shape: %s", sig_df.shape)

    logger.info("📥 Model tahminleri yükleniyor: %s", EVENT_PREDS_PATH)
    preds_df = pd.read_parquet(EVENT_PREDS_PATH)
    logger.info("   ✅ Tahmin df shape: %s", preds_df.shape)

    return sig_df, preds_df


# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
def merge_signals_with_preds(sig_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """Short playbook sinyallerini, event outcome tahminleriyle merge eder.

    Join key: [timestamp, event_type, entry_price]
    """

    logger.info("🔗 Short playbook sinyalleri ile model tahminleri merge ediliyor...")

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

    missing = [c for c in required_cols if c not in preds_df.columns]
    if missing:
        raise ValueError(f"❌ Tahmin datasetinde eksik kolonlar var: {missing}")

    available_optional = [c for c in optional_cols if c in preds_df.columns]
    keep_cols = required_cols + available_optional
    preds_small = preds_df[keep_cols].copy()

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

    merged = _ensure_future_dir_label(merged)
    merged = _ensure_tp_sl_result_label(merged)

    missing_preds = merged["pred_label"].isna().sum()
    if missing_preds > 0:
        logger.warning(
            "⚠️ %d short sinyalde eşleşen model tahmini bulunamadı (pred_label is NaN)",
            missing_preds,
        )
    else:
        logger.info("   ✅ Tüm short sinyaller için model tahmini bulundu.")

    logger.info("   ✅ Merge sonrası shape: %s", merged.shape)
    return merged


# -----------------------------------------------------------------------------
# Structural filters (SHORT context)
# -----------------------------------------------------------------------------
def _is_model_bias_short(row: pd.Series) -> tuple[bool, str]:
    """Golden rule: p_down must dominate p_up and p_chop."""

    if pd.isna(row.get("pred_label")):
        return False, "no_pred"

    p_up = row.get("p_up", np.nan)
    p_down = row.get("p_down", np.nan)
    p_chop = row.get("p_chop", np.nan)

    if pd.isna(p_up) or pd.isna(p_down) or pd.isna(p_chop):
        return False, "missing_probs"

    if (p_down >= p_up) and (p_down >= p_chop):
        return True, "model_bias_short"
    return False, "model_bias_not_short"


def _passes_pattern_wave_filter(row: pd.Series) -> bool:
    if not USE_PATTERN_WAVE_FILTER:
        return True

    wave_val = row.get("signal_wave_at_entry")
    if pd.isna(wave_val):
        return True

    try:
        wave_int = int(wave_val)
    except (TypeError, ValueError):
        return True

    return wave_int in ALLOWED_SHORT_WAVES


def _passes_channel_filter_short(row: pd.Series) -> bool:
    """Channel/regime filter mirrored for SHORT entries."""

    if not USE_CHANNEL_FILTER:
        return True

    down_flag = row.get("chan_is_down_M30_at_entry")
    range_flag = row.get("is_range_M30_at_entry")
    near_upper = row.get("near_upper_chan_M30_at_entry")

    cond_down = pd.notna(down_flag) and int(down_flag) == 1
    cond_range_upper = pd.notna(range_flag) and int(range_flag) == 1 and pd.notna(near_upper) and int(near_upper) == 1
    cond_near_upper = pd.notna(near_upper) and int(near_upper) == 1

    if cond_down or cond_range_upper or cond_near_upper:
        return True
    if pd.isna(down_flag) and pd.isna(range_flag) and pd.isna(near_upper):
        return True
    return False


def _first_available_reaction_count(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None


def _passes_sr_filter_short(row: pd.Series) -> bool:
    if not USE_SR_FILTER:
        return True

    strength = row.get("sr_resistance_strength_at_entry")
    strength_ok = pd.isna(strength) or float(strength) >= MIN_SR_STRENGTH

    near_flag = row.get("sr_near_resistance_at_entry")
    nearest_type = str(row.get("nearest_sr_type", "")).upper()
    nearest_dist = row.get("nearest_sr_dist_pips")
    resistance_distance = row.get("sr_resistance_distance_pips_at_entry")

    dist_candidates = []
    if pd.notna(resistance_distance):
        dist_candidates.append(float(resistance_distance))
    if nearest_type == "RESISTANCE" and pd.notna(nearest_dist):
        dist_candidates.append(float(nearest_dist))

    distance_ok = True
    if dist_candidates:
        distance_ok = min(dist_candidates) <= MAX_SR_DISTANCE_PIPS

    near_ok = True
    if pd.notna(near_flag):
        near_ok = int(near_flag) == 1
    elif nearest_type:
        near_ok = nearest_type == "RESISTANCE"

    reaction_cols = [
        "sr_resistance_reaction_count_at_entry",
        "sr_resistance_num_reactions_at_entry",
        "sr_resistance_reaction_count_band_at_entry",
    ]
    reactions = _first_available_reaction_count(row, reaction_cols)
    reactions_ok = True if reactions is None else reactions >= MIN_SR_REACTIONS

    return strength_ok and distance_ok and near_ok and reactions_ok


# -----------------------------------------------------------------------------
# Decision Logic
# -----------------------------------------------------------------------------
def apply_fake_live_short_logic(merged: pd.DataFrame) -> pd.DataFrame:
    """
    SHORT fake-live karar mantığı (golden rule + yapısal filtreler):

      - Model tahmini yoksa → NO_PRED
      - direction SHORT/SELL değilse → PASS
      - p_down en büyük değilse → PASS
      - Yapısal filtreler (pattern wave, kanal, SR) devredeyse hepsinden geçerse → SHORT
      - Aksi halde → PASS
    """

    logger.info("🧠 Fake-live SHORT karar mantığı uygulanıyor...")
    logger.info(
        "   📊 Filtre parametreleri: pattern_filter=%s channel_filter=%s sr_filter=%s",
        USE_PATTERN_WAVE_FILTER,
        USE_CHANNEL_FILTER,
        USE_SR_FILTER,
    )
    logger.info(
        "   📊 SR eşikleri: MIN_SR_STRENGTH=%.2f MAX_SR_DISTANCE_PIPS=%.1f MIN_SR_REACTIONS=%d",
        MIN_SR_STRENGTH,
        MAX_SR_DISTANCE_PIPS,
        MIN_SR_REACTIONS,
    )

    df = merged.sort_values("timestamp").reset_index(drop=True).copy()

    def decide_row(row) -> str:
        if pd.isna(row.get("pred_label")):
            return "NO_PRED"

        if row.get("direction") not in ("SHORT", "SELL", None):
            return "PASS"

        model_ok, _ = _is_model_bias_short(row)
        if not model_ok:
            return "PASS"

        if not _passes_pattern_wave_filter(row):
            return "PASS"

        if not _passes_channel_filter_short(row):
            return "PASS"

        if not _passes_sr_filter_short(row):
            return "PASS"

        return "SHORT"

    df["final_action"] = df.apply(decide_row, axis=1)

    total = len(df)
    n_short = (df["final_action"] == "SHORT").sum()
    n_pass = (df["final_action"] == "PASS").sum()
    n_nopred = (df["final_action"] == "NO_PRED").sum()

    logger.info("📊 Fake-live SHORT karar dağılımı:")
    logger.info("   • Toplam satır: %d", total)
    logger.info("   • SHORT      : %d (%.2f%%)", n_short, 100 * n_short / total if total else 0)
    logger.info("   • PASS       : %d (%.2f%%)", n_pass, 100 * n_pass / total if total else 0)
    logger.info("   • NO_PRED    : %d (%.2f%%)", n_nopred, 100 * n_nopred / total if total else 0)

    mask_short = df["final_action"] == "SHORT"
    if mask_short.any():
        short_df = df[mask_short].copy()

        dir_col = short_df["future_dir_label"].astype(str).str.upper()
        dir_win_rate = (dir_col == "DOWN").mean()

        tp_rate = sl_rate = be_rate = float("nan")
        required_cols = {"tp_pips", "sl_pips", "max_up_move_pips", "max_down_move_pips"}
        if required_cols.issubset(short_df.columns):
            tp_pips = short_df["tp_pips"].astype(float)
            sl_pips = short_df["sl_pips"].astype(float)
            max_up = short_df["max_up_move_pips"].astype(float)
            max_down = short_df["max_down_move_pips"].astype(float)

            hit_tp = (-max_down) >= tp_pips
            hit_sl = (~hit_tp) & (max_up >= sl_pips)
            be_mask = ~(hit_tp | hit_sl)

            tp_rate = hit_tp.mean()
            sl_rate = hit_sl.mean()
            be_rate = be_mask.mean()
        else:
            logger.warning(
                "⚠️ SHORT trades için TP/SL hesaplamak için gerekli kolonlar eksik: %s",
                required_cols - set(short_df.columns),
            )

        logger.info(
            "   ✅ SHORT trades directional win-rate (future_dir==DOWN): %.3f",
            dir_win_rate,
        )
        logger.info(
            "   ✅ SHORT trades TP%%: %.3f  SL%%: %.3f  BE%%: %.3f",
            tp_rate,
            sl_rate,
            be_rate,
        )
    else:
        logger.warning("⚠️ final_action == 'SHORT' olan hiç satır yok.")

    return df


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
def save_output(df: pd.DataFrame) -> None:
    """Short sonucu parquet + csv olarak kaydeder."""

    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("💾 Kaydedildi (Parquet): %s", OUTPUT_PARQUET)
    logger.info("💾 Kaydedildi (CSV)    : %s", OUTPUT_CSV)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    sig_df, preds_df = load_data()
    merged = merge_signals_with_preds(sig_df, preds_df)
    final_df = apply_fake_live_short_logic(merged)
    save_output(final_df)

    logger.info("=" * 79)
    logger.info("✅ NASDAQ FAKE-LIVE SHORT SIGNAL PIPELINE v1 TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()
