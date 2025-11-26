import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Paths
PLAYBOOK_SIGNALS_PATH = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"
EVENT_PREDS_PATH = "./staging/nasdaq_event_outcomes_with_preds_v2.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_signals_with_model_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_fake_live_signals_with_model_v1.csv"

# -----------------------------------------------------------------------------
# Tunable scoring parameters (keep simple + auditable)
# -----------------------------------------------------------------------------
# Golden rule gate still requires p_up dominance. Scores only modulate pass/keep.

# Base score starts from the model's edge (p_up - max(p_down, p_chop)).
MODEL_EDGE_WEIGHT = 1.0

# Support proximity & robustness (favour longs leaning on solid support)
MIN_SR_STRENGTH = 2.0          # minimum support strength considered helpful
MAX_SR_DISTANCE_PIPS = 50.0    # acceptable distance to support/nearest SUPPORT
MIN_SR_REACTIONS = 4           # minimum reaction count (if such a column exists)
SUPPORT_NEAR_BONUS = 0.6       # bonus when inside support band and strong enough
SUPPORT_STRENGTH_WEIGHT = 0.15 # extra per strength point above MIN_SR_STRENGTH
SUPPORT_REACTION_BONUS = 0.2   # bonus when reaction count crosses MIN_SR_REACTIONS

# Resistance / upper-channel penalties (avoid chasing into overhead supply)
RESISTANCE_DISTANCE_PIPS = 35.0
RESISTANCE_PENALTY = -0.7
UPPER_CHANNEL_PENALTY = -0.4
DOWN_TREND_PENALTY = -0.4

# Channel context bonuses (trend/range near lower band)
CHANNEL_UP_BONUS = 0.5
CHANNEL_RANGE_LOWER_BONUS = 0.35
CHANNEL_NEAR_LOWER_BONUS = 0.25

# Wave/leg context (prefer early/constructive waves)
ALLOWED_LONG_WAVES: set[int] = {1, 2, 3}
GOOD_WAVE_BONUS = 0.2
OTHER_WAVE_PENALTY = -0.15

# Final decision threshold on accumulated score (after golden gate)
CONTEXT_SCORE_THRESHOLD = 0.3

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_live_signal_pipeline")


# -----------------------------------------------------------------------------
# Column documentation for quick reference (playbook context)
# -----------------------------------------------------------------------------
# Support/Resistance columns (at entry):
#   - sr_support_price_at_entry, sr_support_strength_at_entry, sr_support_distance_pips_at_entry
#   - sr_resistance_price_at_entry, sr_resistance_strength_at_entry, sr_resistance_distance_pips_at_entry
#   - sr_near_support_at_entry, sr_near_resistance_at_entry
#   - nearest_sr_type, nearest_sr_dist_pips
#   - optional reaction-count columns (e.g., sr_support_reaction_count_at_entry, sr_support_num_reactions_at_entry)
# Context scoring column:
#   - context_score (computed here: model edge + SR + channel + wave - resistance penalties)
# Channel/regime columns (M30):
#   - chan_is_up_M30_at_entry, chan_is_down_M30_at_entry, is_range_M30_at_entry
#   - near_lower_chan_M30_at_entry, near_upper_chan_M30_at_entry
# Wave/leg columns:
#   - signal_wave_at_entry, wave_strength_pips_at_entry, wave_duration_bars_at_entry
#   - up_move_pips_at_entry, down_move_pips_at_entry, up_duration_bars_at_entry, down_duration_bars_at_entry


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_future_dir_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    future_dir_label yoksa, future_dir'dan türet.
    - Eğer future_dir numeric ise: 0=CHOP, 1=DOWN, 2=UP (v2 label encoder gibi)
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
    - Diğer kolonlar (future_dir, tp_sl_result, max_up_move_pips vs.) sadece log/backtest için.
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
# Structural reasoning helpers (LONG context)
# -----------------------------------------------------------------------------
def _is_model_bias_long(row: pd.Series) -> tuple[bool, str]:
    """Baseline golden rule: p_up must dominate p_down and p_chop."""

    if pd.isna(row.get("pred_label")):
        return False, "no_pred"

    p_up = row.get("p_up", np.nan)
    p_down = row.get("p_down", np.nan)
    p_chop = row.get("p_chop", np.nan)

    if pd.isna(p_up) or pd.isna(p_down) or pd.isna(p_chop):
        return False, "missing_probs"

    if (p_up >= p_down) and (p_up >= p_chop):
        return True, "model_bias_long"
    return False, "model_bias_not_long"


def _first_available_reaction_count(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    """Return the first non-null reaction count from candidate columns, if any."""

    for col in candidates:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return None


def _support_score(row: pd.Series) -> float:
    """Score support proximity/robustness.

    Positive weight when:
    - strength is at/above MIN_SR_STRENGTH
    - price is within MAX_SR_DISTANCE_PIPS of a SUPPORT
    - optional reaction counts show multiple historical bounces
    """

    strength = row.get("sr_support_strength_at_entry")
    distance = row.get("sr_support_distance_pips_at_entry")
    near_flag = row.get("sr_near_support_at_entry")
    nearest_type = str(row.get("nearest_sr_type", "")).upper()
    nearest_dist = row.get("nearest_sr_dist_pips")

    # Pull any available reaction count (different pipelines expose different names)
    reaction_cols = [
        "sr_support_reaction_count_at_entry",
        "sr_support_num_reactions_at_entry",
        "sr_support_reaction_count_band_at_entry",
    ]
    reactions = _first_available_reaction_count(row, reaction_cols)

    score = 0.0

    # Strength contribution (scaled above threshold)
    if pd.notna(strength) and float(strength) >= MIN_SR_STRENGTH:
        score += SUPPORT_NEAR_BONUS  # base bonus when strength passes minimum
        score += (float(strength) - MIN_SR_STRENGTH) * SUPPORT_STRENGTH_WEIGHT

    # Distance contribution (prefer closer to support)
    dist_candidates = []
    if pd.notna(distance):
        dist_candidates.append(float(distance))
    if nearest_type == "SUPPORT" and pd.notna(nearest_dist):
        dist_candidates.append(float(nearest_dist))

    if dist_candidates and min(dist_candidates) <= MAX_SR_DISTANCE_PIPS:
        score += SUPPORT_NEAR_BONUS

    # Reaction contribution (multiple historical reactions within band)
    if reactions is not None and reactions >= MIN_SR_REACTIONS:
        score += SUPPORT_REACTION_BONUS

    # Explicit "near support" flag is an extra nod toward leaning on structure
    if pd.notna(near_flag) and int(near_flag) == 1:
        score += 0.15

    return score


def _resistance_penalty(row: pd.Series) -> float:
    """Penalty when price is too close to resistance / upper band for longs."""

    penalties = 0.0

    res_dist = row.get("sr_resistance_distance_pips_at_entry")
    near_res_flag = row.get("sr_near_resistance_at_entry")
    nearest_type = str(row.get("nearest_sr_type", "")).upper()
    nearest_dist = row.get("nearest_sr_dist_pips")

    dist_candidates = []
    if pd.notna(res_dist):
        dist_candidates.append(float(res_dist))
    if nearest_type == "RESISTANCE" and pd.notna(nearest_dist):
        dist_candidates.append(float(nearest_dist))

    if dist_candidates and min(dist_candidates) <= RESISTANCE_DISTANCE_PIPS:
        penalties += RESISTANCE_PENALTY

    if pd.notna(near_res_flag) and int(near_res_flag) == 1:
        penalties += 0.5 * RESISTANCE_PENALTY

    # Channel context: near upper band is risky for longs
    near_upper = row.get("near_upper_chan_M30_at_entry")
    if pd.notna(near_upper) and int(near_upper) == 1:
        penalties += UPPER_CHANNEL_PENALTY

    return penalties


def _channel_score(row: pd.Series) -> float:
    """Score M30 channel context.

    - Reward uptrends.
    - Reward ranges when we are near the lower boundary.
    - Small bonus for simply being near the lower channel (mean-reversion edge).
    - Penalize clear downtrend a bit for longs.
    """

    up_flag = row.get("chan_is_up_M30_at_entry")
    down_flag = row.get("chan_is_down_M30_at_entry")
    range_flag = row.get("is_range_M30_at_entry")
    near_lower = row.get("near_lower_chan_M30_at_entry")

    score = 0.0

    if pd.notna(up_flag) and int(up_flag) == 1:
        score += CHANNEL_UP_BONUS

    if pd.notna(range_flag) and int(range_flag) == 1 and pd.notna(near_lower) and int(near_lower) == 1:
        score += CHANNEL_RANGE_LOWER_BONUS

    if pd.notna(near_lower) and int(near_lower) == 1:
        score += CHANNEL_NEAR_LOWER_BONUS

    if pd.notna(down_flag) and int(down_flag) == 1:
        score += DOWN_TREND_PENALTY

    return score


def _wave_score(row: pd.Series) -> float:
    """Score pattern/wave leg placement. Prefer early constructive waves for longs."""

    wave_val = row.get("signal_wave_at_entry")
    if pd.isna(wave_val):
        return 0.0

    try:
        wave_int = int(wave_val)
    except (TypeError, ValueError):
        return 0.0

    if wave_int in ALLOWED_LONG_WAVES:
        return GOOD_WAVE_BONUS
    return OTHER_WAVE_PENALTY


def _context_score(row: pd.Series) -> float:
    """Combine model edge + structure into a simple additive score.

    The score is deterministic and auditable:
      - start from model edge (p_up - max(p_down, p_chop)) * MODEL_EDGE_WEIGHT
      - add support, channel, and wave bonuses
      - subtract resistance / upper-channel penalties
    Final action goes LONG when score >= CONTEXT_SCORE_THRESHOLD.
    """

    p_up = float(row.get("p_up", 0))
    p_down = float(row.get("p_down", 0))
    p_chop = float(row.get("p_chop", 0))
    model_edge = p_up - max(p_down, p_chop)

    score = MODEL_EDGE_WEIGHT * model_edge
    score += _support_score(row)
    score += _channel_score(row)
    score += _wave_score(row)
    score += _resistance_penalty(row)

    return score


# -----------------------------------------------------------------------------
# Decision Logic
# -----------------------------------------------------------------------------
def apply_fake_live_logic(merged: pd.DataFrame) -> pd.DataFrame:
    """
    LONG fake-live karar mantığı (golden gate + additive context score):

      - Model tahmini yoksa → NO_PRED
      - direction LONG/BUY değilse → PASS
      - p_up en büyük değilse → PASS
      - Score = model_edge + support + channel + wave - resistance penalties
      - Score >= CONTEXT_SCORE_THRESHOLD → LONG, aksi halde PASS
    """
    logger.info("🧠 Fake-live karar mantığı uygulanıyor...")
    logger.info(
        "   📊 SR threshold/weights: MIN_SR_STRENGTH=%.2f MAX_SR_DISTANCE_PIPS=%.1f MIN_SR_REACTIONS=%d",
        MIN_SR_STRENGTH,
        MAX_SR_DISTANCE_PIPS,
        MIN_SR_REACTIONS,
    )
    logger.info(
        "   📊 Channel bonuses: up=%.2f range_lower=%.2f near_lower=%.2f | penalties down=%.2f upper=%.2f",
        CHANNEL_UP_BONUS,
        CHANNEL_RANGE_LOWER_BONUS,
        CHANNEL_NEAR_LOWER_BONUS,
        DOWN_TREND_PENALTY,
        UPPER_CHANNEL_PENALTY,
    )
    logger.info(
        "   📊 Decision threshold: CONTEXT_SCORE_THRESHOLD=%.2f (model_edge weight=%.2f)",
        CONTEXT_SCORE_THRESHOLD,
        MODEL_EDGE_WEIGHT,
    )

    df = merged.sort_values("timestamp").reset_index(drop=True).copy()

    def decide_row(row) -> str:
        # Model tahmini yoksa → NO_PRED
        if pd.isna(row.get("pred_label")):
            return "NO_PRED"

        # Bu dosya LONG playbook setupları için; direction farklıysa trade alma
        if row.get("direction") not in ("LONG", "BUY", None):
            return "PASS"

        # Golden rule: p_up dominates and all probs present
        model_ok, _ = _is_model_bias_long(row)
        if not model_ok:
            return "PASS"

        # Additive structural score (favour support/uptrend, avoid resistance/upper band)
        score = _context_score(row)

        if score >= CONTEXT_SCORE_THRESHOLD:
            return "LONG"
        return "PASS"

    df["context_score"] = df.apply(_context_score, axis=1)
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

    if "context_score" in df.columns:
        mean_long = df.loc[df["final_action"] == "LONG", "context_score"].mean()
        mean_pass = df.loc[df["final_action"] == "PASS", "context_score"].mean()
        logger.info(
            "   • Ortalama context_score → LONG: %.3f | PASS: %.3f (threshold=%.2f)",
            mean_long,
            mean_pass,
            CONTEXT_SCORE_THRESHOLD,
        )

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
    final_df = apply_fake_live_logic(merged)
    save_output(final_df)

    logger.info("=" * 79)
    logger.info("✅ NASDAQ FAKE-LIVE SIGNAL PIPELINE v1 TAMAMLANDI")
    logger.info("=" * 79)


if __name__ == "__main__":
    main()
