import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PLAYBOOK_SHORT_SIGNALS_PATH = "./staging/nasdaq_playbook_short_signals_from_master_v1.parquet"
EVENT_PREDS_PATH = "./staging/nasdaq_event_outcomes_with_preds_v2.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_short_signals_with_model_v1.parquet"
OUTPUT_CSV = "./staging/nasdaq_fake_live_short_signals_with_model_v1.csv"

# =============================================================================
# SHORT FİLTRE PARAMETRELERİ - Teknik analiz + model tahminleri
# =============================================================================
HIGH_CONF_THRESHOLD = 0.57       # max_prob minimum eşik
P_DOWN_MIN = 0.38                # p_down minimum değer
P_DOWN_MARGIN = 0.03             # p_down, p_up'tan en az bu kadar yüksek olmalı
P_CHOP_MAX = 0.60                # p_chop bu değerin üzerindeyse PASS (çok belirsiz)
# =============================================================================
# SHORT FİLTRE PARAMETRELERİ - Sadece temel filtreler (SHORT iyi çalışıyordu)
# =============================================================================
HIGH_CONF_THRESHOLD = 0.57       # max_prob minimum eşik
P_DOWN_MIN = 0.38                # p_down minimum değer
P_DOWN_MARGIN = 0.03             # p_down, p_up'tan en az bu kadar yüksek olmalı
P_CHOP_MAX = 0.60                # p_chop bu değerin üzerindeyse PASS (çok belirsiz)
REQUIRE_DOWN_TREND = True        # True ise sadece DOWN_TREND regime'de SHORT al

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_live_short_signal_pipeline")


def normalize_future_dir(value) -> str | float:
    """
    future_dir numeric değerini string label'a çevirir.
    Dataset'te 0/1/2 bazen CHOP/ DOWN / UP olarak geliyor.
    """
    if isinstance(value, str):
        val = value.strip().upper()
        if val in {"UP", "DOWN", "CHOP"}:
            return val
    if pd.isna(value):
        return np.nan

    try:
        val = int(value)
    except (TypeError, ValueError):
        return np.nan

    mapping = {
        2: "UP",
        1: "DOWN",   # Dataset'te 1 genelde DOWN encode edilmiş
        0: "CHOP",
        -1: "DOWN",
    }
    return mapping.get(val, np.nan)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Short playbook sinyallerini ve model tahminlerini yükler."""
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


def merge_signals_with_preds(sig_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Short playbook sinyallerini, event outcome tahminleriyle merge eder.
    Join key: [timestamp, event_type, entry_price]
    """
    logger.info("🔗 Short playbook sinyalleri ile model tahminleri merge ediliyor...")

    # Sadece gerekli kolonları al
    keep_cols = [
        "timestamp",
        "event_type",
        "entry_price",
        "p_chop",
        "p_up",
        "p_down",
        "pred_class",
        "pred_label",
        "max_prob",
        "recommendation",
        "tp_sl_result",
        "future_dir",
        "max_up_move_pips",
        "max_down_move_pips",
    ]
    missing = [c for c in keep_cols if c not in preds_df.columns]
    if missing:
        raise ValueError(f"❌ Tahmin datasetinde eksik kolonlar var: {missing}")

    preds_small = preds_df[keep_cols].copy()

    # Tip uyumu (özellikle timestamp ve entry_price)
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
    )  # Eğer string label yoksa, numeric tp_sl_result'tan türet
    if "tp_sl_result_label" not in merged.columns and "tp_sl_result" in merged.columns:
        def map_tp_label(v):
            if pd.isna(v):
                return np.nan
            # v2 pipeline'da tp_sl_result genelde şu şekilde:
            #  >0  → TP
            #  <0  → SL
            #  ==0 → BE
            if v > 0:
                return "TP"
            if v < 0:
                return "SL"
            return "BE"

        merged["tp_sl_result_label"] = merged["tp_sl_result"].apply(map_tp_label)

    # outcome kolonlarını preds versiyonundan al (varsa *_preds)
    for col in ["future_dir", "tp_sl_result", "max_up_move_pips", "max_down_move_pips"]:
        preds_col = f"{col}_preds"
        if preds_col in merged.columns:
            merged[col] = merged[preds_col]

    if "future_dir_label" not in merged.columns and "future_dir" in merged.columns:
        merged["future_dir_label"] = merged["future_dir"].apply(normalize_future_dir)

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


def apply_fake_live_short_logic(merged: pd.DataFrame) -> pd.DataFrame:
    """
    SHORT için karar mantığı (sıkılaştırılmış v2):
      final_action:
        - SHORT : Tüm filtrelerden geçerse
        - PASS  : Diğer tüm durumlar
        - NO_PRED : Model tahmini yoksa
    
    Filtreler:
      1. Model tahmini olmalı
      2. p_down >= P_DOWN_MIN
      3. p_down - p_up >= P_DOWN_MARGIN (DOWN, UP'tan belirgin yüksek)
      4. p_chop <= P_CHOP_MAX (çok belirsiz olmamalı)
      5. max_prob >= HIGH_CONF_THRESHOLD
      6. (opsiyonel) Regime = DOWN_TREND
    """
    logger.info("🧠 Fake-live SHORT karar mantığı uygulanıyor (sıkılaştırılmış v2)...")
    logger.info("   📊 Filtre parametreleri:")
    logger.info("      • HIGH_CONF_THRESHOLD : %.2f", HIGH_CONF_THRESHOLD)
    logger.info("      • P_DOWN_MIN          : %.2f", P_DOWN_MIN)
    logger.info("      • P_DOWN_MARGIN       : %.2f", P_DOWN_MARGIN)
    logger.info("      • P_CHOP_MAX          : %.2f", P_CHOP_MAX)
    logger.info("      • REQUIRE_DOWN_TREND  : %s", REQUIRE_DOWN_TREND)

    df = merged.sort_values("timestamp").reset_index(drop=True).copy()

    # Regime kolonunu hazırla (varsa)
    regime_col = None
    for col in ["regime_M30", "regime", "chan_is_down_M30_at_entry"]:
        if col in df.columns:
            regime_col = col
            break

    def decide_row(row) -> str:
        # 1) Model tahmini yoksa
        if pd.isna(row.get("pred_label")):
            return "NO_PRED"

        # 2) Direction kontrolü
        if row.get("direction") not in ("SHORT", "SELL", None):
            return "PASS"

        # Probability değerlerini al
        p_up = row.get("p_up", np.nan)
        p_down = row.get("p_down", np.nan)
        p_chop = row.get("p_chop", np.nan)
        max_prob = row.get("max_prob", 0.0)

        # 3) Probability değerleri eksikse PASS
        if pd.isna(p_up) or pd.isna(p_down) or pd.isna(p_chop):
            return "PASS"

        # 4) p_down minimum threshold
        if p_down < P_DOWN_MIN:
            return "PASS"

        # 5) p_down, p_up'tan belirgin yüksek olmalı
        if (p_down - p_up) < P_DOWN_MARGIN:
            return "PASS"

        # 6) p_chop çok yüksekse (belirsizlik) PASS
        if p_chop > P_CHOP_MAX:
            return "PASS"

        # 7) max_prob (confidence) kontrolü
        if max_prob < HIGH_CONF_THRESHOLD:
            return "PASS"

        # 8) (Opsiyonel) Regime filtresi
        if REQUIRE_DOWN_TREND and regime_col:
            regime_val = row.get(regime_col)
            # DOWN_TREND veya chan_is_down_M30_at_entry == 1
            if regime_col == "regime_M30" or regime_col == "regime":
                if regime_val != "DOWN_TREND":
                    return "PASS"
            elif regime_col == "chan_is_down_M30_at_entry":
                if regime_val != 1:
                    return "PASS"

        # Tüm filtrelerden geçti → SHORT aç
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

    # Performans (sadece SHORT alınanlar için)
    mask_short = df["final_action"] == "SHORT"
    if mask_short.any():
        short_df = df[mask_short].copy()

        # 1) Yön başarısı (DOWN bekliyoruz)
        if "future_dir_label" in short_df.columns:
            dir_col = short_df["future_dir_label"].astype(str).str.upper()
        else:
            dir_col = short_df["future_dir"].apply(normalize_future_dir)
        dir_win_rate = (dir_col == "DOWN").mean()

        # 2) TP/SL – SHORT için pip bazlı hesap
        tp_rate = sl_rate = be_rate = float("nan")
        if "tp_pips" in short_df.columns and "sl_pips" in short_df.columns:
            tp_pips = short_df["tp_pips"].astype(float)
            sl_pips = short_df["sl_pips"].astype(float)
            max_up = short_df["max_up_move_pips"].astype(float)
            max_down = short_df["max_down_move_pips"].astype(float)

            # SHORT pozisyon için:
            # - max_down negatif gelir (aşağı gitmiş = kâr), mutlak değeri tp_pips'ten büyükse TP
            # - max_up pozitif gelir (yukarı gitmiş = zarar), değeri sl_pips'ten büyükse SL
            # ÖNEMLİ: Önce TP kontrolü, TP yoksa SL kontrolü (bir trade hem TP hem SL'ye ulaşamaz)
            
            # TP kontrolü: max_down negatif olduğu için -max_down = mutlak değer
            hit_tp = (-max_down >= tp_pips)
            
            # SL kontrolü: TP'ye ulaşmayanlarda kontrol et
            hit_sl = (~hit_tp) & (max_up >= sl_pips)
            
            # BE: Ne TP ne SL'ye ulaşanlar
            be_mask = ~(hit_tp | hit_sl)

            n = len(short_df)
            if n > 0:
                tp_rate = hit_tp.mean()
                sl_rate = hit_sl.mean()
                be_rate = be_mask.mean()

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


def save_output(df: pd.DataFrame) -> None:
    """Short sonucu parquet + csv olarak kaydeder."""
    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("💾 Kaydedildi (Parquet): %s", OUTPUT_PARQUET)
    logger.info("💾 Kaydedildi (CSV)    : %s", OUTPUT_CSV)


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