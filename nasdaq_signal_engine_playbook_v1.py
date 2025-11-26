#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# ======================= CONFIG =======================
MASTER_PATH = "./staging/nasdaq_master_features_v1.parquet"
PLAYBOOK_JSON_PATH = "./models/setup_playbook_long_v2.json"
PLAYBOOK_JSON_V2 = "./models/setup_playbook_long_v2.json"
SIGNALS_OUT_PARQUET = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"
SIGNALS_OUT_CSV = "./staging/nasdaq_playbook_signals_from_master_v1.csv"

PRICE_COL = "Close_M30"   # entry price
TIMESTAMP_COL = "timestamp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./models/nasdaq_signal_engine_playbook_v1.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


# ======================= UTILS =======================

def load_playbook(path: str = PLAYBOOK_JSON_PATH) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Playbook JSON bulunamadı: {path}")

    logger.info("📥 Playbook JSON yükleniyor: %s", path)
    with open(p, "r") as f:
        playbook = json.load(f)

    logger.info("✅ Playbook entry sayısı: %d", len(playbook))
    some_key = next(iter(playbook))
    logger.info(
        "   🔎 Örnek setup: %s → TP=%s SL=%s RR=%.2f",
        some_key,
        playbook[some_key].get("suggested_tp_pips"),
        playbook[some_key].get("suggested_sl_pips"),
        playbook[some_key].get("rr", 0.0),
    )
    return playbook


def load_playbook_long_v2(only_live: bool = True) -> dict:
    """
    setup_playbook_long_v2.json dosyasını yükler.

    only_live=True ise is_live == True olan setupları filtreler.
    """
    p = Path(PLAYBOOK_JSON_V2)
    if not p.exists():
        raise FileNotFoundError(f"Playbook v2 JSON bulunamadı: {p}")

    logger.info("📥 Playbook v2 JSON yükleniyor: %s", p)
    data = json.loads(p.read_text(encoding="utf-8"))
    logger.info("   ✅ Toplam setup sayısı (v2): %d", len(data))

    if only_live:
        filtered = {
            k: v for k, v in data.items()
            if v.get("is_live", False)
        }
        logger.info(
            "   ✅ is_live=True filtresinden geçen setup sayısı: %d",
            len(filtered),
        )
        # Eğer filtre çok agresifse fallback yapalım:
        if len(filtered) == 0:
            logger.warning(
                "⚠ is_live filtresi sonrası hiç setup kalmadı, tüm setuplar kullanılacak!"
            )
            return data
        return filtered

    return data


def _safe_get(row: pd.Series, col: str, default=None):
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return default


# ======================= CORE ENGINE =======================

def generate_signals_for_row(
    row: pd.Series,
    playbook: Dict[str, Dict[str, Any]],
    price_col: str = PRICE_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> List[Dict[str, Any]]:
    """
    Tek bir bar için:
    - row'da hangi event_type kolonları varsa VE değeri 1 ise
    - playbook'tan TP/SL alarak sinyal objesi üretir
    - Aynı zamanda SR + wave + channel context kolonlarını ekler
    """

    signals = []

    if price_col not in row.index or pd.isna(row[price_col]):
        return signals

    entry_price = float(row[price_col])
    ts = row[timestamp_col] if timestamp_col in row.index else None

    # --- SR CONTEXT ---
    sr_support_price = _safe_get(row, "sr_support_price")
    sr_support_strength = _safe_get(row, "sr_support_strength")
    sr_support_dist = _safe_get(row, "sr_support_distance")

    sr_res_price = _safe_get(row, "sr_resistance_price")
    sr_res_strength = _safe_get(row, "sr_resistance_strength")
    sr_res_dist = _safe_get(row, "sr_resistance_distance")

    sr_near_support = _safe_get(row, "sr_near_support")
    sr_near_resistance = _safe_get(row, "sr_near_resistance")

    # en yakın SR (destek / direnç)
    nearest_sr_type = None
    nearest_sr_dist = None
    if sr_support_dist is not None or sr_res_dist is not None:
        candidates = []
        if sr_support_dist is not None:
            candidates.append(("SUPPORT", abs(float(sr_support_dist))))
        if sr_res_dist is not None:
            candidates.append(("RESISTANCE", abs(float(sr_res_dist))))
        if candidates:
            nearest_sr_type, nearest_sr_dist = sorted(
                candidates, key=lambda x: x[1]
            )[0]

    # --- WAVE CONTEXT ---
    signal_wave = _safe_get(row, "signal_wave")
    wave_strength_pips = _safe_get(row, "wave_strength_pips")
    wave_duration_bars = _safe_get(row, "wave_duration_bars")
    up_move_pips = _safe_get(row, "up_move_pips")
    down_move_pips = _safe_get(row, "down_move_pips")
    up_duration_bars = _safe_get(row, "up_duration_bars")
    down_duration_bars = _safe_get(row, "down_duration_bars")

    # --- CHANNEL / RANGE CONTEXT ---
    chan_is_up = _safe_get(row, "chan_is_up_M30")
    chan_is_down = _safe_get(row, "chan_is_down_M30")
    is_range = _safe_get(row, "is_range_M30")
    near_lower_chan = _safe_get(row, "near_lower_chan_M30")
    near_upper_chan = _safe_get(row, "near_upper_chan_M30")

    for event_type, cfg in playbook.items():
        if event_type not in row.index:
            continue

        val = row[event_type]
        if pd.isna(val) or float(val) == 0.0:
            continue

        direction = cfg.get("direction", "LONG").upper()
        tp_pips = float(cfg.get("suggested_tp_pips", 0.0))
        sl_pips = float(cfg.get("suggested_sl_pips", 0.0))
        rr = float(cfg.get("rr", 0.0))

        if direction == "LONG":
            tp_price = entry_price + tp_pips
            sl_price = entry_price - sl_pips
        else:
            tp_price = entry_price - tp_pips
            sl_price = entry_price + sl_pips

        signal = {
            # temel
            "timestamp": ts,
            "event_type": event_type,
            "direction": direction,
            "entry_price": entry_price,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "rr": rr,
            # setup istatistikleri
            "n_events": int(cfg.get("n_events", 0)),
            "pct_up": float(cfg.get("pct_up", 0.0)),
            "pct_down": float(cfg.get("pct_down", 0.0)),
            "up_bias": float(cfg.get("up_bias", 0.0)),
            "avg_max_up_pips": float(cfg.get("avg_max_up_pips", 0.0)),
            "avg_max_down_pips": float(cfg.get("avg_max_down_pips", 0.0)),
            # SR context
            "sr_support_price_at_entry": float(sr_support_price) if sr_support_price is not None else None,
            "sr_support_strength_at_entry": float(sr_support_strength) if sr_support_strength is not None else None,
            "sr_support_distance_pips_at_entry": float(sr_support_dist) if sr_support_dist is not None else None,
            "sr_resistance_price_at_entry": float(sr_res_price) if sr_res_price is not None else None,
            "sr_resistance_strength_at_entry": float(sr_res_strength) if sr_res_strength is not None else None,
            "sr_resistance_distance_pips_at_entry": float(sr_res_dist) if sr_res_dist is not None else None,
            "sr_near_support_at_entry": int(sr_near_support) if sr_near_support is not None else None,
            "sr_near_resistance_at_entry": int(sr_near_resistance) if sr_near_resistance is not None else None,
            "nearest_sr_type": nearest_sr_type,
            "nearest_sr_dist_pips": float(nearest_sr_dist) if nearest_sr_dist is not None else None,
            # wave context
            "signal_wave_at_entry": int(signal_wave) if signal_wave is not None else None,
            "wave_strength_pips_at_entry": float(wave_strength_pips) if wave_strength_pips is not None else None,
            "wave_duration_bars_at_entry": float(wave_duration_bars) if wave_duration_bars is not None else None,
            "up_move_pips_at_entry": float(up_move_pips) if up_move_pips is not None else None,
            "down_move_pips_at_entry": float(down_move_pips) if down_move_pips is not None else None,
            "up_duration_bars_at_entry": float(up_duration_bars) if up_duration_bars is not None else None,
            "down_duration_bars_at_entry": float(down_duration_bars) if down_duration_bars is not None else None,
            # channel / range context
            "chan_is_up_M30_at_entry": int(chan_is_up) if chan_is_up is not None else None,
            "chan_is_down_M30_at_entry": int(chan_is_down) if chan_is_down is not None else None,
            "is_range_M30_at_entry": int(is_range) if is_range is not None else None,
            "near_lower_chan_M30_at_entry": int(near_lower_chan) if near_lower_chan is not None else None,
            "near_upper_chan_M30_at_entry": int(near_upper_chan) if near_upper_chan is not None else None,
        }

        signals.append(signal)

    return signals


# ======================= BACKTEST HELPER =======================

def build_signals_from_master(
    master_path: str = MASTER_PATH,
    playbook_path: str = PLAYBOOK_JSON_PATH,
    price_col: str = PRICE_COL,
    timestamp_col: str = TIMESTAMP_COL,
    out_parquet: str = SIGNALS_OUT_PARQUET,
    out_csv: str = SIGNALS_OUT_CSV,
    use_v2: bool = True,
):
    logger.info("📥 MASTER yükleniyor: %s", master_path)
    df = pd.read_parquet(master_path)
    logger.info("✅ MASTER shape: %s", df.shape)

    if use_v2:
        playbook = load_playbook_long_v2(only_live=True)
    else:
        playbook = load_playbook(playbook_path)

    all_signals = []
    total_rows = len(df)
    log_every = max(1, total_rows // 20)

    logger.info("🚀 MASTER üzerinde sinyal üretimi başlıyor...")
    for i, (_, row) in enumerate(df.iterrows()):
        if i % log_every == 0:
            logger.info("   → Progress: %d / %d (%.1f%%)", i, total_rows, 100.0 * i / total_rows)

        sigs = generate_signals_for_row(
            row,
            playbook=playbook,
            price_col=price_col,
            timestamp_col=timestamp_col,
        )
        if sigs:
            all_signals.extend(sigs)

    logger.info("✅ Toplam sinyal sayısı: %d", len(all_signals))

    if not all_signals:
        logger.warning("⚠️ Hiç sinyal üretilmedi. Event kolonları master'da yok olabilir.")
        sig_df = pd.DataFrame()
    else:
        sig_df = pd.DataFrame(all_signals)

    out_dir = Path(out_parquet).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_df.to_parquet(out_parquet, index=False)
    sig_df.to_csv(out_csv, index=False)

    logger.info("💾 Sinyal dataset (Parquet): %s", out_parquet)
    logger.info("💾 Sinyal dataset (CSV): %s", out_csv)
    if not sig_df.empty:
        logger.info("🎯 İlk 5 sinyal:\n%s", sig_df.head().to_string())


def main():
    logger.info("=" * 80)
    logger.info("🚀 NASDAQ SIGNAL ENGINE (PLAYBOOK v2 + CONTEXT) BAŞLIYOR")
    logger.info("=" * 80)

    # Eskiden: playbook_v1 = load_playbook_long_v1()
    playbook_cfg = load_playbook_long_v2(only_live=True)

    logger.info("✅ Aktif playbook setup sayısı: %d", len(playbook_cfg))

    try:
        build_signals_from_master(use_v2=True)
        logger.info("=" * 80)
        logger.info("✅ NASDAQ SIGNAL ENGINE (PLAYBOOK v2 + CONTEXT) TAMAMLANDI")
        logger.info("=" * 80)
    except Exception as e:
        logger.exception("❌ HATA: %s", e)


if __name__ == "__main__":
    main()