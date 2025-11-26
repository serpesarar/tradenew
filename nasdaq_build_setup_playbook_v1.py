#!/usr/bin/env python3
import pandas as pd
import json
import logging
from pathlib import Path

# ======================= CONFIG =======================
TOP_LONG_CSV_PATH = "./models/top_long_setups_v1.csv"
PLAYBOOK_JSON_PATH = "./models/setup_playbook_long_v1.json"
PLAYBOOK_TXT_PATH = "./models/setup_playbook_long_v1.txt"

# TP/SL heuristikleri (istersen sonra oynarız)
TP_FACTOR = 0.5   # TP ≈ avg_max_up_pips * 0.5
SL_FACTOR = 0.7   # SL ≈ |avg_max_down_pips| * 0.7
MIN_SL_PIPS = 20  # çok saçma küçük SL'leri engelle
MIN_TP_PIPS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./models/setup_playbook_long_v1.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


def load_top_long(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Top long setups CSV bulunamadı: {path}")

    logger.info("📥 Top long setups CSV yükleniyor: %s", path)
    df = pd.read_csv(path)

    required = [
        "event_type",
        "n_events",
        "pct_up",
        "pct_down",
        "up_bias",
        "avg_max_up_pips",
        "avg_max_down_pips",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}")

    logger.info("✅ Top long shape: %s", df.shape)
    return df


def build_playbook(df: pd.DataFrame):
    """Her event_type için LONG setup konfigi üret."""
    playbook = {}

    for _, row in df.iterrows():
        event_type = str(row["event_type"])

        pct_up = float(row["pct_up"])
        pct_down = float(row["pct_down"])
        up_bias = float(row.get("up_bias", pct_up - pct_down))

        avg_up = float(row["avg_max_up_pips"])
        avg_down = float(row["avg_max_down_pips"])  # negatif

        n_events = int(row["n_events"])

        # TP / SL hesapla
        raw_tp = avg_up * TP_FACTOR
        raw_sl = abs(avg_down) * SL_FACTOR

        tp_pips = max(round(raw_tp, 1), MIN_TP_PIPS)
        sl_pips = max(round(raw_sl, 1), MIN_SL_PIPS)

        rr = round(tp_pips / sl_pips, 2) if sl_pips > 0 else None

        cfg = {
            "event_type": event_type,
            "direction": "LONG",          # Bu listede hepsi long-bias
            "n_events": n_events,
            "pct_up": round(pct_up, 4),
            "pct_down": round(pct_down, 4),
            "up_bias": round(up_bias, 4),
            "avg_max_up_pips": round(avg_up, 2),
            "avg_max_down_pips": round(avg_down, 2),
            "suggested_tp_pips": tp_pips,
            "suggested_sl_pips": sl_pips,
            "rr": rr,
        }

        playbook[event_type] = cfg

    logger.info("✅ Playbook entry sayısı: %d", len(playbook))
    return playbook


def save_playbook(playbook: dict):
    Path("./models").mkdir(exist_ok=True)

    # JSON
    with open(PLAYBOOK_JSON_PATH, "w") as f:
        json.dump(playbook, f, indent=2)
    logger.info("💾 Playbook JSON kaydedildi: %s", PLAYBOOK_JSON_PATH)

    # İnsan okunabilir TXT
    with open(PLAYBOOK_TXT_PATH, "w") as f:
        f.write("SETUP PLAYBOOK LONG v1\n")
        f.write("=" * 80 + "\n\n")
        for et, cfg in playbook.items():
            f.write(f"EVENT TYPE: {et}\n")
            f.write(f"  Direction     : {cfg['direction']}\n")
            f.write(f"  n_events      : {cfg['n_events']}\n")
            f.write(f"  pct_up        : {cfg['pct_up']:.4f}\n")
            f.write(f"  pct_down      : {cfg['pct_down']:.4f}\n")
            f.write(f"  up_bias       : {cfg['up_bias']:.4f}\n")
            f.write(f"  avg_up_pips   : {cfg['avg_max_up_pips']}\n")
            f.write(f"  avg_down_pips : {cfg['avg_max_down_pips']}\n")
            f.write(f"  TP (pips)     : {cfg['suggested_tp_pips']}\n")
            f.write(f"  SL (pips)     : {cfg['suggested_sl_pips']}\n")
            f.write(f"  RR            : {cfg['rr']}\n")
            f.write("-" * 80 + "\n")
    logger.info("💾 Playbook TXT kaydedildi: %s", PLAYBOOK_TXT_PATH)


def main():
    logger.info("=" * 80)
    logger.info("🚀 SETUP PLAYBOOK BUILDER v1 BAŞLIYOR")
    logger.info("=" * 80)

    try:
        df = load_top_long(TOP_LONG_CSV_PATH)
        playbook = build_playbook(df)
        save_playbook(playbook)

        logger.info("=" * 80)
        logger.info("✅ SETUP PLAYBOOK BUILDER v1 TAMAMLANDI")
        logger.info("=" * 80)
    except Exception as e:
        logger.exception("❌ HATA: %s", e)


if __name__ == "__main__":
    main()