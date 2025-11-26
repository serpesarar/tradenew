# nasdaq_build_setup_playbook_short_v1.py

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN_STATS_CSV = "./models/event_pattern_stats_all_v1.csv"
OUTPUT_JSON = "./models/setup_playbook_short_v1.json"
OUTPUT_REPORT = "./models/setup_playbook_short_v1_report.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_playbook_short_v1")


def build_short_playbook(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Event pattern stats'ten SHORT setup playbook'u üretir.
    down_bias / short_score yoksa burada hesaplıyoruz.
    """

    # 🔹 Sadece gerçekten var olan, temel kolonları zorunlu tutalım
    needed_cols = [
        "event_type",
        "n_events",
        "pct_up",
        "pct_down",
        "avg_max_up_pips",
        "avg_max_down_pips",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Pattern stats CSV'de eksik kolonlar var: {missing}")

    logger.info("📊 Toplam pattern sayısı: %d", len(df))

    df = df.copy()

    # 🔹 Eğer yoksa down_bias = pct_down - pct_up
    if "down_bias" not in df.columns:
        df["down_bias"] = df["pct_down"] - df["pct_up"]

    # 🔹 Eğer yoksa short_score'u basit bir metrike bağlayalım
    # Örnek heuristic: (pct_down - pct_up)*100
    # İleride istersen daha sofistike yaparız.
    if "short_score" not in df.columns:
        df["short_score"] = (df["pct_down"] - df["pct_up"]) * 100

    # Filtre: short için mantıklı candidate'lar
    mask = (
        (df["n_events"] >= 400)      # yeterli sayıda örnek olsun
        & (df["pct_down"] >= 0.48)   # %48 ve üzeri down
        & (df["down_bias"] > 0.0)    # down, up'tan az da olsa yüksek olsun
    )

    filtered = df[mask].copy()
    logger.info("✅ Short candidate pattern sayısı (filter sonrası): %d", len(filtered))

    if filtered.empty:
        logger.warning("⚠️ Filter sonrası hiç short pattern kalmadı, eşikleri gevşetmen gerekebilir.")

    # Skora göre sırala (en yüksek short_score en üstte)
    filtered = filtered.sort_values("short_score", ascending=False)

    playbook: dict[str, dict] = {}

    for _, row in filtered.iterrows():
        event_type = str(row["event_type"])

        n_events = int(row["n_events"])
        pct_down = float(row["pct_down"])
        pct_up = float(row["pct_up"])
        down_bias = float(row["down_bias"])
        short_score = float(row["short_score"])

        avg_max_up = float(row["avg_max_up_pips"])      # aleyhimize (yukarı)
        avg_max_down = float(row["avg_max_down_pips"])  # lehimize (aşağı, negatif)

        # Lehimize hareketi mutlak değer al
        avg_max_down_abs = abs(avg_max_down)

        # TP/SL heuristiği (SHORT için):
        # - TP: lehimize ortalama max hareketin %50'si
        # - SL: aleyhimize ortalama max hareketin %70'i
        tp_pips = round(avg_max_down_abs * 0.5, 1)
        sl_pips = round(avg_max_up * 0.7, 1)

        if sl_pips <= 0:
            sl_pips = 10.0  # minimum güvenlik

        rr = round(tp_pips / sl_pips, 2) if sl_pips > 0 else None

        playbook[event_type] = {
            "event_type": event_type,
            "direction": "SHORT",
            "n_events": n_events,
            "pct_up": round(pct_up, 4),
            "pct_down": round(pct_down, 4),
            "down_bias": round(down_bias, 4),
            "avg_max_up_pips": round(avg_max_up, 2),
            "avg_max_down_pips": round(avg_max_down, 2),
            "short_score": round(short_score, 4),
            "suggested_tp_pips": tp_pips,
            "suggested_sl_pips": sl_pips,
            "rr": rr,
        }

    return playbook, filtered


def save_playbook_json(playbook: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(playbook, f, indent=2)
    logger.info("💾 Short playbook JSON kaydedildi: %s", path)


def save_report(filtered: pd.DataFrame, playbook: dict, path: str) -> None:
    lines: list[str] = []
    lines.append("SETUP PLAYBOOK SHORT v1")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SHORT SETUPS (filtered + sorted by short_score)")
    lines.append("-" * 80)

    for event_type, cfg in playbook.items():
        lines.append(f"EVENT TYPE: {event_type}")
        lines.append(f"  Direction     : {cfg['direction']}")
        lines.append(f"  n_events      : {cfg['n_events']}")
        lines.append(f"  pct_down      : {cfg['pct_down']}")
        lines.append(f"  pct_up        : {cfg['pct_up']}")
        lines.append(f"  down_bias     : {cfg['down_bias']}")
        lines.append(f"  avg_dn_pips   : {cfg['avg_max_down_pips']}")
        lines.append(f"  avg_up_pips   : {cfg['avg_max_up_pips']}")
        lines.append(f"  short_score   : {cfg['short_score']}")
        lines.append(f"  TP (pips)     : {cfg['suggested_tp_pips']}")
        lines.append(f"  SL (pips)     : {cfg['suggested_sl_pips']}")
        lines.append(f"  RR            : {cfg['rr']}")
        lines.append("-" * 80)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("💾 Short playbook report kaydedildi: %s", path)


def main():
    logger.info("=" * 80)
    logger.info("🚀 BUILD PLAYBOOK SHORT v1 BAŞLIYOR")
    logger.info("=" * 80)

    df = pd.read_csv(PATTERN_STATS_CSV)
    playbook, filtered = build_short_playbook(df)

    logger.info("✅ Toplam seçilen SHORT setup sayısı: %d", len(playbook))

    save_playbook_json(playbook, OUTPUT_JSON)
    save_report(filtered, playbook, OUTPUT_REPORT)

    logger.info("=" * 80)
    logger.info("✅ BUILD PLAYBOOK SHORT v1 TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()