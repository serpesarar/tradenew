#!/usr/bin/env python3
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

PLAYBOOK_V1_JSON = "./models/setup_playbook_long_v1.json"
BT_SIGNALS_PATH = "./staging/nasdaq_playbook_signals_bt_v1.parquet"

OUT_JSON = "./models/setup_playbook_long_v2.json"
OUT_REPORT = "./models/setup_playbook_long_v2_report.txt"
OUT_CSV = "./models/setup_playbook_long_v2_table.csv"

# Minimum sinyal sayısı filtresi (setup'ın ciddiye alınması için)
MIN_BT_SIGNALS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("playbook_v2")


# ------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------

def load_playbook_v1(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Playbook v1 JSON bulunamadı: {p}")

    logger.info("📥 Playbook v1 JSON yükleniyor: %s", p)
    data = json.loads(p.read_text(encoding="utf-8"))
    logger.info("✅ Playbook v1 setup sayısı: %d", len(data))
    return data


def load_bt_signals(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Backtest sinyal dosyası bulunamadı: {p}")

    logger.info("📥 Backtest sinyalleri yükleniyor: %s", p)
    df = pd.read_parquet(p)

    required = ["event_type", "bt_outcome", "bt_pnl_pips", "bt_holding_bars"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest dataset'inde eksik kolon(lar) var: {missing}")

    # Sadece anlamlı outcome'lar
    df = df[df["bt_outcome"].isin(["TP", "SL", "NONE"])].copy()
    logger.info("✅ Geçerli backtest satırı sayısı: %d", len(df))

    return df


# ------------------------------------------------------------------------------
# BUILD PERF TABLE
# ------------------------------------------------------------------------------

def build_perf_table(df_bt: pd.DataFrame) -> pd.DataFrame:
    logger.info("📊 Event_type bazında backtest özetleri hesaplanıyor...")

    def rate(series, value):
        return (series == value).mean()

    grp = (
        df_bt.groupby("event_type")
        .agg(
            bt_n_signals=("event_type", "count"),
            bt_tp_rate=("bt_outcome", lambda x: rate(x, "TP")),
            bt_sl_rate=("bt_outcome", lambda x: rate(x, "SL")),
            bt_none_rate=("bt_outcome", lambda x: rate(x, "NONE")),
            bt_avg_pnl_pips=("bt_pnl_pips", "mean"),
            bt_med_pnl_pips=("bt_pnl_pips", "median"),
            bt_avg_holding_bars=("bt_holding_bars", "mean"),
            bt_med_holding_bars=("bt_holding_bars", "median"),
            bt_max_pnl_pips=("bt_pnl_pips", "max"),
            bt_min_pnl_pips=("bt_pnl_pips", "min"),
        )
        .reset_index()
    )

    # Basit “realized_score” metriği: (tp_rate - sl_rate) * avg_pnl
    grp["bt_realized_score"] = (grp["bt_tp_rate"] - grp["bt_sl_rate"]) * grp[
        "bt_avg_pnl_pips"
    ]

    logger.info("✅ Perf tablosu oluşturuldu: %d satır", len(grp))
    return grp


# ------------------------------------------------------------------------------
# MERGE PLAYBOOK + PERF
# ------------------------------------------------------------------------------

def merge_playbook_with_perf(playbook_v1: dict, perf_df: pd.DataFrame) -> dict:
    logger.info("🔗 Playbook v1 ile backtest perf merge ediliyor...")

    perf_map = {row["event_type"]: row for _, row in perf_df.iterrows()}

    playbook_v2 = {}

    for ev_type, cfg in playbook_v1.items():
        cfg2 = dict(cfg)  # copy

        perf_row = perf_map.get(ev_type)
        if perf_row is None:
            # Bu event_type için sinyal tetiklenmemiş veya backtest'te yok
            cfg2.update(
                {
                    "bt_n_signals": 0,
                    "bt_tp_rate": None,
                    "bt_sl_rate": None,
                    "bt_none_rate": None,
                    "bt_avg_pnl_pips": None,
                    "bt_med_pnl_pips": None,
                    "bt_avg_holding_bars": None,
                    "bt_med_holding_bars": None,
                    "bt_realized_score": None,
                    "is_live": False,
                }
            )
        else:
            cfg2.update(
                {
                    "bt_n_signals": int(perf_row["bt_n_signals"]),
                    "bt_tp_rate": float(perf_row["bt_tp_rate"]),
                    "bt_sl_rate": float(perf_row["bt_sl_rate"]),
                    "bt_none_rate": float(perf_row["bt_none_rate"]),
                    "bt_avg_pnl_pips": float(perf_row["bt_avg_pnl_pips"]),
                    "bt_med_pnl_pips": float(perf_row["bt_med_pnl_pips"]),
                    "bt_avg_holding_bars": float(perf_row["bt_avg_holding_bars"]),
                    "bt_med_holding_bars": float(perf_row["bt_med_holding_bars"]),
                    "bt_max_pnl_pips": float(perf_row["bt_max_pnl_pips"]),
                    "bt_min_pnl_pips": float(perf_row["bt_min_pnl_pips"]),
                    "bt_realized_score": float(perf_row["bt_realized_score"]),
                    # Basit live filtresi: yeterli sinyal + tp_rate > 0.55 + avg_pnl > 0
                    "is_live": bool(
                        (perf_row["bt_n_signals"] >= MIN_BT_SIGNALS)
                        and (perf_row["bt_tp_rate"] > 0.55)
                        and (perf_row["bt_avg_pnl_pips"] > 0)
                    ),
                }
            )

        playbook_v2[ev_type] = cfg2

    logger.info("✅ Playbook v2 oluşturuldu: %d setup", len(playbook_v2))
    return playbook_v2


# ------------------------------------------------------------------------------
# REPORT
# ------------------------------------------------------------------------------

def build_report(playbook_v2: dict) -> str:
    lines = []
    lines.append("SETUP PLAYBOOK LONG v2 (with realized backtest stats)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Toplam setup sayısı: {len(playbook_v2)}")
    lines.append("")

    # Tabloya döküp sıralayalım
    rows = []
    for ev_type, cfg in playbook_v2.items():
        rows.append(
            {
                "event_type": ev_type,
                "direction": cfg.get("direction"),
                "n_events": cfg.get("n_events"),
                "pct_up": cfg.get("pct_up"),
                "pct_down": cfg.get("pct_down"),
                "up_bias": cfg.get("up_bias"),
                "tp_pips": cfg.get("suggested_tp_pips"),
                "sl_pips": cfg.get("suggested_sl_pips"),
                "rr": cfg.get("rr"),
                "bt_n_signals": cfg.get("bt_n_signals"),
                "bt_tp_rate": cfg.get("bt_tp_rate"),
                "bt_sl_rate": cfg.get("bt_sl_rate"),
                "bt_avg_pnl_pips": cfg.get("bt_avg_pnl_pips"),
                "bt_realized_score": cfg.get("bt_realized_score"),
                "is_live": cfg.get("is_live"),
            }
        )

    df = pd.DataFrame(rows)

    # CSV export
    df.to_csv(OUT_CSV, index=False)

    # Sadece live olanları (threshold'ı geçenleri) filtrele
    live = df[df["is_live"] == True].copy()
    live = live.sort_values("bt_realized_score", ascending=False)

    lines.append(f"Live (aktif) setup sayısı (threshold sonrası): {len(live)}")
    lines.append("")

    if not live.empty:
        lines.append("TOP 20 LIVE SETUPS (bt_realized_score'e göre sıralı)")
        lines.append("-" * 80)
        header = (
            f"{'event_type':40s} {'bt_n':>6s} {'tp%':>6s} {'sl%':>6s} "
            f"{'avg_pnl':>10s} {'score':>10s} {'TP':>6s} {'SL':>6s}"
        )
        lines.append(header)

        for _, r in live.head(20).iterrows():
            lines.append(
                f"{str(r['event_type'])[:40]:40s} "
                f"{int(r['bt_n_signals']):6d} "
                f"{(r['bt_tp_rate'] or 0):6.3f} "
                f"{(r['bt_sl_rate'] or 0):6.3f} "
                f"{(r['bt_avg_pnl_pips'] or 0):10.2f} "
                f"{(r['bt_realized_score'] or 0):10.2f} "
                f"{(r['tp_pips'] or 0):6.1f} "
                f"{(r['sl_pips'] or 0):6.1f}"
            )
    else:
        lines.append("⚠ Şu anki threshold ile live setup bulunamadı.")
        lines.append("   • MIN_BT_SIGNALS, tp_rate ve avg_pnl şartlarını gevşetebilirsin.")
    lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("🚀 BUILD PLAYBOOK LONG v2 (from backtest) BAŞLIYOR")
    logger.info("=" * 80)

    playbook_v1 = load_playbook_v1(PLAYBOOK_V1_JSON)
    df_bt = load_bt_signals(BT_SIGNALS_PATH)
    perf_df = build_perf_table(df_bt)

    playbook_v2 = merge_playbook_with_perf(playbook_v1, perf_df)

    # JSON kaydet
    Path(OUT_JSON).write_text(
        json.dumps(playbook_v2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("💾 Playbook v2 JSON kaydedildi: %s", OUT_JSON)

    # Rapor
    report = build_report(playbook_v2)
    Path(OUT_REPORT).write_text(report, encoding="utf-8")
    logger.info("💾 Rapor kaydedildi: %s", OUT_REPORT)

    logger.info("=" * 80)
    logger.info("✅ BUILD PLAYBOOK LONG v2 TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()