import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# PATHLER
# ============================================================================
UNIFIED_SIGNALS_PATH = "./staging/nasdaq_fake_live_unified_signals_with_model_v1.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_unified_backtest_regimes_v1.parquet"
REPORT_TXT = "./models/nasdaq_backtest_unified_regime_analysis_v1.txt"

# 🔧 Spread + komisyon (pip cinsinden) – tunable
SPREAD_PIPS_LONG = 1.0
SPREAD_PIPS_SHORT = 1.0
COMMISSION_PIPS = 0.5  # trade başına ek maliyet


# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("unified_backtest_regimes")


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================
def load_unified_df() -> pd.DataFrame:
    logger.info("=" * 78)
    logger.info("🚀 UNIFIED BACKTEST + REGIME ANALYSIS v1 BAŞLIYOR")
    logger.info("=" * 78)

    logger.info("📥 Unified fake-live df yükleniyor: %s", UNIFIED_SIGNALS_PATH)
    df = pd.read_parquet(UNIFIED_SIGNALS_PATH)
    logger.info("   ✅ Unified df shape: %s", df.shape)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # direction/final_action'i string yap
    for col in ["direction", "final_action"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """M30 kanal / range flag'lerinden basit bir regime label üretir."""

    df = df.copy()

    up_col = "chan_is_up_M30_at_entry"
    down_col = "chan_is_down_M30_at_entry"
    range_col = "is_range_M30_at_entry"

    for col in [up_col, down_col, range_col]:
        if col not in df.columns:
            logger.warning("⚠️ Regime kolonu bulunamadı: %s (hepsi yoksa regime=UNKNOWN olacak)", col)

    df[up_col] = df.get(up_col, 0).fillna(0).astype(int)
    df[down_col] = df.get(down_col, 0).fillna(0).astype(int)
    df[range_col] = df.get(range_col, 0).fillna(0).astype(int)

    def _regime_row(row) -> str:
        if row[range_col] == 1:
            return "RANGE"
        if row[up_col] == 1 and row[down_col] == 0:
            return "UP_TREND"
        if row[down_col] == 1 and row[up_col] == 0:
            return "DOWN_TREND"
        return "OTHER"

    df["regime_M30"] = df.apply(_regime_row, axis=1)

    logger.info("   ✅ Regime label eklendi: regime_M30 (unique=%s)", df["regime_M30"].unique())
    return df


def _compute_single_trade(row: pd.Series) -> dict:
    """Realized TP/SL outcome based on observed max moves (pip)."""

    action = str(row.get("final_action", "")).upper()
    tp_pips = float(row.get("tp_pips", np.nan))
    sl_pips = float(row.get("sl_pips", np.nan))
    max_up = float(row.get("max_up_move_pips", np.nan))
    max_down = float(row.get("max_down_move_pips", np.nan))

    hit_tp = False
    hit_sl = False

    if action == "LONG":
        hit_tp = pd.notna(max_up) and pd.notna(tp_pips) and max_up >= tp_pips
        hit_sl = pd.notna(max_down) and pd.notna(sl_pips) and (-max_down) >= sl_pips
    elif action == "SHORT":
        hit_tp = pd.notna(max_down) and pd.notna(tp_pips) and (-max_down) >= tp_pips
        hit_sl = pd.notna(max_up) and pd.notna(sl_pips) and max_up >= sl_pips

    # If both triggered (rare/unknown order), prefer TP for optimism consistency with legacy
    result_label = "BE"
    if hit_tp and not hit_sl:
        result_label = "TP"
    elif hit_sl and not hit_tp:
        result_label = "SL"
    elif hit_tp and hit_sl:
        result_label = "TP"

    gross_pips = 0.0
    if result_label == "TP":
        gross_pips = tp_pips
    elif result_label == "SL":
        gross_pips = -sl_pips

    return {"tp_sl_result_label": result_label, "gross_pips_no_cost": gross_pips}


def compute_trade_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """final_action ∈ {LONG, SHORT} olan satırlar için pip PnL + cost hesapla."""

    if "final_action" not in df.columns:
        raise ValueError("❌ Unified df içinde 'final_action' kolonu yok.")

    trades = df[df["final_action"].isin(["LONG", "SHORT"])].copy()
    logger.info("   ✅ Trade satırları filtrelendi. shape=%s", trades.shape)

    # Zorunlu pip kolonları yoksa erken uyarı ver (hesap doğru yapılamaz)
    required_cols = {"tp_pips", "sl_pips", "max_up_move_pips", "max_down_move_pips"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"❌ Trades içinde eksik zorunlu pip kolonları var: {missing}")

    # Numerik kolon tiplerini zorla
    for col in required_cols:
        trades[col] = pd.to_numeric(trades.get(col, np.nan), errors="coerce")

    # Realized outcome from price path
    realized = trades.apply(_compute_single_trade, axis=1, result_type="expand")
    trades["tp_sl_result_label"] = realized["tp_sl_result_label"]
    trades["gross_pips_no_cost"] = realized["gross_pips_no_cost"]

    # Spread + komisyon maliyeti
    spread_cost = np.where(
        trades["final_action"] == "LONG",
        SPREAD_PIPS_LONG,
        SPREAD_PIPS_SHORT,
    )
    total_cost = spread_cost + COMMISSION_PIPS
    trades["cost_pips"] = total_cost
    trades["net_pips"] = trades["gross_pips_no_cost"] - trades["cost_pips"]

    # Basit win/lose flag'leri
    trades["is_win"] = trades["net_pips"] > 0
    trades["is_loss"] = trades["net_pips"] < 0
    trades["is_be"] = trades["net_pips"] == 0

    logger.info(
        "   ✅ PnL hesaplandı. Örnek satır:\n%s",
        trades[[
            "timestamp", "final_action", "tp_pips", "sl_pips", "tp_sl_result_label",
            "gross_pips_no_cost", "cost_pips", "net_pips"
        ]].head(1).to_string(index=False),
    )

    return trades


def add_time_period_labels(trades: pd.DataFrame) -> pd.DataFrame:
    """Zaman serisi split (pseudo-OOS): ilk %80 → TRAIN_LIKE, son %20 → OOS_LAST20."""

    trades = trades.sort_values("timestamp").reset_index(drop=True)
    n = len(trades)
    if n == 0:
        trades["time_period"] = "UNKNOWN"
        return trades

    split_idx = int(n * 0.8)
    split_ts = trades.loc[split_idx, "timestamp"]

    logger.info("   ⏱️ Time-series split: ilk %80 < %s → TRAIN_LIKE, sonrası → OOS_LAST20", split_ts)

    trades["time_period"] = np.where(
        trades["timestamp"] < split_ts,
        "TRAIN_LIKE",
        "OOS_LAST20",
    )

    return trades


def summarize_backtest(trades: pd.DataFrame) -> str:
    """Genel, direction ve regime bazlı özet rapor üretir (string)."""

    lines = []
    lines.append("=" * 80)
    lines.append("NASDAQ UNIFIED BACKTEST + REGIME ANALYSIS v1")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Toplam trade sayısı: {len(trades)}")
    lines.append("")

    def summary_block(name: str, sub_df: pd.DataFrame):
        if len(sub_df) == 0:
            lines.append(f"[{name}] → trade yok.")
            lines.append("")
            return

        avg_net = sub_df["net_pips"].mean()
        med_net = sub_df["net_pips"].median()
        win_rate = sub_df["is_win"].mean()
        be_rate = sub_df["is_be"].mean()
        loss_rate = sub_df["is_loss"].mean()
        lines.append(f"[{name}]")
        lines.append(f"  n_trades   : {len(sub_df)}")
        lines.append(f"  win_rate   : {win_rate:.3f}")
        lines.append(f"  be_rate    : {be_rate:.3f}")
        lines.append(f"  loss_rate  : {loss_rate:.3f}")
        lines.append(f"  avg_net    : {avg_net:.2f} pips")
        lines.append(f"  med_net    : {med_net:.2f} pips")
        lines.append("")

    summary_block("OVERALL", trades)
    summary_block("LONG ONLY", trades[trades["final_action"] == "LONG"])
    summary_block("SHORT ONLY", trades[trades["final_action"] == "SHORT"])

    # Regime x Direction
    lines.append("=" * 40)
    lines.append("REGIME x DIRECTION")
    lines.append("=" * 40)
    lines.append("")

    for regime in sorted(trades["regime_M30"].unique()):
        sub = trades[trades["regime_M30"] == regime]
        summary_block(f"REGIME={regime} (ALL)", sub)
        summary_block(f"REGIME={regime} (LONG)", sub[sub["final_action"] == "LONG"])
        summary_block(f"REGIME={regime} (SHORT)", sub[sub["final_action"] == "SHORT"])

    # Time period breakdown
    if "time_period" in trades.columns:
        lines.append("=" * 40)
        lines.append("TIME PERIOD x DIRECTION (pseudo-OOS)")
        lines.append("=" * 40)
        lines.append("")
        for period in sorted(trades["time_period"].unique()):
            sub = trades[trades["time_period"] == period]
            summary_block(f"PERIOD={period} (ALL)", sub)
            summary_block(f"PERIOD={period} (LONG)", sub[sub["final_action"] == "LONG"])
            summary_block(f"PERIOD={period} (SHORT)", sub[sub["final_action"] == "SHORT"])

    return "\n".join(lines)


def save_outputs(trades: pd.DataFrame, report_text: str) -> None:
    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("💾 Backtest trades (Parquet): %s", OUTPUT_PARQUET)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("💾 Rapor kaydedildi: %s", REPORT_TXT)


# ---------------------------------------------------------------------------
# Quick diagnostic helper
# ---------------------------------------------------------------------------
def quick_diagnostics(path: str = OUTPUT_PARQUET) -> None:
    """Load backtest parquet and print basic win-rates by direction/regime."""

    if not Path(path).exists():
        logger.warning("⚠️ Diagnostics skipped: %s bulunamadı", path)
        return

    trades = pd.read_parquet(path)
    if "regime_M30" not in trades.columns:
        logger.warning("⚠️ Diagnostics skipped: regime_M30 kolonu yok")
        return

    for col in ["final_action", "regime_M30"]:
        trades[col] = trades[col].astype("string")

    logger.info("📊 Quick diagnostics from %s", path)
    for direction in ["LONG", "SHORT"]:
        sub = trades[trades["final_action"] == direction]
        if len(sub) == 0:
            logger.info("   • %s: trade yok", direction)
            continue
        win_rate = sub.get("is_win", pd.Series(dtype=float)).mean()
        logger.info("   • %s trades: n=%d win_rate=%.3f", direction, len(sub), win_rate)
        for regime in sorted(sub["regime_M30"].dropna().unique()):
            sub_r = sub[sub["regime_M30"] == regime]
            logger.info(
                "       - regime=%s: n=%d win_rate=%.3f",
                regime,
                len(sub_r),
                sub_r.get("is_win", pd.Series(dtype=float)).mean(),
            )


# ============================================================================
# MAIN
# ============================================================================
def main():
    df = load_unified_df()
    df = add_regime_labels(df)

    trades = compute_trade_pnl(df)
    trades = add_time_period_labels(trades)

    report_text = summarize_backtest(trades)
    logger.info("📊 Backtest + regime raporu hazır.")
    logger.info("\n" + "\n".join(report_text.splitlines()[:30]) + "\n... (raporun devamı txt dosyasında)")

    save_outputs(trades, report_text)

    logger.info("=" * 78)
    logger.info("✅ UNIFIED BACKTEST + REGIME ANALYSIS v1 TAMAMLANDI")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
