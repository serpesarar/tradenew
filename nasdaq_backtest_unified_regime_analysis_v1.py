import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
# ============================================================================
# PATHLER
# ============================================================================
UNIFIED_SIGNALS_PATH = "./staging/nasdaq_fake_live_unified_signals_with_model_v1.parquet"

OUTPUT_PARQUET = "./staging/nasdaq_fake_live_unified_backtest_regimes_v1.parquet"
REPORT_TXT = "./models/nasdaq_backtest_unified_regime_analysis_v1.txt"

# 🔧 Spread + komisyon (pip cinsinden)  → bunları istediğin gibi ayarlayabilirsin
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

    # timestamp'i datetime yap
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    M30 kanal / range flag'lerinden basit bir regime label üret:
      - UP_TREND
      - DOWN_TREND
      - RANGE
      - OTHER
    """
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


def compute_trade_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    final_action ∈ {LONG, SHORT} olan satırlar için pip PnL + cost hesapla.

    Öncelik:
      1) tp_sl_result_label varsa → onu kullan (TP / SL / BE)
      2) Yoksa tp_sl_result'ı decode et:
           - primary mapping: 1 → TP, 0 → SL, 2 → BE  (dataset kodun)
           - fallback:        >0 → TP, <0 → SL, 0 → BE
    """

    if "final_action" not in df.columns:
        raise ValueError("❌ Unified df içinde 'final_action' kolonu yok.")

    trades = df[df["final_action"].isin(["LONG", "SHORT"])].copy()
    logger.info("   ✅ Trade satırları filtrelendi. shape=%s", trades.shape)

    # direction / side
    if "direction" in trades.columns:
        trades["direction"] = trades["direction"].fillna(trades["final_action"])
    else:
        trades["direction"] = trades["final_action"]

    # TP / SL pip mesafeleri
    if "tp_pips" not in trades.columns or "sl_pips" not in trades.columns:
        raise ValueError("❌ Trades içinde tp_pips ve/veya sl_pips kolonları yok.")

    trades["tp_pips"] = trades["tp_pips"].astype(float)
    trades["sl_pips"] = trades["sl_pips"].astype(float)

    # ------------------------------------------------------------------
    # 1) Önce hazır label var mı? (tp_sl_result_label)
    # ------------------------------------------------------------------
    if "tp_sl_result_label" in trades.columns:
        res = trades["tp_sl_result_label"].astype("string").str.upper().fillna("UNKNOWN")

    # ------------------------------------------------------------------
    # 2) Yoksa tp_sl_result'ı yorumla
    # ------------------------------------------------------------------
    elif "tp_sl_result" in trades.columns:
        raw = trades["tp_sl_result"]

        # Numeric parse dene
        num = pd.to_numeric(raw, errors="coerce")

        if num.notna().any():
            # Dataset-specific mapping: 1 → TP, 0 → SL, 2 → BE
            # Ama yine de fallback olarak >0/<0/0 kuralını da ekleyelim.
            mapped = np.select(
                [
                    num == 1,      # net TP
                    num == 0,      # net SL
                    num == 2,      # BE / none
                    num > 0,       # herhangi pozitif
                    num < 0        # herhangi negatif
                ],
                [
                    "TP",
                    "SL",
                    "BE",
                    "TP",
                    "SL"
                ],
                default="BE",
            )

            res = pd.Series(mapped, index=trades.index)

            # Parse edilemeyenler (NaN) için, ham string'e bak (TP/SL/BE yazıyor olabilir)
            raw_str = raw.astype("string").str.upper()
            res = res.where(num.notna(), raw_str.fillna("UNKNOWN"))

        else:
            # Hiç numeric parse edemediysek, tamamen string label olarak düşün
            res = raw.astype("string").str.upper().fillna("UNKNOWN")
    else:
        # Hiç kolon yoksa hepsini BE gibi say (çok edge-case)
        res = pd.Series(["BE"] * len(trades), index=trades.index)

    # ------------------------------------------------------------------
    # Brüt PnL (cost hariç) – TP/SL/BE'ye göre
    # ------------------------------------------------------------------
    gross_pips = np.where(
        res == "TP",
        trades["tp_pips"],
        np.where(
            res == "SL",
            -trades["sl_pips"],
            0.0,  # BE
        ),
    )

    trades["tp_sl_result_label"] = res
    trades["gross_pips_no_cost"] = gross_pips

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
            "timestamp", "final_action", "direction",
            "tp_pips", "sl_pips",
            "tp_sl_result", "tp_sl_result_label",
            "gross_pips_no_cost", "cost_pips", "net_pips"
        ]].head(1).to_string(index=False),
    )

    return trades
def add_time_period_labels(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Zaman serisi split (pseudo-OOS):
      - TRAIN_LIKE: ilk %80
      - OOS_LAST20: son %20
    """
    trades = trades.sort_values("timestamp").reset_index(drop=True)
    n = len(trades)
    if n == 0:
        trades["time_period"] = "UNKNOWN"
        return trades

    split_idx = int(n * 0.8)
    split_ts = trades.loc[split_idx, "timestamp"]

    logger.info("   ⏱️ Time-series split: ilk %%80 < %s → TRAIN_LIKE, sonrası → OOS_LAST20", split_ts)

    trades["time_period"] = np.where(
        trades["timestamp"] < split_ts,
        "TRAIN_LIKE",
        "OOS_LAST20",
    )

    return trades


def summarize_backtest(trades: pd.DataFrame) -> str:
    """
    Genel, direction ve regime bazlı özet rapor üretir (string).
    """
    lines = []
    lines.append("=" * 80)
    lines.append("NASDAQ UNIFIED BACKTEST + REGIME ANALYSIS v1")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Toplam trade sayısı: {len(trades)}")
    lines.append("")

    # Genel özet
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