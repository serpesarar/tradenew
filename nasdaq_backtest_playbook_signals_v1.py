#!/usr/bin/env python3
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------

SIGNALS_PATH = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"
MASTER_PATH = "./staging/nasdaq_master_features_v1.parquet"

# Kaç bar ileriye kadar bakacağız? (M30'da 48 bar ≈ 1 gün)
MAX_HORIZON_BARS = 48

OUT_PARQUET = "./staging/nasdaq_playbook_signals_bt_v1.parquet"
OUT_CSV = "./staging/nasdaq_playbook_signals_bt_v1.csv"
OUT_REPORT = "./models/nasdaq_playbook_signals_bt_report_v1.txt"

# --------------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("playbook_bt")


# --------------------------------------------------------------------------------
# DATA LOADERS
# --------------------------------------------------------------------------------

def load_signals(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sinyal dosyası bulunamadı: {p}")

    logger.info("📥 Sinyaller yükleniyor: %s", p)
    df = pd.read_parquet(p)

    if "timestamp" not in df.columns:
        raise ValueError("Sinyal dataset'inde 'timestamp' kolonu yok!")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("✅ Sinyal shape: %s", df.shape)
    return df


def load_price_master(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MASTER dosyası bulunamadı: {p}")

    logger.info("📥 MASTER (price) yükleniyor: %s", p)
    df = pd.read_parquet(p)

    required_cols = ["timestamp", "Open_M30", "High_M30", "Low_M30", "Close_M30"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"MASTER içinde eksik OHLC kolonları var: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    price = df[required_cols].copy()
    price = price.drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    logger.info("✅ Price dataframe shape: %s", price.shape)
    return price


# --------------------------------------------------------------------------------
# BACKTEST LOGIC
# --------------------------------------------------------------------------------

def backtest_signal(row, price_df: pd.DataFrame):
    """
    Tek bir sinyal için TP/SL backtest'i.
    LONG için yazdık; ileride SHORT setup'lar eklenirse mantığı genişletiriz.
    """

    ts = row["timestamp"]
    direction = row.get("direction", "LONG")
    entry = float(row["entry_price"])
    tp_price = float(row["tp_price"])
    sl_price = float(row["sl_price"])

    # timestamp price_df içinde yoksa outcome hesaplayamayız
    if ts not in price_df.index:
        return {
            "bt_outcome": "NO_PRICE",
            "bt_holding_bars": np.nan,
            "bt_pnl_pips": np.nan,
            "bt_max_fav_pips": np.nan,
            "bt_max_adv_pips": np.nan,
        }

    loc = price_df.index.get_loc(ts)

    # loc integer veya dilim olabilir; biz integer bekliyoruz
    if isinstance(loc, slice):
        # Aynı timestamp'ten birden fazla varsa ilkini al
        start_idx = loc.start
    else:
        start_idx = loc

    # Entry bar'dan sonra gelen barlar, max horizon kadar
    # (Entry bar'ın kendisini dahil etmiyoruz)
    fwd = price_df.iloc[start_idx + 1:start_idx + 1 + MAX_HORIZON_BARS]

    if fwd.empty:
        return {
            "bt_outcome": "NO_FWD",
            "bt_holding_bars": np.nan,
            "bt_pnl_pips": np.nan,
            "bt_max_fav_pips": np.nan,
            "bt_max_adv_pips": np.nan,
        }

    high = fwd["High_M30"].values
    low = fwd["Low_M30"].values
    close = fwd["Close_M30"].values

    # LONG senaryosu
    if direction == "LONG":
        hit_tp = high >= tp_price
        hit_sl = low <= sl_price

        tp_idx = np.where(hit_tp)[0][0] if hit_tp.any() else None
        sl_idx = np.where(hit_sl)[0][0] if hit_sl.any() else None

        # Max favorable/adverse (entry'ye göre)
        max_fav = (high.max() - entry)
        max_adv = (low.min() - entry)

        # Outcome belirleme
        if tp_idx is None and sl_idx is None:
            # Ne TP ne SL vurmuş → bar son close'una göre kapatalım
            final_price = close[-1]
            pnl = final_price - entry
            outcome = "NONE"
            holding_bars = len(fwd)
        else:
            # Hangisi önce gelmiş?
            if tp_idx is not None and sl_idx is not None:
                if tp_idx < sl_idx:
                    outcome = "TP"
                    final_price = tp_price
                    holding_bars = tp_idx + 1
                elif sl_idx < tp_idx:
                    outcome = "SL"
                    final_price = sl_price
                    holding_bars = sl_idx + 1
                else:
                    # Aynı bar içinde hem TP hem SL vurma durumu:
                    # Konservatif: SL kabul et
                    outcome = "SL"
                    final_price = sl_price
                    holding_bars = sl_idx + 1
            elif tp_idx is not None:
                outcome = "TP"
                final_price = tp_price
                holding_bars = tp_idx + 1
            else:
                outcome = "SL"
                final_price = sl_price
                holding_bars = sl_idx + 1

            pnl = final_price - entry

        # PnL'i "pips" olarak al (NASDAQ için zaten point gibi kullanıyoruz)
        pnl_pips = pnl

    else:
        # Şimdilik SHORT yok; ileride ekleyebiliriz
        return {
            "bt_outcome": "UNSUPPORTED_DIR",
            "bt_holding_bars": np.nan,
            "bt_pnl_pips": np.nan,
            "bt_max_fav_pips": np.nan,
            "bt_max_adv_pips": np.nan,
        }

    return {
        "bt_outcome": outcome,
        "bt_holding_bars": holding_bars,
        "bt_pnl_pips": pnl_pips,
        "bt_max_fav_pips": max_fav,
        "bt_max_adv_pips": max_adv,
    }


def run_backtest(signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🚀 Backtest başlıyor... (sinyal sayısı: %d)", len(signals))

    results = []
    total = len(signals)

    for i, row in signals.iterrows():
        if i % 1000 == 0:
            logger.info("   → Progress: %d / %d (%.1f%%)", i, total, 100 * i / total)

        res = backtest_signal(row, price_df)
        results.append(res)

    logger.info("✅ Backtest loop bitti.")

    res_df = pd.DataFrame(results)
    out = pd.concat([signals.reset_index(drop=True), res_df], axis=1)
    return out


# --------------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------------

def build_report(df_bt: pd.DataFrame) -> str:
    lines = []
    lines.append("NASDAQ PLAYBOOK SIGNALS BACKTEST v1")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Toplam sinyal: {len(df_bt):,}")
    lines.append("")

    valid = df_bt[df_bt["bt_outcome"].isin(["TP", "SL", "NONE"])].copy()
    lines.append(f"Backtest edilen (geçerli outcome'u olan) sinyal sayısı: {len(valid):,}")
    lines.append("")

    # Genel winrate, PnL
    tp_rate = (valid["bt_outcome"] == "TP").mean()
    sl_rate = (valid["bt_outcome"] == "SL").mean()
    none_rate = (valid["bt_outcome"] == "NONE").mean()

    avg_pnl = valid["bt_pnl_pips"].mean()
    avg_holding = valid["bt_holding_bars"].mean()

    lines.append(f"TP oranı   : {tp_rate:.3f}")
    lines.append(f"SL oranı   : {sl_rate:.3f}")
    lines.append(f"NONE oranı : {none_rate:.3f}")
    lines.append("")
    lines.append(f"Ortalama PnL (pips): {avg_pnl:.2f}")
    lines.append(f"Ortalama bekleme barı: {avg_holding:.2f}")
    lines.append("")

    # Event type bazında özet
    lines.append("TOP EVENT TYPES BY N_SIGNALS (first 30)")
    lines.append("-" * 80)
    grp = valid.groupby("event_type").agg(
        n_signals=("event_type", "count"),
        tp_rate=("bt_outcome", lambda x: (x == "TP").mean()),
        sl_rate=("bt_outcome", lambda x: (x == "SL").mean()),
        none_rate=("bt_outcome", lambda x: (x == "NONE").mean()),
        avg_pnl=("bt_pnl_pips", "mean"),
        avg_holding=("bt_holding_bars", "mean"),
    ).reset_index().sort_values("n_signals", ascending=False)

    top = grp.head(30)

    lines.append(
        f"{'event_type':40s} {'n':>6s} {'tp':>6s} {'sl':>6s} "
        f"{'none':>6s} {'avg_pnl':>10s} {'avg_hold':>10s}"
    )
    for _, r in top.iterrows():
        lines.append(
            f"{str(r['event_type'])[:40]:40s} "
            f"{int(r['n_signals']):6d} "
            f"{r['tp_rate']:.3f} "
            f"{r['sl_rate']:.3f} "
            f"{r['none_rate']:.3f} "
            f"{r['avg_pnl']:10.2f} "
            f"{r['avg_holding']:10.2f}"
        )

    return "\n".join(lines)


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("🚀 NASDAQ PLAYBOOK SIGNAL BACKTEST v1 BAŞLIYOR")
    logger.info("=" * 80)

    signals = load_signals(SIGNALS_PATH)
    price_df = load_price_master(MASTER_PATH)

    df_bt = run_backtest(signals, price_df)

    # KAYDET
    logger.info("💾 Backtest sonuçları kaydediliyor...")
    df_bt.to_parquet(OUT_PARQUET, index=False)
    df_bt.to_csv(OUT_CSV, index=False)
    logger.info("✅ %s", OUT_PARQUET)
    logger.info("✅ %s", OUT_CSV)

    # RAPOR
    report = build_report(df_bt)
    Path(OUT_REPORT).write_text(report, encoding="utf-8")
    logger.info("💾 Rapor kaydedildi: %s", OUT_REPORT)

    logger.info("=" * 80)
    logger.info("✅ NASDAQ PLAYBOOK SIGNAL BACKTEST v1 TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()