import logging
from pathlib import Path

import pandas as pd
import streamlit as st

PANEL_FEED_PATH = "./staging/nasdaq_panel_signals_feed_v1.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nasdaq_panel_app_v1")


@st.cache_data
def load_panel_feed() -> pd.DataFrame:
    logger.info("📥 Panel feed yükleniyor: %s", PANEL_FEED_PATH)
    df = pd.read_parquet(PANEL_FEED_PATH)

    # timestamp normalle
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # side yoksa final_action'dan türet
    if "side" not in df.columns and "final_action" in df.columns:
        df["side"] = df["final_action"]

    return df


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}

    if df.empty:
        return stats

    # Genel
    stats["n_trades"] = len(df)
    stats["n_long"] = (df["side"] == "LONG").sum() if "side" in df.columns else 0
    stats["n_short"] = (df["side"] == "SHORT").sum() if "side" in df.columns else 0

    # Model confidence
    if "max_prob" in df.columns:
        stats["avg_conf"] = df["max_prob"].mean()
        stats["p_high_conf"] = (df["max_prob"] >= 0.6).mean()
    else:
        stats["avg_conf"] = None
        stats["p_high_conf"] = None

    # Directional win-rate
    if "future_dir" in df.columns:
        long_mask = df["side"] == "LONG"
        short_mask = df["side"] == "SHORT"

        if long_mask.any():
            dir_col_long = df.loc[long_mask, "future_dir"]
            stats["long_dir_win"] = ((dir_col_long == "UP") | (dir_col_long == 1)).mean()
        else:
            stats["long_dir_win"] = None

        if short_mask.any():
            dir_col_short = df.loc[short_mask, "future_dir"]
            stats["short_dir_win"] = ((dir_col_short == "DOWN") | (dir_col_short == 2)).mean()
        else:
            stats["short_dir_win"] = None
    else:
        stats["long_dir_win"] = None
        stats["short_dir_win"] = None

    # TP/SL oranları (varsa)
    if "tp_sl_result" in df.columns:
        long_mask = df["side"] == "LONG"
        short_mask = df["side"] == "SHORT"

        def tp_sl_be_rates(mask):
            sub = df.loc[mask, "tp_sl_result"]
            if sub.empty:
                return None, None, None
            tp = (sub == "TP").mean()
            sl = (sub == "SL").mean()
            be = (sub == "BE").mean()
            return tp, sl, be

        stats["long_tp"], stats["long_sl"], stats["long_be"] = tp_sl_be_rates(long_mask)
        stats["short_tp"], stats["short_sl"], stats["short_be"] = tp_sl_be_rates(short_mask)

    else:
        stats["long_tp"] = stats["long_sl"] = stats["long_be"] = None
        stats["short_tp"] = stats["short_sl"] = stats["short_be"] = None

    return stats


def main():
    st.set_page_config(
        page_title="NASDAQ MetaBrain – Signal Panel v1",
        layout="wide",
    )

    st.title("🧠 NASDAQ MetaBrain – Signal Panel v1")
    st.caption("Fake-live unified feed üzerinden LONG + SHORT sinyalleri inceleme")

    df = load_panel_feed()

    if df.empty:
        st.error("Panel feed boş görünüyor. Önce fake-live pipeline'ları çalıştırman gerekiyor.")
        st.stop()

    # -------------------------------------------------------------------------
    # Sidebar Filtreler
    # -------------------------------------------------------------------------
    st.sidebar.header("🔍 Filtreler")

    # Tarih filtresi
    if "timestamp" in df.columns:
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        date_range = st.sidebar.date_input(
            "Tarih aralığı",
            value=(min_ts.date(), max_ts.date()),
            min_value=min_ts.date(),
            max_value=max_ts.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask_date = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
            df = df[mask_date]

    # Side filtresi
    if "side" in df.columns:
        sides = sorted(df["side"].dropna().unique().tolist())
        selected_sides = st.sidebar.multiselect("Side", sides, default=sides)
        if selected_sides:
            df = df[df["side"].isin(selected_sides)]

    # Min model confidence
    if "max_prob" in df.columns:
        min_conf = float(df["max_prob"].min())
        max_conf = float(df["max_prob"].max())
        conf_thr = st.sidebar.slider(
            "Minimum model confidence (max_prob)",
            min_value=round(min_conf, 2),
            max_value=round(max_conf, 2),
            value=0.60,
            step=0.01,
        )
        df = df[df["max_prob"] >= conf_thr]

    # Min RR
    if "rr" in df.columns:
        min_rr = float(df["rr"].min())
        max_rr = float(df["rr"].max())
        rr_thr = st.sidebar.slider(
            "Minimum RR",
            min_value=round(min_rr, 2),
            max_value=round(max_rr, 2),
            value=min(1.0, round(max_rr, 2)),
            step=0.05,
        )
        df = df[df["rr"] >= rr_thr]

    st.sidebar.write(f"Toplam filtrelenmiş trade sayısı: **{len(df)}**")

    # -------------------------------------------------------------------------
    # Üst Metric Kartları
    # -------------------------------------------------------------------------
    stats = compute_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Trades", stats.get("n_trades", 0))
    with col2:
        st.metric("LONG Trades", stats.get("n_long", 0))
    with col3:
        st.metric("SHORT Trades", stats.get("n_short", 0))
    with col4:
        if stats.get("avg_conf") is not None:
            st.metric("Avg. Model Confidence", f"{stats['avg_conf']:.3f}")
        else:
            st.metric("Avg. Model Confidence", "—")

    col5, col6 = st.columns(2)
    with col5:
        if stats.get("long_dir_win") is not None:
            st.metric("LONG Directional Win%", f"{100 * stats['long_dir_win']:.1f}%")
        else:
            st.metric("LONG Directional Win%", "—")
        if stats.get("long_tp") is not None:
            st.caption(
                f"TP: {100*stats['long_tp']:.1f}% • SL: {100*stats['long_sl']:.1f}% • BE: {100*stats['long_be']:.1f}%"
            )

    with col6:
        if stats.get("short_dir_win") is not None:
            st.metric("SHORT Directional Win%", f"{100 * stats['short_dir_win']:.1f}%")
        else:
            st.metric("SHORT Directional Win%", "—")
        if stats.get("short_tp") is not None:
            st.caption(
                f"TP: {100*stats['short_tp']:.1f}% • SL: {100*stats['short_sl']:.1f}% • BE: {100*stats['short_be']:.1f}%"
            )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Trade Listesi (Tablo)
    # -------------------------------------------------------------------------
    st.subheader("📋 Trade Listesi")

    show_cols = [
        "timestamp",
        "side",
        "event_type",
        "entry_price",
        "tp_pips",
        "sl_pips",
        "rr",
        "max_prob",
        "pred_label",
        "future_dir",
        "tp_sl_result",
        "max_up_move_pips",
        "max_down_move_pips",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(
        df[show_cols].sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=600,
    )

    st.caption(
        "Bu v1 panel sadece fake-live unified feed üzerinden çalışıyor. "
        "Sonraki adımda gerçek-time IP / websocket verisine bağlayıp aynı arayüzü canlıya taşıyacağız."
    )


if __name__ == "__main__":
    main()