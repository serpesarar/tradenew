#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

SIGNALS_PATH = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"

# ------------------- LOAD -------------------

@st.cache_data
def load_signals(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sinyal dosyası bulunamadı: {path}")
    df = pd.read_parquet(p)
    # timestamp'i datetime yap
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    st.set_page_config(
        page_title="NASDAQ Playbook Signals",
        layout="wide",
    )

    st.title("📈 NASDAQ – Playbook Signal Panel v1")

    # ---------- DATA ----------
    df = load_signals(SIGNALS_PATH)

    st.sidebar.header("Filtreler")

    # Tarih filtresi
    if "timestamp" in df.columns:
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        start_date, end_date = st.sidebar.date_input(
            "Tarih aralığı",
            [min_ts.date(), max_ts.date()],
        )
        # Streamlit bazen tek date dönüyor, onu da handle edelim
        if isinstance(start_date, list):
            start_date, end_date = start_date
        mask_date = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        df = df[mask_date]

    # Event type filtresi
    event_types = sorted(df["event_type"].unique())
    default_events = event_types[:10] if len(event_types) > 10 else event_types
    selected_events = st.sidebar.multiselect(
        "Event type (pattern) seç",
        options=event_types,
        default=default_events
    )
    if selected_events:
        df = df[df["event_type"].isin(selected_events)]

    # RR, pct_up filtreleri
    min_rr = st.sidebar.slider("Min RR", 0.5, 3.0, 1.0, 0.05)
    df = df[df["rr"] >= min_rr]

    min_pct_up = st.sidebar.slider("Min pct_up", 0.5, 0.8, 0.6, 0.01)
    df = df[df["pct_up"] >= min_pct_up]

    # SR filtresi (optional)
    sr_filter = st.sidebar.selectbox(
        "Nearest SR tipi",
        options=["Hepsi", "SUPPORT", "RESISTANCE", "NONE"],
        index=0,
    )
    if sr_filter == "SUPPORT":
        df = df[df["nearest_sr_type"] == "SUPPORT"]
    elif sr_filter == "RESISTANCE":
        df = df[df["nearest_sr_type"] == "RESISTANCE"]
    elif sr_filter == "NONE":
        df = df[df["nearest_sr_type"].isna()]

    st.markdown(f"**Toplam sinyal (filtre sonrası):** {len(df):,}")

    # ---------- ÜST ÖZET (METRİKLER) ----------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ortalama RR", f"{df['rr'].mean():.2f}")
    with col2:
        st.metric("Ortalama pct_up", f"{df['pct_up'].mean():.3f}")
    with col3:
        st.metric("Ortalama TP (pips)", f"{df['tp_pips'].mean():.1f}")
    with col4:
        st.metric("Ortalama SL (pips)", f"{df['sl_pips'].mean():.1f}")

    # ---------- TABLAR ----------
    tab1, tab2, tab3 = st.tabs(["📋 Sinyal Listesi", "📊 Setup Özeti", "🧱 SR / Wave Context"])

    with tab1:
        st.subheader("Sinyal Tablosu")

        show_cols = [
            "timestamp", "event_type", "direction",
            "entry_price", "tp_price", "sl_price",
            "tp_pips", "sl_pips", "rr",
            "pct_up", "pct_down", "up_bias",
            "avg_max_up_pips", "avg_max_down_pips",
            "nearest_sr_type", "nearest_sr_dist_pips",
            "signal_wave_at_entry", "wave_strength_pips_at_entry",
            "chan_is_up_M30_at_entry", "chan_is_down_M30_at_entry", "is_range_M30_at_entry",
        ]
        show_cols = [c for c in show_cols if c in df.columns]

        st.dataframe(
            df[show_cols].sort_values("timestamp").reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    with tab2:
        st.subheader("Event Type Bazında Özet")

        group_cols = ["event_type", "direction"]
        agg = df.groupby(group_cols).agg(
            n_signals=("event_type", "count"),
            mean_rr=("rr", "mean"),
            mean_pct_up=("pct_up", "mean"),
            mean_tp_pips=("tp_pips", "mean"),
            mean_sl_pips=("sl_pips", "mean"),
            mean_nearest_sr_dist=("nearest_sr_dist_pips", "mean"),
        ).reset_index().sort_values("mean_pct_up", ascending=False)

        st.dataframe(agg, use_container_width=True, height=500)

    with tab3:
        st.subheader("SR / Wave / Channel Context")

        context_cols = [
            "timestamp", "event_type", "direction",
            "entry_price",
            "sr_support_price_at_entry", "sr_support_strength_at_entry", "sr_support_distance_pips_at_entry",
            "sr_resistance_price_at_entry", "sr_resistance_strength_at_entry", "sr_resistance_distance_pips_at_entry",
            "sr_near_support_at_entry", "sr_near_resistance_at_entry",
            "nearest_sr_type", "nearest_sr_dist_pips",
            "signal_wave_at_entry", "wave_strength_pips_at_entry", "wave_duration_bars_at_entry",
            "up_move_pips_at_entry", "down_move_pips_at_entry",
            "chan_is_up_M30_at_entry", "chan_is_down_M30_at_entry",
            "is_range_M30_at_entry", "near_lower_chan_M30_at_entry", "near_upper_chan_M30_at_entry",
        ]
        context_cols = [c for c in context_cols if c in df.columns]

        st.dataframe(
            df[context_cols].sort_values("timestamp").reset_index(drop=True),
            use_container_width=True,
            height=500,
        )


if __name__ == "__main__":
    main()