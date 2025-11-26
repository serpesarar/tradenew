import pandas as pd
import streamlit as st

SIGNALS_PATH = "./staging/nasdaq_playbook_signals_from_master_v1.parquet"


@st.cache_data
def load_signals():
    df = pd.read_parquet(SIGNALS_PATH)

    # timestamp düzgün olsun
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # nearest_sr_type / dist temizle
    if "nearest_sr_type" in df.columns:
        df["nearest_sr_type"] = df["nearest_sr_type"].fillna("NONE").astype(str)

    if "nearest_sr_dist_pips" in df.columns:
        df["nearest_sr_dist_pips"] = df["nearest_sr_dist_pips"].astype(float)

    # direction string olsun
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str)

    return df


def main():
    st.set_page_config(
        page_title="NASDAQ Playbook Panel v2",
        layout="wide",
    )

    st.title("📊 NASDAQ Playbook Panel v2 (Backtested Long Setups)")

    # --- DATA LOAD ---
    try:
        df = load_signals()
    except FileNotFoundError:
        st.error(f"Signals file not found: {SIGNALS_PATH}")
        st.stop()

    if df.empty:
        st.warning("Sinyal datası boş görünüyor.")
        st.stop()

    # --- TOP BAR METRICS ---
    st.subheader("Genel Özet")

    total_signals = len(df)
    unique_setups = df["event_type"].nunique() if "event_type" in df.columns else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Toplam Sinyal", f"{total_signals:,}")

    with col2:
        st.metric("Setup Sayısı", unique_setups)

    if "rr" in df.columns:
        with col3:
            st.metric("Ortalama R/R", f"{df['rr'].mean():.2f}")
    else:
        with col3:
            st.metric("Ortalama R/R", "—")

    st.markdown("---")

    # --- FILTER SIDEBAR ---
    st.sidebar.header("Filtreler")

    df_filtered = df.copy()

    # 1) Tarih aralığı
    if "timestamp" in df_filtered.columns:
        min_dt = df_filtered["timestamp"].min().date()
        max_dt = df_filtered["timestamp"].max().date()

        date_range = st.sidebar.date_input(
            "Tarih aralığı",
            (min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df_filtered["timestamp"].dt.date >= start_date) & (
                df_filtered["timestamp"].dt.date <= end_date
            )
            df_filtered = df_filtered[mask]

    # 2) Setup (event_type) filtresi – en çok görülenlerden
    if "event_type" in df_filtered.columns and not df_filtered.empty:
        st.sidebar.markdown("### Setup (event_type)")

        top_events = (
            df_filtered["event_type"]
            .value_counts()
            .head(30)
            .index
            .tolist()
        )

        default_events = top_events[:10] if len(top_events) > 10 else top_events

        selected_events = st.sidebar.multiselect(
            "Event types (en çok görülenler)",
            options=top_events,
            default=default_events,
        )

        if selected_events:
            df_filtered = df_filtered[df_filtered["event_type"].isin(selected_events)]

    # 3) Nearest SR tipi filtresi
    if "nearest_sr_type" in df_filtered.columns:
        st.sidebar.markdown("### Nearest SR Type")

        sr_options = sorted(df_filtered["nearest_sr_type"].dropna().unique().tolist())
        selected_sr = st.sidebar.multiselect(
            "Nearest SR",
            options=sr_options,
            default=sr_options,
        )
        if selected_sr:
            df_filtered = df_filtered[df_filtered["nearest_sr_type"].isin(selected_sr)]

    # 4) Channel / range filtresi
    chan_cols = []
    if "chan_is_up_M30_at_entry" in df_filtered.columns:
        chan_cols.append("chan_is_up_M30_at_entry")
    if "chan_is_down_M30_at_entry" in df_filtered.columns:
        chan_cols.append("chan_is_down_M30_at_entry")
    if "is_range_M30_at_entry" in df_filtered.columns:
        chan_cols.append("is_range_M30_at_entry")

    if chan_cols:
        st.sidebar.markdown("### Trend / Range Flag'leri")
        if "chan_is_up_M30_at_entry" in chan_cols:
            only_up_trend = st.sidebar.checkbox("Sadece kanal YUKARI (chan_is_up_M30_at_entry=1)")
            if only_up_trend:
                df_filtered = df_filtered[df_filtered["chan_is_up_M30_at_entry"] == 1]

        if "chan_is_down_M30_at_entry" in chan_cols:
            only_down_trend = st.sidebar.checkbox("Sadece kanal AŞAĞI (chan_is_down_M30_at_entry=1)")
            if only_down_trend:
                df_filtered = df_filtered[df_filtered["chan_is_down_M30_at_entry"] == 1]

        if "is_range_M30_at_entry" in chan_cols:
            only_range = st.sidebar.checkbox("Sadece RANGE (is_range_M30_at_entry=1)")
            if only_range:
                df_filtered = df_filtered[df_filtered["is_range_M30_at_entry"] == 1]

    st.markdown("### Filtrelenmiş Sinyaller")

    if df_filtered.empty:
        st.warning("Filtrelere uyan sinyal yok.")
        st.stop()

    # --- TABLE VIEW ---

    # Gösterilecek ana kolonlar (olanları al)
    preferred_cols = [
        "timestamp",
        "event_type",
        "direction",
        "entry_price",
        "tp_pips",
        "sl_pips",
        "tp_price",
        "sl_price",
        "rr",
        "n_events",
        "pct_up",
        "pct_down",
        "up_bias",
        "avg_max_up_pips",
        "avg_max_down_pips",
        "nearest_sr_type",
        "nearest_sr_dist_pips",
        "sr_support_price_at_entry",
        "sr_support_strength_at_entry",
        "sr_support_distance_pips_at_entry",
        "sr_resistance_price_at_entry",
        "sr_resistance_strength_at_entry",
        "sr_resistance_distance_pips_at_entry",
        "signal_wave_at_entry",
        "wave_strength_pips_at_entry",
        "wave_duration_bars_at_entry",
        "up_move_pips_at_entry",
        "down_move_pips_at_entry",
        "up_duration_bars_at_entry",
        "down_duration_bars_at_entry",
        "chan_is_up_M30_at_entry",
        "chan_is_down_M30_at_entry",
        "is_range_M30_at_entry",
        "near_lower_chan_M30_at_entry",
        "near_upper_chan_M30_at_entry",
    ]

    available_cols = [c for c in preferred_cols if c in df_filtered.columns]
    df_view = df_filtered[available_cols].copy()

    # timestamp'a göre son sinyaller üste gelsin
    if "timestamp" in df_view.columns:
        df_view = df_view.sort_values("timestamp", ascending=False)

    # Biraz sayısal kolonları yuvarlayalım
    for col in df_view.columns:
        if pd.api.types.is_float_dtype(df_view[col]):
            df_view[col] = df_view[col].round(2)

    st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
    )

    # Küçük bir setup bazlı özet
    st.markdown("---")
    st.subheader("Setup Bazlı Özet (filtrelenmiş set)")

    if "event_type" in df_filtered.columns:
        grp = df_filtered.groupby("event_type").agg(
            n_signals=("event_type", "size"),
            avg_rr=("rr", "mean") if "rr" in df_filtered.columns else ("event_type", "size"),
            avg_up_bias=("up_bias", "mean") if "up_bias" in df_filtered.columns else ("event_type", "size"),
            avg_nearest_sr_dist=("nearest_sr_dist_pips", "mean")
            if "nearest_sr_dist_pips" in df_filtered.columns
            else ("event_type", "size"),
        ).reset_index()

        if "avg_rr" in grp.columns and pd.api.types.is_float_dtype(grp["avg_rr"]):
            grp["avg_rr"] = grp["avg_rr"].round(2)
        if "avg_up_bias" in grp.columns and pd.api.types.is_float_dtype(grp["avg_up_bias"]):
            grp["avg_up_bias"] = grp["avg_up_bias"].round(3)
        if "avg_nearest_sr_dist" in grp.columns and pd.api.types.is_float_dtype(grp["avg_nearest_sr_dist"]):
            grp["avg_nearest_sr_dist"] = grp["avg_nearest_sr_dist"].round(2)

        grp = grp.sort_values("n_signals", ascending=False)
        st.dataframe(grp, use_container_width=True, hide_index=True)
    else:
        st.info("event_type kolonu yok, setup bazlı özet oluşturulamadı.")


if __name__ == "__main__":
    main()