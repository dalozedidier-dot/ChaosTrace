from __future__ import annotations

import pandas as pd
import streamlit as st

from chaostrace.orchestrator.sweep import build_grid, sweep


st.title("ChaosTrace - Invariants & Variantes")

uploaded_file = st.file_uploader("Charger un CSV multivarié", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu", df.head())

    if "time_s" not in df.columns:
        st.error("La colonne 'time_s' est requise.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            window_s = st.slider("Fenêtre (s)", 3.0, 20.0, 10.0)
            emb_lag = st.slider("Lag embedding", 1, 12, 5)
        with col2:
            runs = st.slider("Nombre de configs", 3, 200, 25)
            drop_threshold = st.slider("Drop threshold", 0.10, 0.70, 0.30)

        if st.button("Lancer sweep"):
            cfgs = build_grid(window_s=[window_s], drop_threshold=[drop_threshold], emb_dim=[3], emb_lag=[emb_lag])
            cfgs = cfgs[: int(runs)]
            metrics, timeline = sweep(df, cfgs, seed=7)
            st.success("Sweep terminé")
            st.dataframe(metrics)
            st.line_chart(timeline.set_index("time_s")[["score_invariant", "score_variant"]])
