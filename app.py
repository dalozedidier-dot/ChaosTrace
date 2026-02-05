
import streamlit as st, pandas as pd
from chaostrace.orchestrator.sweep import sweep
import plotly.express as px

st.title("ChaosTrace Explorer")

f = st.file_uploader("CSV", type="csv")
if f:
    df = pd.read_csv(f)
    m, tl = sweep(df)
    st.dataframe(m)
    st.plotly_chart(px.line(tl, x="time_s", y=["score_invariant","score_variant"]))
