from __future__ import annotations

from pathlib import Path

from chaostrace.data.ingest import load_timeseries
from chaostrace.orchestrator.sweep import build_grid, sweep


def test_import_and_sweep() -> None:
    df = load_timeseries(Path("test_data/sample_timeseries.csv"))
    assert "time_s" in df.columns

    cfgs = build_grid(window_s=[3], drop_threshold=[0.2], emb_dim=[3], emb_lag=[1])
    metrics_df, timeline_df = sweep(df, cfgs, seed=7)

    assert not metrics_df.empty
    assert not timeline_df.empty
    assert metrics_df.loc[0, "run_id"] == 1
    assert timeline_df["run_id"].nunique() == 1
