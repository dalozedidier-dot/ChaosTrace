from __future__ import annotations

from pathlib import Path
from chaostrace.data.ingest import load_timeseries
from chaostrace.orchestrator.sweep import SweepConfig, sweep

def test_import_and_sweep():
    df = load_timeseries(Path("test_data/sample_timeseries.csv"))
    cfgs = [
        SweepConfig(window_s=5.0, drop_threshold=0.30, emb_dim=3, emb_lag=5),
        SweepConfig(window_s=10.0, drop_threshold=0.30, emb_dim=4, emb_lag=5),
        SweepConfig(window_s=30.0, drop_threshold=0.25, emb_dim=5, emb_lag=8),
    ]
    metrics, anoms = sweep(df, cfgs=cfgs, seed=1)
    assert len(metrics) == 3
    assert "run_id" in metrics.columns
    assert set(["run_id", "time_s", "score_mean"]).issubset(anoms.columns)
