from __future__ import annotations

from pathlib import Path

import numpy as np

from chaostrace.data.ingest import load_timeseries
from chaostrace.rqa.advanced import RQAAdvancedConfig, compute_rqa_advanced_from_series
from chaostrace.rqa.multivariate import CrossRQAConfig, compute_cross_rqa


def test_rqa_advanced_basic_metrics() -> None:
    df = load_timeseries(Path("test_data/sample_timeseries.csv"))
    x = df["foil_height_m"].to_numpy(dtype=float)
    cfg = RQAAdvancedConfig(emb_dim=3, emb_lag=5, rr_target=0.03, enable_network=False)
    m = compute_rqa_advanced_from_series(x[:800], config=cfg)

    assert "det" in m and "lam" in m and "rr" in m
    assert 0.0 <= float(m["rr"]) <= 1.0
    assert 0.0 <= float(m["det"]) <= 1.0
    assert 0.0 <= float(m["lam"]) <= 1.0


def test_cross_rqa_runs() -> None:
    df = load_timeseries(Path("test_data/sample_timeseries_1_2_drops.csv"))
    a = df["foil_height_m"].to_numpy(dtype=float)
    b = df["boat_speed"].to_numpy(dtype=float)
    cfg = CrossRQAConfig(emb_dim=3, emb_lag=5, rr_target=0.03)
    out = compute_cross_rqa(a[:1200], b[:1200], cfg=cfg)

    assert "det_cross" in out and "lam_cross" in out and "rr_cross" in out
    assert 0.0 <= float(out["rr_cross"]) <= 1.0
    assert 0.0 <= float(out["det_cross"]) <= 1.0
    assert 0.0 <= float(out["lam_cross"]) <= 1.0
