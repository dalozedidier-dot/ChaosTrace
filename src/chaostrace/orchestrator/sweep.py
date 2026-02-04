from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..analyzers.null_trace import null_trace
from ..analyzers.delta_stats import delta_stats
from ..analyzers.markov_drop import markov_drop
from ..analyzers.foil_states_graph import foil_state_graph
from ..analyzers.rqa_light import rqa_light
from ..analyzers.lyapunov_like import lyapunov_like
from ..features.windowing import WindowSpec, estimate_sample_hz, sliding_windows

@dataclass(frozen=True)
class SweepConfig:
    window_s: float
    drop_threshold: float
    emb_dim: int
    emb_lag: int

def run_one(df_win: pd.DataFrame, cfg: SweepConfig) -> dict[str, Any]:
    results = [
        null_trace(df_win, col="foil_height_m", win=max(int(len(df_win) * 0.2), 5), eps=0.01),
        delta_stats(df_win, a="boat_speed", b="foil_height_m"),
        markov_drop(df_win, drop_threshold=cfg.drop_threshold),
        foil_state_graph(df_win, t1=cfg.drop_threshold, t2=max(cfg.drop_threshold + 0.5, 0.8)),
        rqa_light(df_win, col="boat_speed", dim=cfg.emb_dim, lag=cfg.emb_lag, eps=0.5),
        lyapunov_like(df_win, col="boat_speed", dim=cfg.emb_dim, lag=cfg.emb_lag, max_t=20),
    ]

    metrics: dict[str, float] = {
        "window_s": float(cfg.window_s),
        "drop_threshold": float(cfg.drop_threshold),
        "emb_dim": float(cfg.emb_dim),
        "emb_lag": float(cfg.emb_lag),
    }
    for r in results:
        for k, v in r.metrics.items():
            metrics[f"{r.name}.{k}"] = float(v) if v is not None else float("nan")

    # merge timelines on time_s, compute mean score
    timelines = []
    score_cols = []
    for r in results:
        if "score" in r.timeline.columns:
            t = r.timeline[["time_s", "score"]].rename(columns={"score": f"score__{r.name}"})
            timelines.append(t)
            score_cols.append(f"score__{r.name}")

    merged = timelines[0]
    for t in timelines[1:]:
        merged = merged.merge(t, on="time_s", how="outer")
    merged = merged.sort_values("time_s").reset_index(drop=True)
    merged["score_mean"] = merged[score_cols].mean(axis=1, skipna=True)

    metrics["score_mean_p95"] = float(np.nanpercentile(merged["score_mean"].to_numpy(), 95))
    metrics["score_mean_mean"] = float(np.nanmean(merged["score_mean"].to_numpy()))
    return {"metrics": metrics, "timeline": merged}

def sweep(df: pd.DataFrame, cfgs: list[SweepConfig], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    hz = estimate_sample_hz(df["time_s"].to_numpy(dtype=float))

    all_metrics: list[dict[str, float]] = []
    all_anoms: list[pd.DataFrame] = []

    for run_id, cfg in enumerate(cfgs):
        wins = sliding_windows(df, WindowSpec(seconds=cfg.window_s), sample_hz=hz)
        if not wins:
            continue
        start, end = wins[int(rng.integers(0, len(wins)))]
        dfw = df.iloc[start:end].reset_index(drop=True)

        out = run_one(dfw, cfg=cfg)
        m = out["metrics"]
        m["run_id"] = float(run_id)
        all_metrics.append(m)

        t = out["timeline"][["time_s", "score_mean"]].copy()
        t["run_id"] = run_id
        all_anoms.append(t[["run_id", "time_s", "score_mean"]])

    metrics_df = pd.DataFrame(all_metrics)
    anoms_df = pd.concat(all_anoms, ignore_index=True) if all_anoms else pd.DataFrame(columns=["run_id","time_s","score_mean"])
    return metrics_df, anoms_df
