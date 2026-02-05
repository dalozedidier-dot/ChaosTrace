from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from chaostrace.analyzers.delta_stats import delta_stats
from chaostrace.analyzers.foil_states_graph import foil_state_graph
from chaostrace.analyzers.lyapunov_like import lyapunov_like
from chaostrace.analyzers.markov_drop import markov_drop
from chaostrace.analyzers.null_trace import null_trace
from chaostrace.analyzers.rqa_light import rqa_light
from chaostrace.features.windowing import estimate_sample_hz


ALERT_THRESHOLD = 0.55

# Variant score weights (sum to 1.0)
W_VARIANT_DELTA = 0.25
W_VARIANT_RQA = 0.25
W_VARIANT_LYAP = 0.15
W_VARIANT_MARKOV = 0.35


@dataclass(frozen=True)
class SweepConfig:
    window_s: float
    drop_threshold: float
    emb_dim: int
    emb_lag: int


def build_grid(
    *,
    window_s: Iterable[float],
    drop_threshold: Iterable[float],
    emb_dim: Iterable[int],
    emb_lag: Iterable[int],
) -> list[SweepConfig]:
    """Build a Cartesian product grid of sweep configurations."""
    cfgs: list[SweepConfig] = []
    for w, dt, d, lag in product(window_s, drop_threshold, emb_dim, emb_lag):
        cfgs.append(SweepConfig(window_s=float(w), drop_threshold=float(dt), emb_dim=int(d), emb_lag=int(lag)))
    return cfgs


def _robust_01(x: np.ndarray, *, p: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    with np.errstate(all="ignore"):
        scale = float(np.nanpercentile(x, p)) if np.isfinite(x).any() else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip(x / scale, 0.0, 1.0)


def _fill0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    return x


def sweep(df: pd.DataFrame, cfgs: list[SweepConfig], *, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run analyzers over the full timeseries for each config and aggregate results.

    Returns:
        metrics_df: 1 row per run_id
        timeline_df: 1 row per time step per run_id
    """
    if "time_s" not in df.columns:
        raise ValueError("time_s column required")

    df = df.sort_values("time_s").reset_index(drop=True)
    time_s = df["time_s"].to_numpy(dtype=float)
    hz = estimate_sample_hz(time_s)

    metrics_rows: list[dict[str, float | int]] = []
    timelines: list[pd.DataFrame] = []

    # Deterministic ordering (seed only kept for parity with CLI / future use)
    _ = np.random.default_rng(seed)

    for run_id, cfg in enumerate(cfgs, start=1):
        win_n = max(int(round(cfg.window_s * hz)), 5)

        # Analyzer suite
        res_null = null_trace(df, col="foil_height_m", win=win_n, eps=0.01)
        res_delta = delta_stats(df, a="boat_speed", b="foil_height_m")
        res_markov = markov_drop(df, drop_threshold=cfg.drop_threshold)
        res_rqa = rqa_light(df, col="boat_speed", dim=cfg.emb_dim, lag=cfg.emb_lag, eps=0.5)
        res_lyap = lyapunov_like(df, col="boat_speed", dim=cfg.emb_dim, lag=cfg.emb_lag, max_t=20)
        t2_dyn = float(np.nanmedian(df["foil_height_m"]))
        t2_dyn = max(t2_dyn, cfg.drop_threshold + 0.20)
        t2_dyn = float(np.clip(t2_dyn, cfg.drop_threshold + 0.05, 0.95))
        res_graph = foil_state_graph(df, t1=cfg.drop_threshold, t2=t2_dyn)

        laminar = _fill0(res_null.timeline["score"].to_numpy(dtype=float))  # 0..1, higher = more invariant
        delta01 = _robust_01(_fill0(res_delta.timeline["score"].to_numpy(dtype=float)))
        markov01 = _robust_01(_fill0(res_markov.timeline["score"].to_numpy(dtype=float)))
        rqa_det = res_rqa.metrics.get("det_proxy", float("nan"))
        rqa_rr = res_rqa.metrics.get("rr", float("nan"))
        rqa_inv = float(rqa_det) if np.isfinite(rqa_det) else float(rqa_rr)
        if not np.isfinite(rqa_inv):
            rqa_inv = 0.0
        rqa_inv = float(np.clip(rqa_inv, 0.0, 1.0))
        rqa_inv_series = np.full(len(df), rqa_inv, dtype=float)

        lyap_val = res_lyap.metrics.get("lyap_like", float("nan"))
        lyap_abs = float(abs(lyap_val)) if np.isfinite(lyap_val) else 0.0
        # Smooth mapping to 0..1: small slopes stay small; large slopes saturate.
        lyap01 = float(1.0 - np.exp(-lyap_abs))
        lyap_series = np.full(len(df), lyap01, dtype=float)

        score_invariant = np.clip(
            0.4 * laminar + 0.3 * rqa_inv_series + 0.2 * (1.0 - delta01),
            0.0,
            1.0,
        )

        score_variant = np.clip(
            W_VARIANT_DELTA * delta01
            + W_VARIANT_RQA * (1.0 - rqa_inv_series)
            + W_VARIANT_LYAP * lyap_series
            + W_VARIANT_MARKOV * markov01,
            0.0,
            1.0,
        )

        score_mean = np.clip(0.6 * score_variant + 0.4 * (1.0 - score_invariant), 0.0, 1.0)

        alert_mask = score_mean > ALERT_THRESHOLD
        alert_count = int(np.sum(alert_mask))
        alert_frac = float(alert_count / max(len(score_mean), 1))

        # Metrics row
        m: dict[str, float | int] = {
            "run_id": run_id,
            "hz_est": float(hz),
            "window_s": float(cfg.window_s),
            "window_n": int(win_n),
            "drop_threshold": float(cfg.drop_threshold),
            "emb_dim": int(cfg.emb_dim),
            "emb_lag": int(cfg.emb_lag),
            "score_mean": float(np.mean(score_mean)),
            "score_invariant": float(np.mean(score_invariant)),
            "score_variant": float(np.mean(score_variant)),
            "alert_threshold": float(ALERT_THRESHOLD),
            "alert_count": int(alert_count),
            "alert_frac": float(alert_frac),
            # Analyzer summaries (for audit and quick checks)
            "null_trace_laminar_frac": float(res_null.metrics.get("laminar_frac", float("nan"))),
            "delta_stats_mean": float(res_delta.metrics.get("score_mean", float("nan"))),
            "markov_drop_frac": float(res_markov.metrics.get("drop_frac", float("nan"))),
            "markov_p01": float(res_markov.metrics.get("p01", float("nan"))),
            "markov_p10": float(res_markov.metrics.get("p10", float("nan"))),
            "rqa_rr": float(res_rqa.metrics.get("rr", float("nan"))),
            "rqa_det_proxy": float(res_rqa.metrics.get("det_proxy", float("nan"))),
            "lyap_like": float(res_lyap.metrics.get("lyap_like", float("nan"))),
            "foil_state_unique_edges": float(res_graph.metrics.get("unique_edges", float("nan"))),
            "foil_state_p_low": float(res_graph.metrics.get("p_low", float("nan"))),
        }
        metrics_rows.append(m)

        tl = pd.DataFrame(
            {
                "run_id": run_id,
                "time_s": time_s,
                "score_mean": score_mean,
                "score_invariant": score_invariant,
                "score_variant": score_variant,
                "is_drop": _fill0(res_markov.timeline["is_drop"].to_numpy(dtype=float)),
            }
        )
        timelines.append(tl)

    metrics_df = pd.DataFrame(metrics_rows)
    timeline_df = pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame()
    return metrics_df, timeline_df
