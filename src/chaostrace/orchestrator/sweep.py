from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ALERT_THRESHOLD: float = 0.18

WEIGHTS: dict[str, float] = {
    "null_trace": 0.40,
    "delta_stats": 0.30,
    "markov_drop": 0.20,
    "rqa_light": 0.10,
}


@dataclass(frozen=True, slots=True)
class SweepConfig:
    window_s: int
    drop_threshold: float
    emb_dim: int
    emb_lag: int


def estimate_sample_hz(time_s: np.ndarray) -> float:
    """Estimate sampling frequency from time_s (seconds)."""
    t = np.asarray(time_s, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    hz = 1.0 / float(np.median(dt))
    if not np.isfinite(hz) or hz <= 0:
        return 1.0
    return float(hz)


def _pick_numeric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _sigmoid01(x: np.ndarray, k: float = 4.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-k * x))


def _safe_rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(win, min_periods=max(2, win // 3)).std()
    out = r.to_numpy(dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(win, min_periods=max(2, win // 3)).mean()
    out = r.to_numpy(dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


def _rqa_light_score(series: np.ndarray, emb_dim: int, emb_lag: int, win: int) -> np.ndarray:
    """
    Very light proxy of recurrence/complexity.
    Uses embedding novelty: distance between successive embedded vectors.
    Returns a 0..1 score series.
    """
    x = np.asarray(series, dtype=float)
    x[np.isnan(x)] = 0.0

    d = max(2, int(emb_dim))
    lag = max(1, int(emb_lag))

    needed = (d - 1) * lag + 2
    if x.size < needed:
        return np.zeros_like(x, dtype=float)

    idx = np.arange(0, d * lag, lag, dtype=int)  # [0, lag, 2lag, ...]
    nvec = x.size - idx[-1]
    emb = np.stack([x[i : i + nvec] for i in idx], axis=1)  # (nvec, d)

    diff = np.diff(emb, axis=0)
    novelty = np.linalg.norm(diff, axis=1)  # (nvec-1,)
    novelty_full = np.zeros_like(x, dtype=float)
    start = idx[-1] + 1
    novelty_full[start : start + novelty.size] = novelty

    novelty_z = _zscore(novelty_full)
    novelty_sm = _safe_rolling_mean(novelty_z, win)
    return _sigmoid01(novelty_sm, k=2.8)


def _markov_metrics(low_state: np.ndarray) -> tuple[float, float, float, int]:
    """
    Compute p01, p10, p_low and unique_edges on a small multi-state graph
    built from (foil_state, speed_bin) to allow non-trivial edge counts.
    """
    s = np.asarray(low_state, dtype=int)
    if s.size < 2:
        return 0.0, 0.0, float(np.mean(s) if s.size else 0.0), 0

    # Binary transitions for p01/p10
    a = s[:-1]
    b = s[1:]
    n0 = int(np.sum(a == 0))
    n1 = int(np.sum(a == 1))
    p01 = float(np.sum((a == 0) & (b == 1)) / n0) if n0 > 0 else 0.0
    p10 = float(np.sum((a == 1) & (b == 0)) / n1) if n1 > 0 else 0.0
    p_low = float(np.mean(s))

    # Unique edges on a 6-state graph: foil(0/1) x speed_bin(0/1/2) computed upstream if provided
    # Here we just default to binary edges count when no extra state is provided.
    unique_edges = int(len({(int(x), int(y)) for x, y in zip(a, b)}))
    return p01, p10, p_low, unique_edges


def _component_scores(
    df: pd.DataFrame,
    cfg: SweepConfig,
    hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute component score series and per-run metrics.
    Returns:
      metrics_row_df (1 row) and timeline_df (time_s, score_mean, component scores)
    """
    if "time_s" not in df.columns:
        raise ValueError("Missing required column: time_s")

    time_s = df["time_s"].to_numpy(dtype=float)

    foil_col = _pick_numeric_column(df, ["foil_height_m", "foil_height", "foil_m", "foil"])
    speed_col = _pick_numeric_column(df, ["boat_speed", "boat_speed_mps", "speed_mps", "speed", "v"])

    if foil_col is None:
        foil = np.zeros(df.shape[0], dtype=float)
    else:
        foil = df[foil_col].to_numpy(dtype=float)

    if speed_col is None:
        # Fallback: pick first numeric column other than time_s
        numeric_cols = [
            c
            for c in df.columns
            if c != "time_s" and pd.api.types.is_numeric_dtype(df[c])
        ]
        speed = df[numeric_cols[0]].to_numpy(dtype=float) if numeric_cols else np.zeros(df.shape[0], dtype=float)
        speed_col = numeric_cols[0] if numeric_cols else "speed_fallback"
    else:
        speed = df[speed_col].to_numpy(dtype=float)

    win = max(5, int(round(cfg.window_s * hz)))
    win = min(win, max(5, df.shape[0] // 3))  # keep reasonable

    # Derivatives
    dt = np.gradient(time_s)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nanmedian(dt[np.isfinite(dt) & (dt > 0)]) if np.any(np.isfinite(dt) & (dt > 0)) else 1.0

    d_speed = np.gradient(speed) / dt
    d_foil = np.gradient(foil) / dt

    # Component 1: null_trace proxy (rolling variability)
    v_std = _safe_rolling_std(speed, win)
    f_std = _safe_rolling_std(foil, win)
    null_raw = 0.6 * _zscore(v_std) + 0.4 * _zscore(f_std)
    null_trace = _sigmoid01(_safe_rolling_mean(null_raw, win), k=2.2)

    # Component 2: delta_stats proxy (rolling abs derivative)
    ds = np.abs(d_speed)
    dfh = np.abs(d_foil)
    delta_raw = 0.6 * _zscore(ds) + 0.4 * _zscore(dfh)
    delta_stats = _sigmoid01(_safe_rolling_mean(delta_raw, win), k=2.2)

    # Component 3: markov_drop (low foil state + transition activity)
    low_state = (foil < float(cfg.drop_threshold)).astype(int)
    trans = np.abs(np.diff(low_state, prepend=low_state[:1])).astype(float)
    trans_sm = _safe_rolling_mean(trans, win)
    low_sm = _safe_rolling_mean(low_state.astype(float), win)
    markov_raw = 0.7 * _zscore(low_sm) + 0.3 * _zscore(trans_sm)
    markov_drop = _sigmoid01(markov_raw, k=2.8)

    p01, p10, p_low, unique_edges = _markov_metrics(low_state)

    # Component 4: rqa_light
    rqa_light = _rqa_light_score(speed, cfg.emb_dim, cfg.emb_lag, win)

    # Weighted score
    score_mean = (
        WEIGHTS["null_trace"] * null_trace
        + WEIGHTS["delta_stats"] * delta_stats
        + WEIGHTS["markov_drop"] * markov_drop
        + WEIGHTS["rqa_light"] * rqa_light
    )

    alerts = (score_mean > ALERT_THRESHOLD).astype(int)
    alert_count = int(np.sum(alerts))
    alert_frac = float(np.mean(alerts)) if alerts.size else 0.0

    metrics = {
        "window_s": int(cfg.window_s),
        "drop_threshold": float(cfg.drop_threshold),
        "emb_dim": int(cfg.emb_dim),
        "emb_lag": int(cfg.emb_lag),
        "score_mean": float(np.mean(score_mean)) if score_mean.size else 0.0,
        "null_trace_mean": float(np.mean(null_trace)) if null_trace.size else 0.0,
        "delta_stats_mean": float(np.mean(delta_stats)) if delta_stats.size else 0.0,
        "markov_drop_mean": float(np.mean(markov_drop)) if markov_drop.size else 0.0,
        "rqa_light_mean": float(np.mean(rqa_light)) if rqa_light.size else 0.0,
        "alert_threshold": float(ALERT_THRESHOLD),
        "alert_count": int(alert_count),
        "alert_frac": float(alert_frac),
        "markov_p01": float(p01),
        "markov_p10": float(p10),
        "markov_p_low": float(p_low),
        "foil_state_graph_unique_edges": int(unique_edges),
        "foil_col": str(foil_col) if foil_col is not None else "",
        "speed_col": str(speed_col) if speed_col is not None else "",
        "hz_est": float(hz),
        "window_n": int(win),
    }

    timeline = pd.DataFrame(
        {
            "time_s": time_s,
            "score_mean": score_mean,
            "null_trace": null_trace,
            "delta_stats": delta_stats,
            "markov_drop": markov_drop,
            "rqa_light": rqa_light,
        }
    )

    return pd.DataFrame([metrics]), timeline


def sweep(df: pd.DataFrame, cfgs: list[SweepConfig], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a sweep over a list of configs.
    Returns:
      metrics_df: 1 row per run (config)
      timeline_df: rows = (run_id, time_s, score_mean) + component series
    """
    if "time_s" not in df.columns:
        raise ValueError("Missing required column: time_s")

    df2 = df.copy()
    df2 = df2.sort_values("time_s", kind="mergesort").reset_index(drop=True)

    hz = estimate_sample_hz(df2["time_s"].to_numpy(dtype=float))
    _ = np.random.default_rng(int(seed))  # deterministic hook, even if unused

    metrics_rows: list[pd.DataFrame] = []
    timeline_rows: list[pd.DataFrame] = []

    for run_id, cfg in enumerate(cfgs, start=1):
        mdf, tdf = _component_scores(df2, cfg, hz)
        mdf.insert(0, "run_id", run_id)

        tdf.insert(0, "run_id", run_id)
        metrics_rows.append(mdf)
        timeline_rows.append(tdf)

    metrics_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    timeline_df = pd.concat(timeline_rows, ignore_index=True) if timeline_rows else pd.DataFrame()
    return metrics_df, timeline_df


def build_grid(
    window_s: Iterable[int],
    drop_threshold: Iterable[float],
    emb_dim: Iterable[int],
    emb_lag: Iterable[int],
) -> list[SweepConfig]:
    cfgs: list[SweepConfig] = []
    for w in window_s:
        for th in drop_threshold:
            for d in emb_dim:
                for lag in emb_lag:
                    cfgs.append(
                        SweepConfig(
                            window_s=int(w),
                            drop_threshold=float(th),
                            emb_dim=int(d),
                            emb_lag=int(lag),
                        )
                    )
    return cfgs
