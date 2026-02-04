from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# A higher default threshold so a stable run doesn't flag everything.
# The effective threshold is also calibrated from the baseline window (see _calibrate_threshold()).
ALERT_THRESHOLD: float = 0.55

# Base weights. Effective weights are made dynamic per-run (scaled by component variance).
BASE_WEIGHTS: dict[str, float] = {
    "null_trace": 0.35,
    "delta_stats": 0.25,
    "markov_drop": 0.25,
    "rqa_light": 0.15,
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


def _safe_rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    r = s.rolling(win, min_periods=max(2, win // 3)).std()
    out = r.to_numpy(dtype=float).copy()  # ensure writable
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    r = s.rolling(win, min_periods=max(2, win // 3)).mean()
    out = r.to_numpy(dtype=float).copy()  # ensure writable
    out[~np.isfinite(out)] = 0.0
    return out


def _robust_center_scale(x: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """
    Robust baseline parameters using median and MAD.
    Returns (center, scale) where scale ~= std.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        # Fallback to std
        std = float(np.std(a))
        scale = std if (np.isfinite(std) and std >= eps) else 1.0
    return med, float(scale)


def _z_from_baseline(x: np.ndarray, base_center: float, base_scale: float) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    z = (a - float(base_center)) / float(base_scale)
    z[~np.isfinite(z)] = 0.0
    return z


def _sigmoid01(x: np.ndarray) -> np.ndarray:
    """
    Logistic squashing to 0..1, but with a built-in shift:
    values around z=2 map near 0.5. Stable baseline (z~0) maps low.
    """
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -20.0, 20.0)
    # Shift by +2 sigmas: stable ~0 => low; significant deviations => high.
    return 1.0 / (1.0 + np.exp(-(z - 2.0)))


def _calibrate_threshold(score: np.ndarray, baseline_mask: np.ndarray) -> float:
    """
    Calibrate an effective threshold from the baseline region.

    Key rule: never let calibration push the threshold so high that spikes become invisible.
    - Use the baseline 99.5% quantile (no extra margin).
    - Enforce a floor at ALERT_THRESHOLD.
    - Cap the threshold to keep discrimination.
    """
    s = np.asarray(score, dtype=float)
    if s.size == 0:
        return ALERT_THRESHOLD

    base = s[baseline_mask]
    base = base[np.isfinite(base)]
    if base.size < 50:
        return ALERT_THRESHOLD

    q = float(np.quantile(base, 0.995))
    thr = max(ALERT_THRESHOLD, q)
    thr = min(thr, 0.90)
    return float(np.clip(thr, 0.0, 1.0))


    base = s[baseline_mask]
    base = base[np.isfinite(base)]
    if base.size < 50:
        return ALERT_THRESHOLD

    q = float(np.quantile(base, 0.995))
    thr = max(ALERT_THRESHOLD, q + 0.10)
    return float(np.clip(thr, 0.0, 1.0))


def _baseline_mask(time_s: np.ndarray, baseline_seconds: float = 30.0) -> np.ndarray:
    """
    Baseline is the earliest portion of the run. This is used to normalize component scores.
    """
    t = np.asarray(time_s, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=bool)

    t0 = float(t[0])
    tmax = float(t[-1])
    # Cap baseline window to 25% of run duration, but at least 10s if available.
    dur = max(0.0, tmax - t0)
    bs = float(baseline_seconds)
    if dur > 0:
        bs = min(bs, max(10.0, 0.25 * dur))
    return (t - t0) <= bs


def _rqa_light_raw(series: np.ndarray, emb_dim: int, emb_lag: int, win: int) -> np.ndarray:
    """
    Very light proxy of recurrence/complexity: novelty in embedded space.
    Returns a raw (unbounded) series, later normalized to baseline z-score.
    """
    x = np.asarray(series, dtype=float).copy()
    x[np.isnan(x)] = 0.0

    d = max(2, int(emb_dim))
    lag = max(1, int(emb_lag))

    needed = (d - 1) * lag + 2
    if x.size < needed:
        return np.zeros_like(x, dtype=float)

    idx = np.arange(0, d * lag, lag, dtype=int)
    nvec = x.size - idx[-1]
    emb = np.stack([x[i : i + nvec] for i in idx], axis=1)  # (nvec, d)

    diff = np.diff(emb, axis=0)
    novelty = np.linalg.norm(diff, axis=1)  # (nvec-1,)

    novelty_full = np.zeros_like(x, dtype=float)
    start = idx[-1] + 1
    novelty_full[start : start + novelty.size] = novelty

    novelty_sm = _safe_rolling_mean(novelty_full, win)
    return novelty_sm


def _state_bins(foil: np.ndarray, speed: np.ndarray, drop_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-state discretization to avoid 'unique_edges=1' unless signal is perfectly constant.
    - foil_bin: 0 low, 1 mid, 2 high
    - speed_bin: 0/1/2 by quantiles
    - state_id: foil_bin*3 + speed_bin in [0..8]
    """
    f = np.asarray(foil, dtype=float)
    v = np.asarray(speed, dtype=float)

    thr = float(drop_threshold)
    mid_thr = thr + max(0.10, 0.5 * thr)  # ensures a 'mid' band even if thr small

    foil_bin = np.zeros_like(f, dtype=int)
    foil_bin[(f >= thr) & (f < mid_thr)] = 1
    foil_bin[f >= mid_thr] = 2

    # Speed bins by quantiles
    v_clean = v[np.isfinite(v)]
    if v_clean.size < 10:
        q1, q2 = float(np.nanmin(v)), float(np.nanmax(v))
    else:
        q1, q2 = float(np.quantile(v_clean, 0.33)), float(np.quantile(v_clean, 0.66))

    speed_bin = np.zeros_like(v, dtype=int)
    speed_bin[(v >= q1) & (v < q2)] = 1
    speed_bin[v >= q2] = 2

    state_id = foil_bin * 3 + speed_bin
    return foil_bin, speed_bin, state_id


def _binary_markov_metrics(state: np.ndarray) -> tuple[float, float, float]:
    s = np.asarray(state, dtype=int)
    if s.size < 2:
        return 0.0, 0.0, float(np.mean(s) if s.size else 0.0)

    a = s[:-1]
    b = s[1:]
    n0 = int(np.sum(a == 0))
    n1 = int(np.sum(a == 1))
    p01 = float(np.sum((a == 0) & (b == 1)) / n0) if n0 > 0 else 0.0
    p10 = float(np.sum((a == 1) & (b == 0)) / n1) if n1 > 0 else 0.0
    frac = float(np.mean(s))
    return p01, p10, frac


def _unique_edges(state_id: np.ndarray) -> int:
    s = np.asarray(state_id, dtype=int)
    if s.size < 2:
        return 0
    edges = {(int(x), int(y)) for x, y in zip(s[:-1], s[1:])}
    return int(len(edges))


def _component_scores(
    df: pd.DataFrame,
    cfg: SweepConfig,
    hz: float,
    baseline_seconds: float = 30.0,
    use_dynamic_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        numeric_cols = [c for c in df.columns if c != "time_s" and pd.api.types.is_numeric_dtype(df[c])]
        speed = df[numeric_cols[0]].to_numpy(dtype=float) if numeric_cols else np.zeros(df.shape[0], dtype=float)
        speed_col = numeric_cols[0] if numeric_cols else "speed_fallback"
    else:
        speed = df[speed_col].to_numpy(dtype=float)

    win = max(5, int(round(cfg.window_s * hz)))
    win = min(win, max(5, df.shape[0] // 3))

    # derivatives
    dt = np.gradient(time_s)
    dt_mask = (~np.isfinite(dt)) | (dt <= 0)
    if np.any(dt_mask):
        good = dt[np.isfinite(dt) & (dt > 0)]
        fill = float(np.nanmedian(good)) if good.size else 1.0
        dt = dt.copy()
        dt[dt_mask] = fill

    d_speed = np.gradient(speed) / dt
    d_foil = np.gradient(foil) / dt

    base_mask = _baseline_mask(time_s, baseline_seconds=baseline_seconds)

    # -------- Raw component features (unbounded) --------
    # null_trace: rolling variability of speed + foil
    v_std = _safe_rolling_std(speed, win)
    f_std = _safe_rolling_std(foil, win)
    null_raw = 0.6 * v_std + 0.4 * f_std
    null_raw = _safe_rolling_mean(null_raw, win)

    # delta_stats: rolling mean abs derivative
    delta_raw = 0.6 * np.abs(d_speed) + 0.4 * np.abs(d_foil)
    delta_raw = _safe_rolling_mean(delta_raw, win)

    # markov_drop: drop persistence + transition activity
    low = (foil < float(cfg.drop_threshold)).astype(int)
    persist_n = max(1, int(round(0.50 * hz)))  # 0.5s persistence to count as a drop state
    low_run = pd.Series(low).rolling(persist_n, min_periods=persist_n).sum().to_numpy(dtype=float)
    drop_state = (low_run == float(persist_n)).astype(int)
    drop_state[: persist_n - 1] = 0  # not enough history

    trans = np.abs(np.diff(drop_state, prepend=drop_state[:1])).astype(float)
    markov_raw = 0.7 * _safe_rolling_mean(drop_state.astype(float), win) + 0.3 * _safe_rolling_mean(trans, win)

    # rqa_light raw novelty
    rqa_raw = _rqa_light_raw(speed, cfg.emb_dim, cfg.emb_lag, win)

    # -------- Baseline normalization -> z -> 0..1 --------
    n_c, n_s = _robust_center_scale(null_raw[base_mask])
    d_c, d_s = _robust_center_scale(delta_raw[base_mask])
    m_c, m_s = _robust_center_scale(markov_raw[base_mask])
    r_c, r_s = _robust_center_scale(rqa_raw[base_mask])

    null_z = _z_from_baseline(null_raw, n_c, n_s)
    delta_z = _z_from_baseline(delta_raw, d_c, d_s)
    markov_z = _z_from_baseline(markov_raw, m_c, m_s)
    rqa_z = _z_from_baseline(rqa_raw, r_c, r_s)

    null_trace = _sigmoid01(null_z)
    delta_stats = _sigmoid01(delta_z)
    markov_drop = _sigmoid01(markov_z)
    rqa_light = _sigmoid01(rqa_z)

    # -------- Dynamic weights (variance-scaled) --------
    comps = {
        "null_trace": null_trace,
        "delta_stats": delta_stats,
        "markov_drop": markov_drop,
        "rqa_light": rqa_light,
    }
    base_w = dict(BASE_WEIGHTS)
    w_eff = base_w.copy()

    if use_dynamic_weights:
        eps = 1e-6
        scale = {k: float(np.nanstd(v)) + eps for k, v in comps.items()}
        w_raw = {k: float(base_w[k]) * scale[k] for k in base_w}
        ssum = float(sum(w_raw.values()))
        if ssum > 0:
            w_eff = {k: w_raw[k] / ssum for k in w_raw}

    score_mean = (
        w_eff["null_trace"] * null_trace
        + w_eff["delta_stats"] * delta_stats
        + w_eff["markov_drop"] * markov_drop
        + w_eff["rqa_light"] * rqa_light
    )

    thr_used = _calibrate_threshold(score_mean, base_mask)

    alerts = (score_mean > thr_used).astype(int)
    alert_count = int(np.sum(alerts))
    alert_frac = float(np.mean(alerts)) if alerts.size else 0.0

    # Markov metrics
    p01, p10, drop_frac = _binary_markov_metrics(drop_state)

    foil_bin, speed_bin, state_id = _state_bins(foil, speed, cfg.drop_threshold)
    uniq_edges = _unique_edges(state_id)

    metrics = {
        "window_s": int(cfg.window_s),
        "drop_threshold": float(cfg.drop_threshold),
        "emb_dim": int(cfg.emb_dim),
        "emb_lag": int(cfg.emb_lag),

        "score_mean": float(np.nanmean(score_mean)) if score_mean.size else 0.0,
        "score_std": float(np.nanstd(score_mean)) if score_mean.size else 0.0,
        "score_min": float(np.nanmin(score_mean)) if score_mean.size else 0.0,
        "score_max": float(np.nanmax(score_mean)) if score_mean.size else 0.0,

        "null_trace_mean": float(np.nanmean(null_trace)) if null_trace.size else 0.0,
        "delta_stats_mean": float(np.nanmean(delta_stats)) if delta_stats.size else 0.0,
        "markov_drop_mean": float(np.nanmean(markov_drop)) if markov_drop.size else 0.0,
        "rqa_light_mean": float(np.nanmean(rqa_light)) if rqa_light.size else 0.0,

        "w_null_trace": float(w_eff["null_trace"]),
        "w_delta_stats": float(w_eff["delta_stats"]),
        "w_markov_drop": float(w_eff["markov_drop"]),
        "w_rqa_light": float(w_eff["rqa_light"]),
        "use_dynamic_weights": bool(use_dynamic_weights),

        "alert_threshold_param": float(ALERT_THRESHOLD),
        "alert_threshold_used": float(thr_used),
        "alert_count": int(alert_count),
        "alert_frac": float(alert_frac),

        "markov_p01": float(p01),
        "markov_p10": float(p10),
        "markov_p_low": float(drop_frac),  # fraction of persistent drop-state
        "foil_state_graph_unique_edges": int(uniq_edges),

        "baseline_seconds": float(baseline_seconds),
        "hz_est": float(hz),
        "window_n": int(win),
        "foil_col": str(foil_col) if foil_col else "",
        "speed_col": str(speed_col) if speed_col else "",
    }

    timeline = pd.DataFrame(
        {
            "time_s": time_s,
            "score_mean": score_mean,
            "alert": alerts,
            "null_trace": null_trace,
            "delta_stats": delta_stats,
            "markov_drop": markov_drop,
            "rqa_light": rqa_light,
            "drop_state": drop_state,
            "foil_bin": foil_bin,
            "speed_bin": speed_bin,
            "state_id": state_id,
        }
    )

    return pd.DataFrame([metrics]), timeline


def sweep(
    df: pd.DataFrame,
    cfgs: list[SweepConfig],
    seed: int = 0,
    baseline_seconds: float = 30.0,
    use_dynamic_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a sweep over a list of configs.
    Returns:
      metrics_df: 1 row per run (config)
      timeline_df: rows = (run_id, time_s, score_mean, ...)
    """
    if "time_s" not in df.columns:
        raise ValueError("Missing required column: time_s")

    df2 = df.copy()
    df2 = df2.sort_values("time_s", kind="mergesort").reset_index(drop=True)

    hz = estimate_sample_hz(df2["time_s"].to_numpy(dtype=float))
    _ = np.random.default_rng(int(seed))  # deterministic hook

    metrics_rows: list[pd.DataFrame] = []
    timeline_rows: list[pd.DataFrame] = []

    for run_id, cfg in enumerate(cfgs, start=1):
        mdf, tdf = _component_scores(
            df2,
            cfg,
            hz,
            baseline_seconds=float(baseline_seconds),
            use_dynamic_weights=bool(use_dynamic_weights),
        )
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
