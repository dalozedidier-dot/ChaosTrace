from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalDriftOutput:
    cols: list[str]
    window_n: int
    baseline_n: int
    baseline_cov: np.ndarray
    drift_raw: np.ndarray
    score: np.ndarray


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def _robust_unit(x: np.ndarray, *, hi_p: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=float)
    hi = float(np.percentile(x[m], hi_p))
    if hi <= 1e-12:
        return np.zeros_like(x, dtype=float)
    y = x / hi
    return np.clip(y, 0.0, 1.0)


def compute_causal_drift(
    df: pd.DataFrame,
    *,
    cols: list[str],
    window_n: int,
    baseline_n: int,
) -> CausalDriftOutput:
    """Compute a lightweight causal-drift proxy based on covariance drift.

    This is not a full Granger-causality graph. It is a fast, interpretable proxy:
    it tracks how the inter-sensor covariance structure changes vs a baseline segment.

    Returns a score aligned to df length (same length), where higher indicates larger drift.
    """
    if window_n < 8:
        raise ValueError("window_n must be >= 8 for causal drift proxy")
    if baseline_n < window_n:
        baseline_n = int(window_n)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for causal drift: {missing}")

    X = df[cols].to_numpy(dtype=float)
    n = int(len(X))
    if n < baseline_n + 2:
        base = np.eye(len(cols), dtype=float)
        return CausalDriftOutput(cols=list(cols), window_n=int(window_n), baseline_n=int(baseline_n), baseline_cov=base,
                                drift_raw=np.zeros(n, dtype=float), score=np.zeros(n, dtype=float))

    # Baseline covariance on initial segment
    Xb = _zscore(X[:baseline_n])
    base_cov = np.cov(Xb, rowvar=False)

    drift = np.zeros(n, dtype=float)
    for i in range(window_n - 1, n):
        w = X[max(0, i - window_n + 1): i + 1]
        wz = _zscore(w)
        cov = np.cov(wz, rowvar=False)
        # Frobenius norm of drift
        d = float(np.linalg.norm(cov - base_cov, ord="fro"))
        drift[i] = d

    score = _robust_unit(drift, hi_p=99.5)
    return CausalDriftOutput(
        cols=list(cols),
        window_n=int(window_n),
        baseline_n=int(baseline_n),
        baseline_cov=np.asarray(base_cov, dtype=float),
        drift_raw=drift,
        score=score,
    )
