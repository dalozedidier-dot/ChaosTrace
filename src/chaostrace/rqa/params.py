"""Helpers to choose defensible embedding and recurrence parameters.

RQA outputs can be sensitive to:
- time delay tau
- embedding dimension m
- recurrence threshold epsilon

This module provides dependency-light estimators suitable for automated
pipelines and CI.

The intent is to give "good enough" defaults that are reproducible and
auditable. They are heuristic and not meant to replace careful analysis.
"""

from __future__ import annotations
from typing import Literal

import numpy as np


TauMethod = Literal["autocorr", "ami"]


def _first_local_min(y: np.ndarray) -> int | None:
    """Return index (>=1) of first local minimum, or None."""
    if y.size < 3:
        return None
    for i in range(1, y.size - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            return i
    return None


def estimate_tau_autocorr(x: np.ndarray, *, max_lag: int = 200) -> int:
    """Estimate tau as the first local minimum of the autocorrelation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 1

    max_lag = int(min(max_lag, x.size - 2))
    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)

    idx = _first_local_min(ac)
    return int(idx) if idx is not None and idx > 0 else 1


def estimate_tau_ami(x: np.ndarray, *, max_lag: int = 200, bins: int = 32) -> int:
    """Estimate tau as the first local minimum of Average Mutual Information.

    This is a lightweight AMI using binned mutual information.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return 1

    max_lag = int(min(max_lag, x.size - 2))
    # Precompute bin edges on the full series for stability.
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1
    edges = np.linspace(lo, hi, int(bins) + 1)

    def mi_for_lag(lag: int) -> float:
        a = x[:-lag]
        b = x[lag:]
        ha, _ = np.histogram(a, bins=edges)
        hb, _ = np.histogram(b, bins=edges)
        hab, _, _ = np.histogram2d(a, b, bins=(edges, edges))

        pa = ha / max(1, ha.sum())
        pb = hb / max(1, hb.sum())
        pab = hab / max(1, hab.sum())

        # Avoid log(0)
        nz = pab > 0
        pa_nz = pa[:, None]
        pb_nz = pb[None, :]
        denom = pa_nz * pb_nz
        nz = nz & (denom > 0)

        return float(np.sum(pab[nz] * np.log(pab[nz] / denom[nz])))

    ami = np.array([mi_for_lag(lag) if lag > 0 else np.nan for lag in range(max_lag + 1)], dtype=float)
    ami[0] = np.nanmax(ami[1:]) if np.isfinite(ami[1:]).any() else 0.0

    idx = _first_local_min(ami[1:])  # ignore 0-lag
    return int(idx + 1) if idx is not None else 1


def estimate_tau(x: np.ndarray, *, method: TauMethod = "ami", max_lag: int = 200) -> int:
    """Estimate tau using the selected method."""
    if method == "autocorr":
        return estimate_tau_autocorr(x, max_lag=max_lag)
    return estimate_tau_ami(x, max_lag=max_lag)


def takens_embedding_1d(x: np.ndarray, *, m: int, tau: int) -> np.ndarray:
    """Takens embedding for a 1D series.

    Returns an array of shape (N_eff, m).
    """
    x = np.asarray(x, dtype=float)
    if m < 1 or tau < 1:
        raise ValueError("m and tau must be >= 1")
    n_eff = x.size - (m - 1) * tau
    if n_eff <= 0:
        raise ValueError("Not enough points for the requested embedding")
    return np.stack([x[i : i + n_eff] for i in range(0, m * tau, tau)], axis=1)


def estimate_m_fnn(
    x: np.ndarray,
    *,
    tau: int,
    m_max: int = 10,
    fnn_target: float = 0.01,
    rtol: float = 10.0,
    atol: float = 2.0,
) -> int:
    """Estimate embedding dimension m with a lightweight False Nearest Neighbors test.

    Parameters
    ----------
    x : series
    tau : time delay
    m_max : maximum dimension to test
    fnn_target : stop when FNN fraction <= this value
    rtol : relative threshold on distance growth
    atol : absolute threshold in units of std(x)

    Returns
    -------
    Estimated m in [2, m_max].
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return 3
    tau = int(max(1, tau))
    m_max = int(max(2, m_max))
    sigma = float(np.std(x))
    if sigma <= 1e-12:
        return 2

    for m in range(1, m_max):
        try:
            emb_m = takens_embedding_1d(x, m=m, tau=tau)
            emb_m1 = takens_embedding_1d(x, m=m + 1, tau=tau)
        except ValueError:
            return int(max(2, m))

        from scipy.spatial import cKDTree
        tree = cKDTree(emb_m)
        dists, idxs = tree.query(emb_m, k=2, workers=-1)
        # nearest neighbor excluding self
        nn = idxs[:, 1]
        d_m = dists[:, 1]
        # Avoid division by 0
        d_m = np.where(d_m <= 1e-12, 1e-12, d_m)

        extra = np.abs(emb_m1[:, -1] - emb_m1[nn, -1])
        ratio = extra / d_m

        fnn = (ratio > rtol) | (extra > atol * sigma)
        fnn_frac = float(np.mean(fnn.astype(float)))

        if fnn_frac <= float(fnn_target) and m + 1 >= 2:
            return int(m + 1)

    return int(m_max)


def fixed_recurrence_rate_epsilon(
    points: np.ndarray,
    *,
    rr_target: float = 0.02,
    rng_seed: int = 7,
    sample_pairs: int = 20000,
) -> float:
    """Pick epsilon so that RR is approximately rr_target (fixed recurrence rate).

    Uses random pair sampling to avoid O(N^2) on large windows.

    Returns a distance quantile used as epsilon.
    """
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n < 10:
        return 0.0

    rr_target = float(np.clip(rr_target, 1e-4, 0.2))
    rng = np.random.default_rng(int(rng_seed))

    # Sample unique pairs (i<j). This is approximate and sufficient for selecting epsilon.
    k = int(min(sample_pairs, n * (n - 1) // 2))
    i = rng.integers(0, n, size=k, endpoint=False)
    j = rng.integers(0, n, size=k, endpoint=False)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return 0.0

    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0

    # RR is proportion of distances <= epsilon, so pick epsilon at rr_target quantile.
    return float(np.quantile(d, rr_target))