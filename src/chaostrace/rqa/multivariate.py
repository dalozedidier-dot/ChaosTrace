"""Multivariate and Cross-RQA helpers.

- mdRQA: build a multivariate embedding and compute RQA metrics on the joint state.
- cross-RQA: compute recurrence between two embeddings and quantify coupling.

The intent is to capture changes in coupling (e.g., foil_height vs boat_speed)
that may precede or accompany drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.spatial.distance as ssd


@dataclass(frozen=True)
class CrossRQAConfig:
    emb_dim: int = 5
    emb_lag: int = 8
    rr_target: float = 0.02
    theiler_window: Optional[int] = None
    l_min: int = 2
    v_min: int = 2
    max_points: int = 500
    rng_seed: int = 7


def _downsample(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    stride = int(np.ceil(x.shape[0] / max_points))
    return x[::stride]


def _robust_scale(x: np.ndarray) -> np.ndarray:
    """Robust scaling per feature using median and IQR."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    q1 = np.nanpercentile(x, 25, axis=0)
    q3 = np.nanpercentile(x, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    return (x - med) / iqr


def _shannon_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = float(np.sum(counts))
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _line_lengths_diagonal_rect(R: np.ndarray) -> np.ndarray:
    # collect diagonal line lengths (rectangular matrix)
    n, m = R.shape
    lengths = []
    for k in range(-(n - 1), m):
        diag = np.diagonal(R, offset=k)
        if diag.size == 0:
            continue
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _line_lengths_vertical(R: np.ndarray) -> np.ndarray:
    lengths = []
    for j in range(R.shape[1]):
        col = R[:, j]
        run = 0
        for v in col:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _choose_epsilon_from_cross_distances(
    D: np.ndarray, *, rr_target: float, theiler_window: int
) -> float:
    rr_target = float(np.clip(rr_target, 1e-4, 0.25))
    D = np.asarray(D, dtype=float).copy()

    if theiler_window > 0:
        n, m = D.shape
        for i in range(n):
            j0 = max(0, i - theiler_window)
            j1 = min(m, i + theiler_window + 1)
            D[i, j0:j1] = np.inf

    vals = D[np.isfinite(D)]
    if vals.size == 0:
        return float(np.nan)
    return float(np.quantile(vals, rr_target))


def compute_cross_rqa(
    a: np.ndarray,
    b: np.ndarray,
    *,
    cfg: CrossRQAConfig,
) -> Dict[str, Any]:
    """Compute cross-RQA metrics between two 1D series.

    a, b: same length (will be masked for finite values)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 50:
        return {"rr_cross": 0.0, "det_cross": 0.0, "lam_cross": 0.0, "sync_entropy": 0.0}

    n_eff = a.size - (int(cfg.emb_dim) - 1) * int(cfg.emb_lag)
    if n_eff <= 5:
        return {"rr_cross": 0.0, "det_cross": 0.0, "lam_cross": 0.0, "sync_entropy": 0.0}

    idxs = list(range(0, int(cfg.emb_dim) * int(cfg.emb_lag), int(cfg.emb_lag)))
    emb_a = np.stack([a[i : i + n_eff] for i in idxs], axis=1)
    emb_b = np.stack([b[i : i + n_eff] for i in idxs], axis=1)

    emb_a = _downsample(emb_a, int(cfg.max_points))
    emb_b = _downsample(emb_b, int(cfg.max_points))

    # Robust scaling makes cross distances comparable (different units).
    pooled = np.vstack([emb_a, emb_b])
    pooled = _robust_scale(pooled)
    emb_a = pooled[: emb_a.shape[0]]
    emb_b = pooled[emb_a.shape[0] :]

    D = ssd.cdist(emb_a, emb_b, metric="euclidean")

    tw = int(cfg.theiler_window) if cfg.theiler_window is not None else int(cfg.emb_lag) + 1
    eps = _choose_epsilon_from_cross_distances(D, rr_target=float(cfg.rr_target), theiler_window=int(tw))
    if not np.isfinite(eps):
        return {"rr_cross": 0.0, "det_cross": 0.0, "lam_cross": 0.0, "sync_entropy": 0.0, "epsilon_cross": float(eps), "theiler_window": int(tw), "n_points": int(min(D.shape[0], D.shape[1]))}

    R = D <= float(eps)

    denom = float(R.size)
    rr = float(np.mean(R.astype(float))) if denom > 0 else 0.0

    diag = _line_lengths_diagonal_rect(R)
    diag_sel = diag[diag >= int(cfg.l_min)]
    det = float(np.sum(diag_sel) / np.sum(diag)) if np.sum(diag) > 0 else 0.0

    vert = _line_lengths_vertical(R)
    vert_sel = vert[vert >= int(cfg.v_min)]
    lam = float(np.sum(vert_sel) / np.sum(vert)) if np.sum(vert) > 0 else 0.0

    if diag_sel.size:
        _, counts = np.unique(diag_sel, return_counts=True)
        sync_entropy = _shannon_entropy(counts.astype(float))
    else:
        sync_entropy = 0.0

    return {
        "rr_cross": rr,
        "det_cross": det,
        "lam_cross": lam,
        "sync_entropy": sync_entropy,
        "epsilon_cross": float(eps),
        "theiler_window": int(tw),
        "n_points": int(min(R.shape[0], R.shape[1])),
    }
