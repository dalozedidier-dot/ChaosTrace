"""Multivariate and Cross-RQA helpers.

- mdRQA: build a multivariate embedding and compute RQA metrics on the joint state.
- cross-RQA: compute recurrence between two embeddings and quantify coupling.

The intent is to capture changes in coupling (e.g., foil_height vs boat_speed)
that may precede or accompany drops.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy.spatial.distance as ssd

from chaostrace.rqa.params import fixed_recurrence_rate_epsilon


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


def takens_embedding_multivariate(x: np.ndarray, *, m: int, tau: int) -> np.ndarray:
    """Takens embedding for multivariate series.

    x: array shape (N, C)
    returns: (N_eff, m*C)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D (N, C)")
    if m < 1 or tau < 1:
        raise ValueError("m and tau must be >= 1")

    n_eff = x.shape[0] - (m - 1) * tau
    if n_eff <= 0:
        raise ValueError("Not enough points for requested embedding")

    chunks = [x[i : i + n_eff] for i in range(0, m * tau, tau)]
    return np.concatenate(chunks, axis=1)


def _line_lengths_diagonal_rect(R: np.ndarray) -> np.ndarray:
    """Diagonal line lengths on a rectangular boolean matrix."""
    n, m = R.shape
    lengths: list[int] = []
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
    lengths: list[int] = []
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


def _shannon_entropy(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / float(np.sum(c))
    return float(-np.sum(p * np.log(p)))


def compute_md_rqa(
    x: np.ndarray,
    *,
    cfg: CrossRQAConfig,
) -> Dict[str, Any]:
    """Compute mdRQA metrics on a multivariate series."""
    emb = takens_embedding_multivariate(x, m=int(cfg.emb_dim), tau=int(cfg.emb_lag))
    emb = _downsample(emb, int(cfg.max_points))
    eps = fixed_recurrence_rate_epsilon(emb, rr_target=float(cfg.rr_target), rng_seed=int(cfg.rng_seed))
    tw = int(cfg.theiler_window) if cfg.theiler_window is not None else int(cfg.emb_dim) * int(cfg.emb_lag)

    d = ssd.squareform(ssd.pdist(emb))
    R = d <= float(eps)
    np.fill_diagonal(R, False)
    if tw > 0:
        for k in range(-tw, tw + 1):
            if k == 0:
                continue
            R &= ~np.eye(R.shape[0], k=k, dtype=bool)

    denom = float(R.size - R.shape[0]) if R.shape[0] > 1 else 1.0
    rr = float(np.sum(R)) / denom

    diag = _line_lengths_diagonal_rect(R)
    diag_sel = diag[diag >= int(cfg.l_min)]
    det = float(np.sum(diag_sel) / np.sum(diag)) if np.sum(diag) > 0 else 0.0

    vert = _line_lengths_vertical(R)
    vert_sel = vert[vert >= int(cfg.v_min)]
    lam = float(np.sum(vert_sel) / np.sum(vert)) if np.sum(vert) > 0 else 0.0

    return {
        "rr_md": rr,
        "det_md": det,
        "lam_md": lam,
        "epsilon_md": float(eps),
        "theiler_window": int(tw),
        "n_points": int(R.shape[0]),
    }


def compute_cross_rqa(
    a: np.ndarray,
    b: np.ndarray,
    *,
    cfg: CrossRQAConfig,
) -> Dict[str, Any]:
    """Compute cross-RQA metrics between two 1D series.

    a, b: same length
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 50:
        return {"det_cross": 0.0, "lam_cross": 0.0, "sync_entropy": 0.0}

    # Embeddings (same m,tau for both)
    n_eff = a.size - (int(cfg.emb_dim) - 1) * int(cfg.emb_lag)
    if n_eff <= 5:
        return {"det_cross": 0.0, "lam_cross": 0.0, "sync_entropy": 0.0}

    emb_a = np.stack([a[i : i + n_eff] for i in range(0, int(cfg.emb_dim) * int(cfg.emb_lag), int(cfg.emb_lag))], axis=1)
    emb_b = np.stack([b[i : i + n_eff] for i in range(0, int(cfg.emb_dim) * int(cfg.emb_lag), int(cfg.emb_lag))], axis=1)

    emb_a = _downsample(emb_a, int(cfg.max_points))
    emb_b = _downsample(emb_b, int(cfg.max_points))

    # Choose epsilon from the pooled distances (fixed RR)
    pooled = np.vstack([emb_a, emb_b])
    eps = fixed_recurrence_rate_epsilon(pooled, rr_target=float(cfg.rr_target), rng_seed=int(cfg.rng_seed))

    D = ssd.cdist(emb_a, emb_b, metric="euclidean")
    R = D <= float(eps)

    tw = int(cfg.theiler_window) if cfg.theiler_window is not None else int(cfg.emb_lag) + 1
    if tw > 0:
        # Exclude near the main diagonal band (time-matched trivial correlation)
        n, m = R.shape
        for i in range(n):
            j0 = max(0, i - tw)
            j1 = min(m, i + tw + 1)
            R[i, j0:j1] = False

    denom = float(R.size)
    rr = float(np.mean(R.astype(float))) if denom > 0 else 0.0

    diag = _line_lengths_diagonal_rect(R)
    diag_sel = diag[diag >= int(cfg.l_min)]
    det = float(np.sum(diag_sel) / np.sum(diag)) if np.sum(diag) > 0 else 0.0

    vert = _line_lengths_vertical(R)
    vert_sel = vert[vert >= int(cfg.v_min)]
    lam = float(np.sum(vert_sel) / np.sum(vert)) if np.sum(vert) > 0 else 0.0

    # Synchronisation entropy: entropy of diagonal line length distribution
    if diag_sel.size:
        uniq, counts = np.unique(diag_sel, return_counts=True)
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
