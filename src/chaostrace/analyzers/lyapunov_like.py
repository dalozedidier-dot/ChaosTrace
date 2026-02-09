from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .base import AnalyzerResult
from ..phase.embedding import takens_embedding


def lyapunov_like(
    df: pd.DataFrame,
    col: str = "boat_speed",
    dim: int = 3,
    lag: int = 5,
    max_t: int = 20,
    *,
    max_points: int = 800,
) -> AnalyzerResult:
    """Lyapunov-like slope proxy (Rosenstein-style), bounded via downsampling.

    The goal is to provide a cheap stability indicator for sweeps, not an exact Lyapunov exponent.
    """
    x = df[col].to_numpy(dtype=float)
    X = takens_embedding(x, dim=dim, lag=lag)
    if len(X) < max_t + 10:
        timeline = pd.DataFrame({"time_s": df["time_s"], "score": np.zeros(len(df))})
        return AnalyzerResult(
            name=f"lyap_like_{col}",
            metrics={"lyap_like": float("nan"), "used_points": 0},
            timeline=timeline,
        )

    stride = int(math.ceil(len(X) / max_points)) if len(X) > max_points else 1
    Xs = X[::stride]

    n = len(Xs)
    if n < max_t + 10:
        timeline = pd.DataFrame({"time_s": df["time_s"], "score": np.zeros(len(df))})
        return AnalyzerResult(
            name=f"lyap_like_{col}",
            metrics={"lyap_like": float("nan"), "used_points": int(n)},
            timeline=timeline,
        )

    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)

    div = []
    for t in range(1, max_t + 1):
        i = np.arange(n - t)
        j = nn[: n - t] + t
        ok = j < n
        i = i[ok]
        j = j[ok]
        d = np.linalg.norm(Xs[i + t] - Xs[j], axis=1)
        d0 = np.linalg.norm(Xs[i] - Xs[nn[i]], axis=1)
        div.append(np.log((d + 1e-9) / (d0 + 1e-9)).mean())

    div = np.array(div, dtype=float)
    tt = np.arange(1, max_t + 1, dtype=float)
    slope = float(np.polyfit(tt, div, deg=1)[0])

    score = np.full(len(df), slope, dtype=float)
    timeline = pd.DataFrame({"time_s": df["time_s"], "score": score})
    return AnalyzerResult(
        name=f"lyap_like_{col}",
        metrics={"lyap_like": slope, "used_points": int(len(Xs))},
        timeline=timeline,
    )
