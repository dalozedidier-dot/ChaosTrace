from __future__ import annotations

import numpy as np
import pandas as pd
from .base import AnalyzerResult
from ..phase.embedding import takens_embedding

def lyapunov_like(df: pd.DataFrame, col: str = "boat_speed", dim: int = 3, lag: int = 5, max_t: int = 20) -> AnalyzerResult:
    x = df[col].to_numpy(dtype=float)
    X = takens_embedding(x, dim=dim, lag=lag)
    if len(X) < max_t + 10:
        timeline = pd.DataFrame({"time_s": df["time_s"], "score": np.zeros(len(df))})
        return AnalyzerResult(name=f"lyap_like_{col}", metrics={"lyap_like": float("nan")}, timeline=timeline)

    n = len(X)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)

    div = []
    for t in range(1, max_t + 1):
        i = np.arange(n - t)
        j = nn[: n - t] + t
        ok = j < n
        i = i[ok]
        j = j[ok]
        d = np.linalg.norm(X[i + t] - X[j], axis=1)
        d0 = np.linalg.norm(X[i] - X[nn[i]], axis=1)
        div.append(np.log((d + 1e-9) / (d0 + 1e-9)).mean())
    div = np.array(div, dtype=float)
    t = np.arange(1, max_t + 1, dtype=float)
    slope = float(np.polyfit(t, div, deg=1)[0])

    score = np.full(len(df), slope, dtype=float)
    timeline = pd.DataFrame({"time_s": df["time_s"], "score": score})
    return AnalyzerResult(name=f"lyap_like_{col}", metrics={"lyap_like": slope}, timeline=timeline)
