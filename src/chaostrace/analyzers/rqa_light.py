from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .base import AnalyzerResult
from ..phase.embedding import takens_embedding


def rqa_light(
    df: pd.DataFrame,
    col: str = "boat_speed",
    dim: int = 3,
    lag: int = 5,
    eps: float = 0.5,
    *,
    max_points: int = 800,
) -> AnalyzerResult:
    """Lightweight RQA proxy computed on a downsampled Takens embedding.

    This is intentionally simple and bounded in cost to support sweep runs.
    """
    x = df[col].to_numpy(dtype=float)
    X = takens_embedding(x, dim=dim, lag=lag)
    if len(X) < 5:
        timeline = pd.DataFrame({"time_s": df["time_s"], "score": np.zeros(len(df))})
        return AnalyzerResult(
            name=f"rqa_light_{col}",
            metrics={"rr": float("nan"), "det_proxy": float("nan"), "used_points": 0},
            timeline=timeline,
        )

    stride = int(math.ceil(len(X) / max_points)) if len(X) > max_points else 1
    Xs = X[::stride]

    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    R = (D <= eps).astype(int)
    rr = float(R.mean())
    diag = np.diag(R, k=1)
    det = float(diag.mean()) if len(diag) else float("nan")

    score = np.full(len(df), rr, dtype=float)
    timeline = pd.DataFrame({"time_s": df["time_s"], "score": score})
    return AnalyzerResult(
        name=f"rqa_light_{col}",
        metrics={"rr": rr, "det_proxy": det, "used_points": int(len(Xs))},
        timeline=timeline,
    )
