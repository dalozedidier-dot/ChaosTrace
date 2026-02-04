from __future__ import annotations

import numpy as np
import pandas as pd
from .base import AnalyzerResult

def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.zeros_like(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(win - 1, len(x)):
        out[i] = float(np.std(x[i - win + 1 : i + 1]))
    return out

def null_trace(df: pd.DataFrame, col: str = "foil_height_m", win: int = 25, eps: float = 0.01) -> AnalyzerResult:
    x = df[col].to_numpy(dtype=float)
    rs = _rolling_std(x, win=win)
    laminar = (rs < eps).astype(float)
    timeline = pd.DataFrame({"time_s": df["time_s"], "score": laminar, "rolling_std": rs})
    metrics = {
        "laminar_frac": float(np.nanmean(laminar)),
        "rolling_std_median": float(np.nanmedian(rs)),
    }
    return AnalyzerResult(name=f"null_trace_{col}", metrics=metrics, timeline=timeline)
