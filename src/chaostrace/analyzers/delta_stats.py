from __future__ import annotations

import numpy as np
import pandas as pd
from .base import AnalyzerResult

def delta_stats(df: pd.DataFrame, a: str = "boat_speed", b: str = "foil_height_m") -> AnalyzerResult:
    xa = df[a].to_numpy(dtype=float)
    xb = df[b].to_numpy(dtype=float)
    da = np.diff(xa, prepend=xa[0])
    db = np.diff(xb, prepend=xb[0])
    ra = da / (np.abs(xa) + 1e-6)
    rb = db / (np.abs(xb) + 1e-6)
    score = np.abs(ra - rb)
    timeline = pd.DataFrame({"time_s": df["time_s"], "score": score, "rel_delta_a": ra, "rel_delta_b": rb})
    metrics = {
        "score_p95": float(np.nanpercentile(score, 95)),
        "score_mean": float(np.nanmean(score)),
    }
    return AnalyzerResult(name=f"delta_stats_{a}_vs_{b}", metrics=metrics, timeline=timeline)
