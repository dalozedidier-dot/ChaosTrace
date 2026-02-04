from __future__ import annotations

import numpy as np
import pandas as pd
from .base import AnalyzerResult

def markov_drop(df: pd.DataFrame, drop_threshold: float = 0.30) -> AnalyzerResult:
    h = df["foil_height_m"].to_numpy(dtype=float)
    drop = (h < drop_threshold).astype(int)

    prev = drop[:-1]
    nxt = drop[1:]

    def p(a: int, b: int) -> float:
        mask = prev == a
        if mask.sum() == 0:
            return float("nan")
        return float((nxt[mask] == b).mean())

    metrics = {
        "drop_frac": float(drop.mean()),
        "p00": p(0, 0),
        "p01": p(0, 1),
        "p10": p(1, 0),
        "p11": p(1, 1),
    }

    win = 25
    score = np.full_like(drop, np.nan, dtype=float)
    for i in range(win, len(drop)):
        seg = drop[i - win : i]
        if seg[:-1].sum() == 0:
            score[i] = float(np.std(h[i - win : i]))
        else:
            score[i] = float(seg.mean())

    timeline = pd.DataFrame({"time_s": df["time_s"], "score": score, "is_drop": drop})
    return AnalyzerResult(name="markov_drop", metrics=metrics, timeline=timeline)
