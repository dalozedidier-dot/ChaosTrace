from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class WindowSpec:
    seconds: float

def estimate_sample_hz(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return 1.0
    med = float(np.median(dt))
    if med <= 0:
        return 1.0
    return 1.0 / med

def sliding_windows(df: pd.DataFrame, spec: WindowSpec, sample_hz: float) -> list[tuple[int, int]]:
    win = max(int(round(spec.seconds * sample_hz)), 1)
    return [(s, s + win) for s in range(0, len(df) - win + 1)]
