from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    seconds: float


def estimate_sample_hz(time_s: Any) -> float:
    """Estimate sample frequency in Hz from a time axis.

    Accepts:
      - np.ndarray / list-like of time seconds
      - pandas Series of time seconds
      - pandas DataFrame with a 'time_s' column (preferred) or numeric index

    Returns 1.0 if the estimate cannot be computed reliably.
    """
    try:
        if isinstance(time_s, pd.DataFrame):
            if "time_s" in time_s.columns:
                arr = time_s["time_s"].to_numpy(dtype=float)
            else:
                arr = np.asarray(time_s.index.to_numpy(), dtype=float)
        elif isinstance(time_s, pd.Series):
            arr = time_s.to_numpy(dtype=float)
        else:
            arr = np.asarray(time_s, dtype=float)
    except Exception:
        return 1.0

    if arr.size < 2:
        return 1.0

    dt = np.diff(arr)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return 1.0

    med = float(np.median(dt))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / med


def sliding_windows(df: pd.DataFrame, spec: WindowSpec, sample_hz: float) -> list[tuple[int, int]]:
    win = max(int(round(spec.seconds * sample_hz)), 1)
    return [(s, s + win) for s in range(0, len(df) - win + 1)]
