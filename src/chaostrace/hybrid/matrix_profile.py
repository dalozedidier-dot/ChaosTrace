from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatrixProfileOutput:
    window_n: int
    profile: np.ndarray
    score: np.ndarray


def _robust_unit(x: np.ndarray, *, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=float)
    lo = float(np.percentile(x[m], lo_p))
    hi = float(np.percentile(x[m], hi_p))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def compute_matrix_profile(df: pd.DataFrame, *, col: str, window_n: int) -> MatrixProfileOutput:
    """Compute a 1D Matrix Profile based anomaly score.

    Notes
    - Requires optional dependency: `stumpy` (install with `pip install -e '.[mp]'`).
    - Returns a score aligned to the input length (same length as df).

    The score is a robustly normalized version of the matrix profile values.
    Higher score means more discord-like / anomalous.
    """
    if window_n < 4:
        raise ValueError("window_n must be >= 4")

    x = df.get(col)
    if x is None:
        raise ValueError(f"Column not found: {col!r}")
    x = np.asarray(x.to_numpy(dtype=float), dtype=float)
    n = int(len(x))
    if n < window_n + 2:
        return MatrixProfileOutput(window_n=int(window_n), profile=np.full(n, np.nan), score=np.zeros(n, dtype=float))

    try:
        import stumpy  # type: ignore
    except Exception as e:
        raise RuntimeError("Matrix Profile requires optional dependency 'stumpy'. Install with: pip install -e '.[mp]'") from e

    # stumpy.stump returns an array of shape (n - m + 1, 4+), first col is MP.
    mp = stumpy.stump(x, m=int(window_n))
    prof = np.asarray(mp[:, 0], dtype=float)

    # Align to length n by padding at both ends (centered alignment)
    pad_left = int(window_n // 2)
    pad_right = n - (len(prof) + pad_left)
    if pad_right < 0:
        pad_right = 0
    prof_full = np.concatenate([np.full(pad_left, np.nan), prof, np.full(pad_right, np.nan)])

    # Score: robust normalize, treating NaN as 0.
    score = _robust_unit(np.nan_to_num(prof_full, nan=np.nanmedian(prof)), lo_p=1.0, hi_p=99.5)
    score[np.isnan(prof_full)] *= 0.0
    return MatrixProfileOutput(window_n=int(window_n), profile=prof_full, score=score)
