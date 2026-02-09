from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray  # shape: (n_windows, n_channels, window_n)
    y: np.ndarray  # shape: (n_windows,)
    centers: np.ndarray  # shape: (n_windows,) indices into original series


def make_windows(
    df: pd.DataFrame,
    *,
    cols: list[str],
    window_n: int,
    stride_n: int,
    horizon_n: int,
    label_col: str = "is_drop",
) -> WindowedDataset:
    """Create sliding windows for supervised training.

    Label convention: a window is positive if a drop happens within [center, center+horizon_n].
    This supports "early warning" training when horizon_n > 0.
    """
    if window_n < 5:
        raise ValueError("window_n must be >= 5")
    if stride_n < 1:
        raise ValueError("stride_n must be >= 1")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    X_raw = df[cols].to_numpy(dtype=float)
    y_raw = df[label_col].to_numpy(dtype=float)

    n = len(df)
    starts = list(range(0, max(0, n - window_n + 1), stride_n))
    if not starts:
        raise ValueError("Series is too short for requested window/stride")

    n_ch = len(cols)
    X = np.empty((len(starts), n_ch, window_n), dtype=np.float32)
    y = np.empty((len(starts),), dtype=np.float32)
    centers = np.empty((len(starts),), dtype=np.int64)

    for i, s in enumerate(starts):
        e = s + window_n
        w = X_raw[s:e, :]
        # channel-first for conv1d: (C, L)
        X[i] = np.transpose(w, (1, 0)).astype(np.float32, copy=False)

        c = s + window_n // 2
        centers[i] = c

        h_end = min(n, c + int(max(0, horizon_n)) + 1)
        y[i] = 1.0 if float(np.nanmax(y_raw[c:h_end])) > 0.5 else 0.0

    # Normalize per-channel robustly (median/IQR)
    med = np.nanmedian(X, axis=(0, 2), keepdims=True)
    q1 = np.nanpercentile(X, 25, axis=(0, 2), keepdims=True)
    q3 = np.nanpercentile(X, 75, axis=(0, 2), keepdims=True)
    iqr = np.maximum(q3 - q1, 1e-6)
    X = (X - med) / iqr

    return WindowedDataset(X=X, y=y, centers=centers)
