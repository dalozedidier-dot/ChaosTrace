from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray  # shape: (n_windows, n_channels, window_n)
    y: np.ndarray  # shape: (n_windows,)
    centers: np.ndarray  # indices into original series (ANCHOR index; see make_windows)


def make_windows(
    df: pd.DataFrame,
    *,
    cols: list[str],
    window_n: int,
    stride_n: int,
    horizon_n: int,
    label_col: str = "is_drop",
) -> WindowedDataset:
    """Create causal sliding windows for supervised training/inference.

    Anchor convention:
      - Each window uses ONLY past context and ENDS at the anchor index `a`.
      - The returned `centers` array stores these anchor indices (kept for backwards compat).

    Label convention (early warning):
      - If horizon_n > 0, a window is positive if a drop occurs strictly in the future interval
        (a, a + horizon_n].
      - If horizon_n == 0, a window is positive if label_col is positive at the anchor `a`
        (classic "detect-now" training).

    This supports true early-warning training while remaining causal.
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

    # last usable anchor must leave room for a future horizon
    h = int(max(0, horizon_n))
    last_anchor = n - 1 if h == 0 else (n - 1 - h)
    first_anchor = window_n - 1
    anchors = list(range(first_anchor, last_anchor + 1, stride_n))
    if not anchors:
        raise ValueError("Series is too short for requested window/stride/horizon")

    n_ch = len(cols)
    X = np.empty((len(anchors), n_ch, window_n), dtype=np.float32)
    y = np.empty((len(anchors),), dtype=np.float32)
    centers = np.empty((len(anchors),), dtype=np.int64)

    for i, a in enumerate(anchors):
        s = a - window_n + 1
        e = a + 1  # exclusive
        w = X_raw[s:e, :]
        X[i] = np.transpose(w, (1, 0)).astype(np.float32, copy=False)
        centers[i] = int(a)

        if h > 0:
            fut_s = a + 1
            fut_e = min(n, a + 1 + h)
            y[i] = 1.0 if float(np.nanmax(y_raw[fut_s:fut_e])) > 0.5 else 0.0
        else:
            y[i] = 1.0 if float(y_raw[a]) > 0.5 else 0.0

    # Normalize per-channel robustly (median/IQR)
    med = np.nanmedian(X, axis=(0, 2), keepdims=True)
    q1 = np.nanpercentile(X, 25, axis=(0, 2), keepdims=True)
    q3 = np.nanpercentile(X, 75, axis=(0, 2), keepdims=True)
    iqr = np.maximum(q3 - q1, 1e-6)
    X = (X - med) / iqr

    return WindowedDataset(X=X, y=y, centers=centers)
