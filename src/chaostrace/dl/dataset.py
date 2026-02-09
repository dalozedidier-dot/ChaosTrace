from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSpec:
    window_n: int
    stride_n: int
    horizon_n: int
    cols: list[str]


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arrs = []
    for c in cols:
        if c not in df.columns:
            arrs.append(np.zeros(len(df), dtype=float))
        else:
            arrs.append(df[c].to_numpy(dtype=float))
    X = np.stack(arrs, axis=0)
    X = np.where(np.isfinite(X), X, 0.0)
    return X


class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        spec: WindowSpec,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        label_col: str = "is_drop",
    ):
        self.spec = spec
        self.X = _ensure_cols(df, spec.cols)
        self.N = self.X.shape[1]

        if label_col in df.columns:
            y = df[label_col].to_numpy(dtype=float)
            self.y = (np.where(np.isfinite(y), y, 0.0) > 0.5).astype(np.int64)
        else:
            self.y = np.zeros(self.N, dtype=np.int64)

        if mean is None:
            mean = self.X.mean(axis=1, keepdims=True)
        if std is None:
            std = self.X.std(axis=1, keepdims=True)
            std = np.where(std > 1e-9, std, 1.0)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.starts = list(range(0, max(0, self.N - spec.window_n - spec.horizon_n), spec.stride_n))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        e = s + self.spec.window_n
        x = self.X[:, s:e].astype(np.float32)
        x = (x - self.mean) / self.std
        h = self.spec.horizon_n
        if h > 0:
            ywin = self.y[e : e + h]
        else:
            ywin = self.y[s:e]
        y = 1.0 if int(np.any(ywin)) == 1 else 0.0
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
