from __future__ import annotations

import numpy as np

def takens_embedding(x: np.ndarray, dim: int, lag: int) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n = len(x) - (dim - 1) * lag
    if n <= 1:
        return np.empty((0, dim), dtype=float)
    emb = np.empty((n, dim), dtype=float)
    for i in range(dim):
        emb[:, i] = x[i * lag : i * lag + n]
    return emb
