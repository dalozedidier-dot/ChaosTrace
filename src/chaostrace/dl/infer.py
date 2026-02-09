from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import HybridTinyModel


@dataclass(frozen=True)
class DLInferenceResult:
    score: np.ndarray
    raw_prob: np.ndarray
    cols: list[str]
    window_n: int
    stride_n: int
    horizon_n: int


def _robust01(x: np.ndarray, *, p: float = 95.0) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    if np.all(np.isnan(v)):
        return np.zeros_like(v, dtype=float)
    scale = float(np.nanpercentile(v, p))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(v, dtype=float)
    return np.clip(v / scale, 0.0, 1.0)


def infer_series(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    device: str = "cpu",
    col_override: list[str] | None = None,
) -> DLInferenceResult:
    meta_path = model_dir / "model_meta.json"
    model_path = model_dir / "model.pt"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cols = col_override if col_override is not None else list(meta["cols"])
    window_n = int(meta["window_n"])
    stride_n = int(meta["stride_n"])
    horizon_n = int(meta["horizon_n"])

    mean = np.array(meta["mean"], dtype=np.float32)[:, None]
    std = np.array(meta["std"], dtype=np.float32)[:, None]
    std = np.where(std > 1e-9, std, 1.0).astype(np.float32)

    data = []
    for c in cols:
        if c not in df.columns:
            data.append(np.zeros(len(df), dtype=float))
        else:
            data.append(df[c].to_numpy(dtype=float))
    X = np.stack(data, axis=0).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    X = (X - mean) / std

    model = HybridTinyModel(in_ch=len(cols), **meta.get("arch", {}))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    N = X.shape[1]
    raw = np.full(N, np.nan, dtype=float)
    mid = window_n // 2

    with torch.no_grad():
        for s in range(0, max(0, N - window_n - horizon_n), stride_n):
            e = s + window_n
            xb = torch.from_numpy(X[:, s:e][None, :, :]).to(device)
            logit = model(xb).float().cpu().numpy().reshape(-1)[0]
            prob = float(1.0 / (1.0 + np.exp(-logit)))
            j = s + mid
            if 0 <= j < N:
                raw[j] = prob

    raw_filled = np.where(np.isfinite(raw), raw, 0.0)
    score = _robust01(raw_filled, p=95.0)
    return DLInferenceResult(score=score, raw_prob=raw, cols=list(cols), window_n=window_n, stride_n=stride_n, horizon_n=horizon_n)
