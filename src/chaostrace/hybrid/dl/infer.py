from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import make_windows
from .model import HybridNet, ModelConfig, sigmoid_np


def _torch() -> Any:
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Inference requires optional dependency 'torch'. "
            "Install with: pip install -e '.[dl]'"
        ) from e


@dataclass(frozen=True)
class DLInferenceOutput:
    score: np.ndarray  # per-timepoint [0,1] score aligned to df
    proba_window: np.ndarray  # per-window probabilities
    centers: np.ndarray  # center indices into df


def infer_series(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    device: str = "cpu",
) -> DLInferenceOutput:
    """Run sliding-window inference and return a per-timepoint score."""
    torch = _torch()

    cfg_path = model_dir / "config.json"
    model_path = model_dir / "model.pt"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError("Expected model_dir to contain config.json and model.pt")

    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    cols = [str(x) for x in cfg_json.get("cols", [])]
    window_n = int(cfg_json.get("window_n", 100))
    stride_n = int(cfg_json.get("stride_n", 10))
    horizon_n = int(cfg_json.get("horizon_n", 0))

    # labels not required at inference time; create dummy if missing
    df2 = df.copy()
    if "is_drop" not in df2.columns:
        df2["is_drop"] = 0.0

    ds = make_windows(
        df2,
        cols=cols,
        window_n=window_n,
        stride_n=stride_n,
        horizon_n=horizon_n,
        label_col="is_drop",
    )

    mcfg = ModelConfig(**cfg_json["model"])
    model = HybridNet(mcfg).to(device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X = torch.tensor(ds.X, dtype=torch.float32, device=device)
    probs = np.empty((X.shape[0],), dtype=float)

    bs = 256
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            e = min(X.shape[0], s + bs)
            logits, _ = model(X[s:e])
            probs[s:e] = sigmoid_np(logits.detach().cpu().numpy())

    score = np.zeros((len(df),), dtype=float)
    score[:] = np.nan
    score[ds.centers] = probs

    # fill gaps by simple nearest-neighbor forward/back fill
    m = np.isfinite(score)
    if np.any(m):
        # forward fill
        last = None
        for i in range(len(score)):
            if np.isfinite(score[i]):
                last = float(score[i])
            elif last is not None:
                score[i] = last
        # backward fill
        last = None
        for i in range(len(score) - 1, -1, -1):
            if np.isfinite(score[i]):
                last = float(score[i])
            elif last is not None:
                score[i] = last
    else:
        score[:] = 0.0

    return DLInferenceOutput(score=score, proba_window=probs, centers=ds.centers)
