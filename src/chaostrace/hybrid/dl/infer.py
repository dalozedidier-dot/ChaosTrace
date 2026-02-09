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
    centers: np.ndarray  # anchor indices into df (window END indices)


def infer_series(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    device: str = "cpu",
) -> DLInferenceOutput:
    """Run causal sliding-window inference and return a per-timepoint score.

    The score is aligned to the window anchor index (the END of each past-only window).

    IMPORTANT: the returned per-timepoint score is filled CAUSALLY (forward fill only).
    That way, scores at time t never incorporate windows whose anchor is > t.
    """
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

    # Sparse per-anchor probabilities
    score = np.zeros((len(df),), dtype=float)
    anchor_mask = np.zeros((len(df),), dtype=bool)
    score[ds.centers] = probs
    anchor_mask[ds.centers] = True

    # Causal fill: forward only. Before first anchor -> 0.0.
    last = 0.0
    for i in range(len(score)):
        if anchor_mask[i]:
            last = float(score[i])
        else:
            score[i] = last

    return DLInferenceOutput(score=score, proba_window=probs, centers=ds.centers)
