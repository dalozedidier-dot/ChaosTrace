from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import make_windows
from .model import HybridNet, ModelConfig


def _torch() -> Any:
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Training requires optional dependency 'torch'. Install with: pip install -e '.[dl]'"
        ) from e


def _seed_all(seed: int) -> None:
    """Best-effort determinism across python/numpy/torch."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)

    torch = _torch()
    try:
        torch.manual_seed(s)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(s)
    except Exception:  # pragma: no cover
        pass

    try:  # pragma: no cover
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover
        pass
    try:  # pragma: no cover
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass

    try:  # pragma: no cover
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover
        pass


def _augment(x: Any) -> Any:
    torch = _torch()
    noise = 0.02 * torch.randn_like(x)
    scale = 1.0 + 0.02 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device)
    x2 = x * scale + noise

    if x2.shape[-1] >= 10:
        m = int(max(1, x2.shape[-1] * 0.05))
        start = int(torch.randint(0, x2.shape[-1] - m + 1, (1,)).item())
        x2[:, :, start : start + m] = 0.0
    return x2


def _nt_xent(z1: Any, z2: Any, *, temperature: float = 0.2) -> Any:
    torch = _torch()
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / float(temperature)
    n = z1.shape[0]
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)
    denom = torch.logsumexp(sim, dim=1)
    return -(pos - denom).mean()


def train_hybrid_model(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    cols: list[str],
    window_n: int,
    stride_n: int,
    horizon_n: int,
    contrastive_epochs: int = 0,
    supervised_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    pos_weight: float = 3.0,
    device: str = "cpu",
    seed: int = 7,
) -> Path:
    torch = _torch()
    _seed_all(int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_windows(
        df,
        cols=cols,
        window_n=int(window_n),
        stride_n=int(stride_n),
        horizon_n=int(horizon_n),
    )

    n = X.shape[0]
    split = int(max(1, n * 0.8))
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]

    dev = torch.device(str(device))
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva = torch.tensor(Xva, dtype=torch.float32, device=dev)
    yva = torch.tensor(yva, dtype=torch.float32, device=dev)

    cfg = ModelConfig(in_ch=int(Xtr.shape[1]), hidden=64, emb=32)
    model = HybridNet(cfg).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=dev))

    def batches(Xb: Any, yb: Any):
        idx = torch.randperm(Xb.shape[0], device=Xb.device)
        for i in range(0, Xb.shape[0], int(batch_size)):
            j = idx[i : i + int(batch_size)]
            yield Xb[j], yb[j]

    if int(contrastive_epochs) > 0:
        for _ in range(int(contrastive_epochs)):
            model.train()
            for xb, _yb in batches(Xtr, ytr):
                xb1 = _augment(xb)
                xb2 = _augment(xb)
                _log1, z1 = model(xb1)
                _log2, z2 = model(xb2)
                loss = _nt_xent(z1, z2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    best_loss = float("inf")
    best_state = None

    for _ in range(int(supervised_epochs)):
        model.train()
        for xb, yb in batches(Xtr, ytr):
            logits, _z = model(xb)
            loss = bce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, _z = model(Xva)
            loss = float(bce(logits, yva).item())
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model_path = out_dir / "model.pt"
    torch.save(best_state if best_state is not None else model.state_dict(), model_path)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(cfg),
                "cols": cols,
                "window_n": int(window_n),
                "stride_n": int(stride_n),
                "horizon_n": int(horizon_n),
                "best_val_loss": float(best_loss),
                "seed": int(seed),
                "deterministic_requested": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return model_path
