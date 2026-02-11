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
            "Training requires optional dependency 'torch'. "
            "Install with: pip install -e '.[dl]'"
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

    # Best effort deterministic behavior
    try:  # pragma: no cover
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover
        pass
    try:  # pragma: no cover
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass

    # Stabilize CPU threading where possible
    try:  # pragma: no cover
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover
        pass


def _augment(x: Any) -> Any:
    """Simple augmentations for contrastive pretraining."""
    torch = _torch()
    # x: (N, C, L)
    noise = 0.02 * torch.randn_like(x)
    scale = 1.0 + 0.02 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device)
    x2 = x * scale + noise

    # random time masking
    if x2.shape[-1] >= 10:
        m = int(max(1, x2.shape[-1] * 0.05))
        start = int(torch.randint(0, x2.shape[-1] - m + 1, (1,)).item())
        x2[:, :, start : start + m] = 0.0
    return x2


def _nt_xent(z1: Any, z2: Any, *, temperature: float = 0.2) -> Any:
    """NT-Xent loss for SimCLR-like contrastive learning."""
    torch = _torch()
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.T) / float(temperature)  # (2N,2N)
    n = z1.shape[0]
    # mask self-similarity
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+n
    pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)
    # denominator: logsumexp over row
    denom = torch.logsumexp(sim, dim=1)
    loss = -(pos - denom).mean()
    return loss


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
    """Train a lightweight DL classifier for early drop warnings.

    Contrastive pretrain is optional and fast.
    Saves:
    - <out_dir>/model.pt
    - <out_dir>/config.json
    """
    torch = _torch()
    _seed_all(int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = make_windows(
        df,
        cols=cols,
        window_n=window_n,
        stride_n=stride_n,
        horizon_n=horizon_n,
        label_col="is_drop",
    )

    # deterministic split
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(ds.y))
    rng.shuffle(idx)
    split = int(round(0.8 * len(idx)))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < len(idx) else idx[:1]

    Xtr = torch.tensor(ds.X[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(ds.y[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.tensor(ds.X[va_idx], dtype=torch.float32, device=device)
    yva = torch.tensor(ds.y[va_idx], dtype=torch.float32, device=device)

    cfg = ModelConfig(n_channels=len(cols), window_n=window_n)
    model = HybridNet(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))

    def batches(Xb: Any, yb: Any):
        n = Xb.shape[0]
        for s in range(0, n, int(batch_size)):
            e = min(n, s + int(batch_size))
            yield Xb[s:e], yb[s:e]

    # optional contrastive pretraining
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
    if best_state is not None:
        torch.save(best_state, model_path)
    else:  # pragma: no cover
        torch.save(model.state_dict(), model_path)

    cfg_path = out_dir / "config.json"
    cfg_path.write_text(
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
