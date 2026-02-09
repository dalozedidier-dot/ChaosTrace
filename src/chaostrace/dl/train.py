from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import WindowDataset, WindowSpec
from .model import HybridTinyModel, ProjectionHead


def _augment(x: torch.Tensor, *, jitter_std: float = 0.02, drop_p: float = 0.05) -> torch.Tensor:
    y = x.clone()
    if jitter_std > 0:
        y = y + jitter_std * torch.randn_like(y)
    if drop_p > 0:
        mask = (torch.rand_like(y) < drop_p).float()
        y = y * (1.0 - mask)
    return y


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, *, temp: float = 0.2) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temp
    sim = sim - torch.max(sim, dim=1, keepdim=True).values

    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = nn.functional.cross_entropy(sim, pos)
    return loss


def pretrain_contrastive(
    model: HybridTinyModel,
    loader: DataLoader,
    *,
    epochs: int = 5,
    lr: float = 3e-4,
    device: str = "cpu",
) -> None:
    model.train()
    model.to(device)
    proj = ProjectionHead(d_in=model.head.in_features, d_proj=64).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=lr)

    for _ in range(int(epochs)):
        for xb, _ in loader:
            xb = xb.to(device)
            x1 = _augment(xb)
            x2 = _augment(xb)
            z1 = proj(model.encode(x1))
            z2 = proj(model.encode(x2))
            loss = _nt_xent(z1, z2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def finetune_supervised(
    model: HybridTinyModel,
    loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 3e-4,
    pos_weight: float = 3.0,
    device: str = "cpu",
) -> None:
    model.train()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def train_hybrid_model(
    df,
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
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WindowSpec(window_n=int(window_n), stride_n=int(stride_n), horizon_n=int(horizon_n), cols=list(cols))
    tmp_ds = WindowDataset(df, spec=spec)
    mean = tmp_ds.mean
    std = tmp_ds.std

    ds = WindowDataset(df, spec=spec, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)

    model = HybridTinyModel(in_ch=len(cols), d_model=64, nhead=4, num_layers=2, dropout=0.1)

    if int(contrastive_epochs) > 0:
        pretrain_contrastive(model, loader, epochs=int(contrastive_epochs), lr=lr, device=device)

    finetune_supervised(
        model,
        loader,
        epochs=int(supervised_epochs),
        lr=lr,
        pos_weight=float(pos_weight),
        device=device,
    )

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "cols": list(cols),
        "window_n": int(window_n),
        "stride_n": int(stride_n),
        "horizon_n": int(horizon_n),
        "mean": mean.squeeze(-1).tolist(),
        "std": std.squeeze(-1).tolist(),
        "arch": {"d_model": 64, "nhead": 4, "num_layers": 2},
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return model_path
