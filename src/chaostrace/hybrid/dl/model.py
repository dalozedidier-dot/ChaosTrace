from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelConfig:
    n_channels: int
    window_n: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "This feature requires optional dependency 'torch'. "
            "Install with: pip install -e '.[dl]'"
        )


class HybridNet(nn.Module):
    """Lightweight Conv + Transformer encoder for window classification.

    Outputs:
      - logits: (N,)
      - emb: (N, d_model) pooled representation
    """

    def __init__(self, cfg: ModelConfig) -> None:
        _require_torch()
        super().__init__()
        self.cfg = cfg

        self.conv = nn.Sequential(
            nn.Conv1d(cfg.n_channels, cfg.d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        # x: (N, C, L)
        h = self.conv(x)  # (N, D, L)
        h = h.permute(2, 0, 1)  # (L, N, D)
        h = self.encoder(h)  # (L, N, D)
        h = h.mean(dim=0)  # (N, D)
        h = self.norm(h)
        logits = self.head(h).squeeze(-1)  # (N,)
        return logits, h

    def embed(self, x: "torch.Tensor") -> "torch.Tensor":
        _, h = self.forward(x)
        z = self.proj(h)
        return nn.functional.normalize(z, dim=-1)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))
