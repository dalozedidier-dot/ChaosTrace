from __future__ import annotations

import torch
from torch import nn


class HybridTinyModel(nn.Module):
    """Small Conv + Transformer model for window-level anomaly scoring."""

    def __init__(self, in_ch: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> logits: (B,)"""
        z = self.conv(x)
        z = z.permute(2, 0, 1)
        z = self.encoder(z)
        z = z.mean(dim=0)
        logits = self.head(z).squeeze(-1)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding (B, D) for interpretability / contrastive pretrain."""
        z = self.conv(x)
        z = z.permute(2, 0, 1)
        z = self.encoder(z)
        z = z.mean(dim=0)
        return z


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_proj: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = nn.functional.normalize(y, dim=-1)
        return y
