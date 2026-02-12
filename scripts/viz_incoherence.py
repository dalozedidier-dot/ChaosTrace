#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LEVEL_NAMES = {0: "stable", 1: "mild", 2: "unstable", 3: "critical"}
LEVEL_COLORS = {0: "#1b9e77", 1: "#d8b365", 2: "#f46d43", 3: "#d73027"}


@dataclass(frozen=True)
class Cfg:
    variables: Tuple[str, ...]
    time_col: str = "time_s"
    score_col: str = "score_mean"
    knn_k: int = 25
    ridge: float = 1e-3
    max_eval_points: int = 300
    q_stable: float = 0.50
    q_mild: float = 0.80
    q_unstable: float = 0.95


def _load_manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_input_path(run_dir: Path, input_path: str | None) -> Path:
    if input_path:
        return Path(input_path)
    mani = _load_manifest(run_dir)
    p = mani.get("params", {}).get("input")
    if not p:
        raise ValueError("input_path absent et manifest.json ne contient pas params.input")
    return Path(p)


def _takens_1d(x: np.ndarray, dim: int, lag: int) -> np.ndarray:
    n = len(x) - (dim - 1) * lag
    if n <= 1:
        return np.empty((0, dim), dtype=float)
    emb = np.empty((n, dim), dtype=float)
    for i in range(dim):
        emb[:, i] = x[i * lag : i * lag + n]
    return emb


def build_multivariate_embedding(
    df: pd.DataFrame, variables: Sequence[str], time_col: str, dim: int, lag: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    for v in variables:
        if v not in df.columns:
            raise ValueError(f"Colonne manquante: {v}")

    t = df[time_col].to_numpy(dtype=float)

    blocks = []
    coord_map: list[tuple[str, int]] = []
    n_eff = None

    for v in variables:
        emb = _takens_1d(df[v].to_numpy(dtype=float), dim=dim, lag=lag)
        n_eff = emb.shape[0] if n_eff is None else min(n_eff, emb.shape[0])
        blocks.append(emb)
        for j in range(dim):
            coord_map.append((str(v), int(j)))

    if n_eff is None or n_eff <= 1:
        return np.empty((0, 0), dtype=float), np.asarray([], dtype=float), []

    X = np.concatenate([b[:n_eff] for b in blocks], axis=1)
    time_sub = t[:n_eff]

    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-6)
    X = (X - med) / iqr

    return X, time_sub, coord_map


def _ridge_solve(DX: np.ndarra
