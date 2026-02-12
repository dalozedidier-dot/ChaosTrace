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


def _ridge_solve(DX: np.ndarray, dy: np.ndarray, ridge: float) -> np.ndarray:
    d = DX.shape[1]
    A = DX.T @ DX
    A.flat[:: d + 1] += float(ridge)
    b = DX.T @ dy
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def estimate_local_gradients(
    X: np.ndarray, score: np.ndarray, knn_k: int, ridge: float, max_eval_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    s = np.asarray(score, dtype=float)
    n, d = X.shape
    if n <= 5 or d == 0:
        return np.zeros((n, d)), np.zeros(n), np.zeros(n)

    k = int(np.clip(knn_k, 5, max(5, n - 1)))

    stride = 1
    if max_eval_points and max_eval_points > 0 and n > max_eval_points:
        stride = int(np.ceil(n / max_eval_points))
        stride = max(1, stride)

    eval_idx = np.arange(0, n, stride, dtype=int)

    tree = cKDTree(X)
    dist, idx = tree.query(X[eval_idx], k=k + 1, workers=-1)

    Gs = np.zeros((eval_idx.size, d), dtype=float)
    knn_mean_s = np.zeros(eval_idx.size, dtype=float)
    aniso_s = np.zeros(eval_idx.size, dtype=float)

    for ii, i in enumerate(eval_idx):
        neigh = idx[ii, 1:]
        di = dist[ii, 1:]
        m = np.isfinite(di)
        neigh = neigh[m]
        di = di[m]
        if neigh.size < 5:
            continue

        knn_mean_s[ii] = float(np.mean(di))

        Xi = X[i]
        DX = X[neigh] - Xi
        dy = s[neigh] - s[i]

        m2 = np.isfinite(dy) & np.all(np.isfinite(DX), axis=1)
        DX = DX[m2]
        dy = dy[m2]
        if DX.shape[0] < 5:
            continue

        g = _ridge_solve(DX, dy, ridge=ridge)
        if np.all(np.isfinite(g)):
            Gs[ii] = g

        C = (DX.T @ DX) / max(DX.shape[0] - 1, 1)
        try:
            w = np.linalg.eigvalsh(C)
            w = np.maximum(w, 0.0)
            tr = float(np.sum(w))
            aniso_s[ii] = float(np.max(w) / tr) if tr > 1e-12 else 0.0
        except np.linalg.LinAlgError:
            aniso_s[ii] = 0.0

    if stride > 1 and eval_idx.size >= 2:
        full = np.arange(n, dtype=float)
        xi = eval_idx.astype(float)
        G = np.empty((n, d), dtype=float)
        for j in range(d):
            G[:, j] = np.interp(full, xi, Gs[:, j])
        knn_mean = np.interp(full, xi, knn_mean_s)
        aniso = np.interp(full, xi, aniso_s)
        return G, knn_mean, aniso

    G = np.zeros((n, d), dtype=float)
    knn_mean = np.zeros(n, dtype=float)
    aniso = np.zeros(n, dtype=float)
    G[eval_idx] = Gs
    knn_mean[eval_idx] = knn_mean_s
    aniso[eval_idx] = aniso_s
    return G, knn_mean, aniso


def levels_from_quantiles(x: np.ndarray, q50: float, q80: float, q95: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xf = x[np.isfinite(x)]
    if xf.size < 20:
        return np.zeros_like(x, dtype=int)
    t0, t1, t2 = np.quantile(xf, [q50, q80, q95])
    lvl = np.zeros_like(x, dtype=int)
    lvl[x >= t0] = 1
    lvl[x >= t1] = 2
    lvl[x >= t2] = 3
    return lvl


def plot_var(out_png: Path, t: np.ndarray, v: np.ndarray, inst: np.ndarray, lvl: np.ndarray, name: str, suffix: str):
    fig, ax1 = plt.subplots()
    ax1.plot(t, v, linewidth=1.2, label=name)
    ax1.set_xlabel("time_s")
    ax1.set_ylabel(name)

    for k in range(4):
        m = lvl == k
        if np.any(m):
            ax1.scatter(
                t[m],
                v[m],
                s=8,
                alpha=0.85,
                color=LEVEL_COLORS[k],
                label=LEVEL_NAMES[k],
                zorder=3,
            )

    ax2 = ax1.twinx()
    ax2.plot(t, inst, linewidth=1.0, alpha=0.60, label="instability_index")
    ax2.set_ylabel("instability_index")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    labels = []
    handles = []
    for h, label in list(zip(h1, l1)) + list(zip(h2, l2)):
        if label not in labels:
            labels.append(label)
            handles.append(h)
    ax1.legend(handles, labels, loc="upper right")

    ax1.set_title(f"{name} | instability levels ({suffix})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--vars", default="foil_height_m,boat_speed,wind_shear,wave_height")
    ap.add_argument("--knn-k", type=int, default=25)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-eval-points", type=int, default=300)
    ap.add_argument("--q-stable", type=float, default=0.50)
    ap.add_argument("--q-mild", type=float, default=0.80)
    ap.add_argument("--q-unstable", type=float, default=0.95)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else (run_dir / "viz_incoherence")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    tl_all = pd.read_csv(run_dir / "anomalies.csv")

    if args.run_id is None:
        mani = _load_manifest(run_dir)
        rid = mani.get("params", {}).get("run_choice")
        run_id = int(rid) if rid is not None else int(metrics.sort_values("run_id").iloc[0]["run_id"])
    else:
        run_id = int(args.run_id)

    row = metrics.loc[metrics["run_id"] == run_id]
    if row.empty:
        raise ValueError(f"run_id {run_id} absent de metrics.csv")

    emb_dim = int(row.iloc[0].get("emb_dim", 5))
    emb_lag = int(row.iloc[0].get("emb_lag", 8))

    tl = tl_all[tl_all["run_id"] == run_id].reset_index(drop=True)
    if tl.empty:
        raise ValueError(f"run_id {run_id} absent de anomalies.csv")

    input_csv = _resolve_input_path(run_dir, args.input)
    df = pd.read_csv(input_csv)

    variables = tuple([x.strip() for x in str(args.vars).split(",") if x.strip()])
    cfg = Cfg(
        variables=variables,
        knn_k=int(args.knn_k),
        ridge=float(args.ridge),
        max_eval_points=int(args.max_eval_points),
        q_stable=float(args.q_stable),
        q_mild=float(args.q_mild),
        q_unstable=float(args.q_unstable),
    )

    X, time_sub, coord_map = build_multivariate_embedding(df, cfg.variables, cfg.time_col, emb_dim, emb_lag)
    if X.size == 0:
        raise ValueError("Pas assez de points pour construire l embedding")

    score = tl[cfg.score_col].to_numpy(dtype=float)[: X.shape[0]]

    G, knn_mean, aniso = estimate_local_gradients(X, score, cfg.knn_k, cfg.ridge, cfg.max_eval_points)
    grad_norm = np.linalg.norm(G, axis=1)

    base = np.nanmedian(knn_mean[np.isfinite(knn_mean)]) if np.isfinite(knn_mean).any() else 1.0
    infl = np.where(base > 1e-12, knn_mean / base, 1.0)
    infl = np.where(np.isfinite(infl), infl, 1.0)

    instability_global = grad_norm * infl * (1.0 + aniso)

    var_to_cols = {v: [] for v in cfg.variables}
    for j, (v, lag_idx) in enumerate(coord_map):
        if v in var_to_cols:
            var_to_cols[v].append(j)

    contrib = {}
    levels = {}
    for v in cfg.variables:
        cols = var_to_cols[v]
        contrib[v] = (np.sum(np.abs(G[:, cols]), axis=1) if cols else np.zeros(X.shape[0])) * infl
        levels[v] = levels_from_quantiles(contrib[v], cfg.q_stable, cfg.q_mild, cfg.q_unstable)

    out_csv = out_dir / "incoherence_vectors.csv"
    data = {
        "time_s": time_sub,
        "score": score,
        "grad_norm": grad_norm,
        "knn_mean_dist": knn_mean,
        "knn_inflation": infl,
        "anisotropy": aniso,
        "instability": instability_global,
    }
    for v in cfg.variables:
        data[f"contrib_{v}"] = contrib[v]
        data[f"level_{v}"] = levels[v]
    for j in range(G.shape[1]):
        v, lag_idx = coord_map[j]
        data[f"g_{j:02d}_{v}_lag{lag_idx}"] = G[:, j]
    pd.DataFrame(data).to_csv(out_csv, index=False, float_format="%.6f")

    suffix = f"run_id={run_id} dim={emb_dim} lag={emb_lag} k={cfg.knn_k}"
    for v in cfg.variables:
        series = df[v].to_numpy(dtype=float)[: X.shape[0]]
        plot_var(out_dir / f"incoherence_{v}.png", time_sub, series, contrib[v], levels[v], v, suffix)

    rows = []
    for v in cfg.variables:
        x = contrib[v]
        rows.append(
            {
                "variable": v,
                "mean_contrib": float(np.nanmean(x)),
                "p95_contrib": float(np.nanpercentile(x[np.isfinite(x)], 95)) if np.isfinite(x).any() else float("nan"),
                "critical_frac": float(np.mean(levels[v] == 3)),
            }
        )
    pd.DataFrame(rows).sort_values("p95_contrib", ascending=False).to_csv(
        out_dir / "incoherence_summary.csv", index=False
    )

    print(f"OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
