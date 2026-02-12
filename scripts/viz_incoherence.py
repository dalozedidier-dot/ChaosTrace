#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Plotly est requis pour la visualisation 3D interactive. "
        "Installe le paquet plotly, puis relance. "
        f"Erreur import: {exc!r}"
    ) from exc


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
    max_points_3d: int = 5000
    sphere: bool = False
    sphere_surface: bool = False
    time_path: bool = True
    time_path_mode: str = "level"  # "level" or "single"


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

    blocks: list[np.ndarray] = []
    coord_map: list[tuple[str, int]] = []
    n_eff: int | None = None

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


def pca3(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    m = min(3, U.shape[1])
    coords = U[:, :m] * S[:m]
    if m < 3:
        pad = np.zeros((coords.shape[0], 3 - m), dtype=float)
        coords = np.hstack([coords, pad])
    return coords


def _uniform_subsample_idx(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def to_unit_sphere(coords3: np.ndarray) -> np.ndarray:
    coords3 = np.asarray(coords3, dtype=float)
    nrm = np.linalg.norm(coords3, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return coords3 / nrm


def add_sphere_surface(fig: go.Figure, radius: float = 1.0, steps: int = 30) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, steps)
    v = np.linspace(0.0, np.pi, steps)
    uu, vv = np.meshgrid(u, v)
    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            showscale=False,
            opacity=0.15,
            colorscale=[[0, "rgba(180,180,180,0.25)"], [1, "rgba(180,180,180,0.25)"]],
            hoverinfo="skip",
            name="sphere",
        )
    )


def _iter_level_segments(levels: np.ndarray) -> list[tuple[int, int, int]]:
    lv = np.asarray(levels, dtype=int)
    n = lv.size
    if n <= 1:
        return [(0, max(0, n - 1), int(lv[0]) if n else 0)]
    segs: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, n):
        if int(lv[i]) != int(lv[i - 1]):
            segs.append((start, i - 1, int(lv[start])))
            start = i
    segs.append((start, n - 1, int(lv[start])))
    return segs


def plot_timeseries_with_levels(
    out_png: Path,
    t: np.ndarray,
    v: np.ndarray,
    contrib: np.ndarray,
    lvl: np.ndarray,
    name: str,
    suffix: str,
):
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
    ax2.plot(t, contrib, linewidth=1.0, alpha=0.60, label="incoherence_contrib")
    ax2.set_ylabel("incoherence_contrib")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    labels: list[str] = []
    handles = []
    for handle, lab in list(zip(h1, l1)) + list(zip(h2, l2)):
        if lab not in labels:
            labels.append(lab)
            handles.append(handle)
    ax1.legend(handles, labels, loc="upper right")

    ax1.set_title(f"{name} | instability levels ({suffix})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_embedding_3d_html(
    out_html: Path,
    coords3: np.ndarray,
    lvl: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    value: np.ndarray,
    value_name: str,
    title: str,
    max_points: int,
    sphere: bool,
    sphere_surface: bool,
    time_path: bool,
    time_path_mode: str,
):
    n = coords3.shape[0]
    idx = _uniform_subsample_idx(n, max_points)

    coords3_s = coords3[idx]
    lvl_s = lvl[idx]
    t_s = t[idx]
    score_s = score[idx]
    value_s = value[idx]

    if sphere:
        coords3_s = to_unit_sphere(coords3_s)

    fig = go.Figure()
    if sphere and sphere_surface:
        add_sphere_surface(fig, radius=1.0, steps=30)

    if time_path and coords3_s.shape[0] >= 2:
        if time_path_mode not in {"level", "single"}:
            time_path_mode = "level"

        if time_path_mode == "single":
            custom = np.stack([t_s, score_s, value_s], axis=1)
            fig.add_trace(
                go.Scatter3d(
                    x=coords3_s[:, 0],
                    y=coords3_s[:, 1],
                    z=coords3_s[:, 2],
                    mode="lines",
                    name="time_path",
                    legendgroup="time_path",
                    line={"width": 2, "color": "rgba(120,120,120,0.55)"},
                    customdata=custom,
                    hovertemplate=(
                        "PC1=%{x:.3f}<br>"
                        "PC2=%{y:.3f}<br>"
                        "PC3=%{z:.3f}<br>"
                        "time_s=%{customdata[0]:.3f}<br>"
                        "score=%{customdata[1]:.6f}<br>"
                        f"{value_name}=%{{customdata[2]:.6f}}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )
        else:
            segs = _iter_level_segments(lvl_s)
            shown_level = {0: False, 1: False, 2: False, 3: False}
            for a, b, lev in segs:
                if b <= a:
                    continue

                xs = coords3_s[a : b + 1, 0]
                ys = coords3_s[a : b + 1, 1]
                zs = coords3_s[a : b + 1, 2]
                custom = np.stack([t_s[a : b + 1], score_s[a : b + 1], value_s[a : b + 1]], axis=1)

                lg = f"time_path_{lev}"
                show_legend = not shown_level.get(lev, False)
                shown_level[lev] = True

                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=f"time_path_{LEVEL_NAMES.get(lev, str(lev))}",
                        legendgroup=lg,
                        line={"width": 3, "color": LEVEL_COLORS.get(lev, "#888888")},
                        customdata=custom,
                        hovertemplate=(
                            "PC1=%{x:.3f}<br>"
                            "PC2=%{y:.3f}<br>"
                            "PC3=%{z:.3f}<br>"
                            "time_s=%{customdata[0]:.3f}<br>"
                            "score=%{customdata[1]:.6f}<br>"
                            f"{value_name}=%{{customdata[2]:.6f}}<extra></extra>"
                        ),
                        showlegend=show_legend,
                    )
                )

    for k in range(4):
        m = lvl_s == k
        if not np.any(m):
            continue
        custom = np.stack([t_s[m], score_s[m], value_s[m]], axis=1)
        fig.add_trace(
            go.Scatter3d(
                x=coords3_s[m, 0],
                y=coords3_s[m, 1],
                z=coords3_s[m, 2],
                mode="markers",
                name=LEVEL_NAMES[k],
                legendgroup=f"lvl_{k}",
                marker={"size": 3, "color": LEVEL_COLORS[k], "opacity": 0.85},
                customdata=custom,
                hovertemplate=(
                    "PC1=%{x:.3f}<br>"
                    "PC2=%{y:.3f}<br>"
                    "PC3=%{z:.3f}<br>"
                    "time_s=%{customdata[0]:.3f}<br>"
                    "score=%{customdata[1]:.6f}<br>"
                    f"{value_name}=%{{customdata[2]:.6f}}<extra></extra>"
                ),
            )
        )

    scene = {"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"}
    if sphere:
        scene["aspectmode"] = "cube"
        scene["xaxis"] = {"range": [-1.05, 1.05]}
        scene["yaxis"] = {"range": [-1.05, 1.05]}
        scene["zaxis"] = {"range": [-1.05, 1.05]}
    else:
        scene["aspectmode"] = "data"

    fig.update_layout(
        title=title,
        scene=scene,
        legend={"orientation": "h", "y": 1.02, "x": 0, "groupclick": "togglegroup"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)


def process_run(
    *,
    run_id: int,
    out_base: Path,
    input_csv: Path,
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    anomalies: pd.DataFrame,
    cfg: Cfg,
) -> Path:
    row = metrics.loc[metrics["run_id"] == run_id]
    if row.empty:
        raise ValueError(f"run_id {run_id} absent de metrics.csv")

    emb_dim = int(row.iloc[0].get("emb_dim", 5))
    emb_lag = int(row.iloc[0].get("emb_lag", 8))

    tl = anomalies[anomalies["run_id"] == run_id].reset_index(drop=True)
    if tl.empty:
        raise ValueError(f"run_id {run_id} absent de anomalies.csv")

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
    lvl_global = levels_from_quantiles(instability_global, cfg.q_stable, cfg.q_mild, cfg.q_unstable)

    var_to_cols: dict[str, list[int]] = {v: [] for v in cfg.variables}
    for j, (var_name, _lag_idx) in enumerate(coord_map):
        if var_name in var_to_cols:
            var_to_cols[var_name].append(j)

    contrib: dict[str, np.ndarray] = {}
    levels: dict[str, np.ndarray] = {}
    for v in cfg.variables:
        cols = var_to_cols[v]
        contrib[v] = (np.sum(np.abs(G[:, cols]), axis=1) if cols else np.zeros(X.shape[0])) * infl
        levels[v] = levels_from_quantiles(contrib[v], cfg.q_stable, cfg.q_mild, cfg.q_unstable)

    run_out = out_base / f"run_{run_id}"
    run_out.mkdir(parents=True, exist_ok=True)

    coords3 = pca3(X)

    data = {
        "time_s": time_sub,
        "pc1": coords3[:, 0],
        "pc2": coords3[:, 1],
        "pc3": coords3[:, 2],
        "score": score,
        "grad_norm": grad_norm,
        "knn_mean_dist": knn_mean,
        "knn_inflation": infl,
        "anisotropy": aniso,
        "instability_global": instability_global,
        "level_global": lvl_global,
    }
    for v in cfg.variables:
        data[f"contrib_{v}"] = contrib[v]
        data[f"level_{v}"] = levels[v]
    for j in range(G.shape[1]):
        var_name, lag_idx = coord_map[j]
        data[f"g_{j:02d}_{var_name}_lag{lag_idx}"] = G[:, j]

    pd.DataFrame(data).to_csv(run_out / "incoherence_vectors.csv", index=False, float_format="%.6f")

    suffix = f"run_id={run_id} dim={emb_dim} lag={emb_lag} k={cfg.knn_k}"

    for v in cfg.variables:
        series = df[v].to_numpy(dtype=float)[: X.shape[0]]
        plot_timeseries_with_levels(
            run_out / f"incoherence_{v}.png",
            time_sub,
            series,
            contrib[v],
            levels[v],
            v,
            suffix,
        )

    plot_embedding_3d_html(
        run_out / "incoherence_embedding_3d_global.html",
        coords3,
        lvl_global,
        time_sub,
        score,
        instability_global,
        "instability_global",
        f"Embedding 3D (PCA) | global | {suffix}",
        cfg.max_points_3d,
        False,
        False,
        cfg.time_path,
        cfg.time_path_mode,
    )
    for v in cfg.variables:
        plot_embedding_3d_html(
            run_out / f"incoherence_embedding_3d_{v}.html",
            coords3,
            levels[v],
            time_sub,
            score,
            contrib[v],
            f"contrib_{v}",
            f"Embedding 3D (PCA) | {v} | {suffix}",
            cfg.max_points_3d,
            False,
            False,
            cfg.time_path,
            cfg.time_path_mode,
        )

    if cfg.sphere:
        plot_embedding_3d_html(
            run_out / "incoherence_embedding_3d_global_sphere.html",
            coords3,
            lvl_global,
            time_sub,
            score,
            instability_global,
            "instability_global",
            f"Embedding 3D (unit sphere) | global | {suffix}",
            cfg.max_points_3d,
            True,
            cfg.sphere_surface,
            cfg.time_path,
            cfg.time_path_mode,
        )
        for v in cfg.variables:
            plot_embedding_3d_html(
                run_out / f"incoherence_embedding_3d_{v}_sphere.html",
                coords3,
                levels[v],
                time_sub,
                score,
                contrib[v],
                f"contrib_{v}",
                f"Embedding 3D (unit sphere) | {v} | {suffix}",
                cfg.max_points_3d,
                True,
                cfg.sphere_surface,
                cfg.time_path,
                cfg.time_path_mode,
            )

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
    pd.DataFrame(rows).sort_values("p95_contrib", ascending=False).to_csv(run_out / "incoherence_summary.csv", index=False)

    run_meta = {
        "run_id": int(run_id),
        "emb_dim": int(emb_dim),
        "emb_lag": int(emb_lag),
        "score_mean_metrics": float(row.iloc[0].get("score_mean", float("nan"))),
        "input_csv": str(input_csv),
        "variables": list(cfg.variables),
        "knn_k": int(cfg.knn_k),
        "max_eval_points": int(cfg.max_eval_points),
        "max_points_3d": int(cfg.max_points_3d),
        "sphere": bool(cfg.sphere),
        "sphere_surface": bool(cfg.sphere_surface),
        "time_path": bool(cfg.time_path),
        "time_path_mode": str(cfg.time_path_mode),
    }
    (run_out / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return run_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", type=int, default=None, help="Si fourni, ne génère que run_id")
    ap.add_argument("--vars", default="foil_height_m,boat_speed,wind_shear,wave_height")
    ap.add_argument("--knn-k", type=int, default=25)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-eval-points", type=int, default=300)
    ap.add_argument("--max-points-3d", type=int, default=5000)
    ap.add_argument("--sphere", action="store_true", help="Ajoute en plus une projection sur sphère unité")
    ap.add_argument("--sphere-surface", action="store_true", help="Ajoute la surface de la sphère (si --sphere)")
    ap.add_argument(
        "--time-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relie les points dans l ordre temporel (3D). Utilise --no-time-path pour désactiver.",
    )
    ap.add_argument(
        "--time-path-mode",
        default="level",
        choices=["level", "single"],
        help="Mode du tracé temporel: level (segments colorés) ou single (ligne grise)",
    )
    ap.add_argument("--q-stable", type=float, default=0.50)
    ap.add_argument("--q-mild", type=float, default=0.80)
    ap.add_argument("--q-unstable", type=float, default=0.95)
    ap.add_argument("--copy-best", action="store_true", help="Copie le meilleur run vers out/best")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_base = Path(args.out) if args.out else (run_dir / "viz_incoherence")
    out_base.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    anomalies = pd.read_csv(run_dir / "anomalies.csv")

    if "run_id" not in metrics.columns:
        raise ValueError("metrics.csv ne contient pas la colonne run_id")
    if "score_mean" not in metrics.columns:
        raise ValueError("metrics.csv ne contient pas la colonne score_mean")
    if "run_id" not in anomalies.columns:
        raise ValueError("anomalies.csv ne contient pas la colonne run_id")

    run_ids = sorted(int(x) for x in metrics["run_id"].dropna().unique().tolist())
    if not run_ids:
        raise ValueError("Aucun run_id dans metrics.csv")

    best_row = metrics.loc[metrics["score_mean"].astype(float).idxmax()]
    best_run_id = int(best_row["run_id"])
    best_score = float(best_row["score_mean"])

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
        max_points_3d=int(args.max_points_3d),
        sphere=bool(args.sphere),
        sphere_surface=bool(args.sphere_surface),
        time_path=bool(args.time_path),
        time_path_mode=str(args.time_path_mode),
    )

    run_ids_to_process = [int(args.run_id)] if args.run_id is not None else run_ids

    index_rows = []
    produced_dirs: dict[int, Path] = {}
    for rid in run_ids_to_process:
        out_dir = process_run(
            run_id=rid,
            out_base=out_base,
            input_csv=input_csv,
            df=df,
            metrics=metrics,
            anomalies=anomalies,
            cfg=cfg,
        )
        produced_dirs[rid] = out_dir
        row = metrics.loc[metrics["run_id"] == rid].iloc[0]
        index_rows.append(
            {
                "run_id": rid,
                "score_mean": float(row.get("score_mean", float("nan"))),
                "window_s": float(row.get("window_s", float("nan"))),
                "drop_threshold": float(row.get("drop_threshold", float("nan"))),
                "emb_dim": int(row.get("emb_dim", 0)),
                "emb_lag": int(row.get("emb_lag", 0)),
                "out_dir": str(out_dir),
            }
        )

    pd.DataFrame(index_rows).sort_values("score_mean", ascending=False).to_csv(out_base / "runs_index.csv", index=False)

    (out_base / "best_run.txt").write_text(
        f"best_run_id={best_run_id}\nscore_mean={best_score:.6f}\n",
        encoding="utf-8",
    )

    if args.copy_best:
        src = produced_dirs.get(best_run_id)
        if src is not None:
            best_dir = out_base / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(src, best_dir)
            (best_dir / "BEST_FROM.txt").write_text(
                f"Copie de {src.name}. best_run_id={best_run_id}. score_mean={best_score:.6f}\n",
                encoding="utf-8",
            )

    print(f"OK -> {out_base} | best_run_id={best_run_id} score_mean={best_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
