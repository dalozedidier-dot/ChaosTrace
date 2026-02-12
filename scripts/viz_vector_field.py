#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Plotly est requis pour la visualisation 3D interactive (cone plot). "
        "Installe le paquet plotly, puis relance. "
        f"Erreur import: {exc!r}"
    ) from exc


LEVEL_NAMES = {0: "stable", 1: "mild", 2: "unstable", 3: "critical"}
LEVEL_COLORS = {0: "#1b9e77", 1: "#d8b365", 2: "#f46d43", 3: "#d73027"}


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


def _subsample_idx(n: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or n <= max_items:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_items, dtype=int)


def _segment_levels(lvl: np.ndarray) -> np.ndarray:
    lv = np.asarray(lvl, dtype=int)
    if lv.size < 2:
        return np.asarray([], dtype=int)
    return np.maximum(lv[:-1], lv[1:])


def build_cone_vectorfield(
    coords3: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    value: np.ndarray,
    levels: np.ndarray,
    title: str,
    out_html: Path,
    *,
    max_vectors: int,
    sphere: bool,
    sphere_surface: bool,
    show_points: bool,
):
    coords3 = np.asarray(coords3, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    value = np.asarray(value, dtype=float)
    levels = np.asarray(levels, dtype=int)

    n = coords3.shape[0]
    if n < 3:
        raise ValueError("Pas assez de points pour construire un champ de vecteurs")

    order = np.argsort(t)
    coords3 = coords3[order]
    t = t[order]
    score = score[order]
    value = value[order]
    levels = levels[order]

    if sphere:
        coords3 = to_unit_sphere(coords3)

    pos = coords3[:-1]
    vec = coords3[1:] - coords3[:-1]
    seg_level = _segment_levels(levels)

    idx = _subsample_idx(pos.shape[0], max_vectors)
    pos = pos[idx]
    vec = vec[idx]
    seg_level = seg_level[idx]
    t0 = t[:-1][idx]
    score0 = score[:-1][idx]
    val0 = value[:-1][idx]

    mag = np.linalg.norm(vec, axis=1)
    mag_med = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    sizeref = max(mag_med * 2.0, 1e-6)

    fig = go.Figure()
    if sphere and sphere_surface:
        add_sphere_surface(fig, radius=1.0, steps=30)

    for k in range(4):
        m = seg_level == k
        if not np.any(m):
            continue

        custom = np.stack([t0[m], score0[m], val0[m], mag[m]], axis=1)
        color = LEVEL_COLORS[k]
        fig.add_trace(
            go.Cone(
                x=pos[m, 0],
                y=pos[m, 1],
                z=pos[m, 2],
                u=vec[m, 0],
                v=vec[m, 1],
                w=vec[m, 2],
                anchor="tail",
                sizemode="scaled",
                sizeref=sizeref,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.85,
                name=f"vector_{LEVEL_NAMES[k]}",
                legendgroup=f"vec_{k}",
                customdata=custom,
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
                    "time_s=%{customdata[0]:.3f}<br>"
                    "score=%{customdata[1]:.6f}<br>"
                    "value=%{customdata[2]:.6f}<br>"
                    "||Δ||=%{customdata[3]:.6f}<extra></extra>"
                ),
            )
        )

    if show_points:
        for k in range(4):
            m = levels == k
            if not np.any(m):
                continue
            custom = np.stack([t[m], score[m], value[m]], axis=1)
            fig.add_trace(
                go.Scatter3d(
                    x=coords3[m, 0],
                    y=coords3[m, 1],
                    z=coords3[m, 2],
                    mode="markers",
                    name=f"points_{LEVEL_NAMES[k]}",
                    legendgroup=f"pts_{k}",
                    marker={"size": 2, "color": LEVEL_COLORS[k], "opacity": 0.65},
                    customdata=custom,
                    hovertemplate=(
                        "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
                        "time_s=%{customdata[0]:.3f}<br>"
                        "score=%{customdata[1]:.6f}<br>"
                        "value=%{customdata[2]:.6f}<extra></extra>"
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
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-dir", required=True, help="Dossier contenant run_*/incoherence_vectors.csv")
    ap.add_argument("--max-vectors", type=int, default=800, help="Nombre max de flèches (subsampling uniforme)")
    ap.add_argument("--sphere", action="store_true", help="Projette positions sur sphère unité")
    ap.add_argument("--sphere-surface", action="store_true", help="Ajoute surface de sphère translucide")
    ap.add_argument("--show-points", action="store_true", help="Ajoute aussi les points (markers) en contexte")
    ap.add_argument("--out-subdir", default="vectorfield", help="Sous-dossier dans run_* pour écrire les HTML")
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir)
    if not viz_dir.exists():
        raise SystemExit(f"viz-dir introuvable: {viz_dir}")

    run_dirs = sorted([p for p in viz_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        raise SystemExit("Aucun run_* trouvé dans viz-dir")

    for rd in run_dirs:
        vec_csv = rd / "incoherence_vectors.csv"
        if not vec_csv.exists():
            continue

        df = pd.read_csv(vec_csv)
        required = {"time_s", "pc1", "pc2", "pc3", "score", "level_global", "instability_global"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{rd.name}: colonnes manquantes dans incoherence_vectors.csv: {sorted(missing)}")

        coords3 = df[["pc1", "pc2", "pc3"]].to_numpy(dtype=float)
        t = df["time_s"].to_numpy(dtype=float)
        score = df["score"].to_numpy(dtype=float)

        out_dir = rd / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        build_cone_vectorfield(
            coords3,
            t,
            score,
            df["instability_global"].to_numpy(dtype=float),
            df["level_global"].to_numpy(dtype=int),
            title=f"{rd.name} | Vector field (cones) | global",
            out_html=out_dir / "incoherence_vectorfield_3d_global.html",
            max_vectors=int(args.max_vectors),
            sphere=bool(args.sphere),
            sphere_surface=bool(args.sphere_surface),
            show_points=bool(args.show_points),
        )
        if args.sphere:
            build_cone_vectorfield(
                coords3,
                t,
                score,
                df["instability_global"].to_numpy(dtype=float),
                df["level_global"].to_numpy(dtype=int),
                title=f"{rd.name} | Vector field (cones) | global | unit sphere",
                out_html=out_dir / "incoherence_vectorfield_3d_global_sphere.html",
                max_vectors=int(args.max_vectors),
                sphere=True,
                sphere_surface=bool(args.sphere_surface),
                show_points=bool(args.show_points),
            )

        for col in df.columns:
            if col.startswith("level_"):
                var = col[len("level_") :]
                ccol = f"contrib_{var}"
                if ccol not in df.columns:
                    continue
                build_cone_vectorfield(
                    coords3,
                    t,
                    score,
                    df[ccol].to_numpy(dtype=float),
                    df[col].to_numpy(dtype=int),
                    title=f"{rd.name} | Vector field (cones) | {var}",
                    out_html=out_dir / f"incoherence_vectorfield_3d_{var}.html",
                    max_vectors=int(args.max_vectors),
                    sphere=bool(args.sphere),
                    sphere_surface=bool(args.sphere_surface),
                    show_points=bool(args.show_points),
                )
                if args.sphere:
                    build_cone_vectorfield(
                        coords3,
                        t,
                        score,
                        df[ccol].to_numpy(dtype=float),
                        df[col].to_numpy(dtype=int),
                        title=f"{rd.name} | Vector field (cones) | {var} | unit sphere",
                        out_html=out_dir / f"incoherence_vectorfield_3d_{var}_sphere.html",
                        max_vectors=int(args.max_vectors),
                        sphere=True,
                        sphere_surface=bool(args.sphere_surface),
                        show_points=bool(args.show_points),
                    )

        meta = {
            "viz_dir": str(viz_dir),
            "runs": [p.name for p in run_dirs],
            "max_vectors": int(args.max_vectors),
            "sphere": bool(args.sphere),
            "sphere_surface": bool(args.sphere_surface),
            "show_points": bool(args.show_points),
            "out_subdir": str(args.out_subdir),
        }
        (out_dir / "vectorfield_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"OK: {rd.name} -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
