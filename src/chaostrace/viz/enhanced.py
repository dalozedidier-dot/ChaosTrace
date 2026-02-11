from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnhancedVizArtifacts:
    out_dir: Path
    phase3d_html: Path
    recurrence_html: Path
    rqa_heatmap_html: Path
    dashboard_html: Path
    metadata_json: Path


def _lazy_plotly() -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for enhanced visualizations. "
            "Install it with: pip install -r requirements_viz.txt"
        ) from e


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_html(fig: Any, path: Path) -> None:
    path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def _dark_layout_kwargs() -> Dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "#06070a",
        "plot_bgcolor": "#06070a",
        "font": {"family": "system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif", "size": 13},
    }


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def _takens_embedding_1d(x: np.ndarray, *, dim: int, lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if dim <= 1 or lag <= 0:
        raise ValueError("dim must be >=2 and lag must be >=1")
    n = x.size
    m = n - (dim - 1) * lag
    if m <= 0:
        return np.zeros((0, dim), dtype=float)
    out = np.empty((m, dim), dtype=float)
    for i in range(dim):
        out[:, i] = x[i * lag : i * lag + m]
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan_to_num(x - mu)
    return np.nan_to_num((x - mu) / sd)


def _linear_detrend(y: np.ndarray) -> np.ndarray:
    """Remove a best-fit line along the last axis (numpy-only)."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    x = np.arange(y.shape[-1], dtype=float)
    x = x - np.mean(x)
    denom = float(np.sum(x * x)) if x.size else 1.0
    if denom <= 0:
        return y - np.mean(y, axis=-1, keepdims=True)
    # slope per row
    y_mean = np.mean(y, axis=-1, keepdims=True)
    num = np.sum((y - y_mean) * x, axis=-1, keepdims=True)
    slope = num / denom
    trend = y_mean + slope * x
    return y - trend


def _rolling_median_detrend(y: np.ndarray, *, win: int = 9) -> np.ndarray:
    """Detrend by subtracting rolling median (robust), numpy-only."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    # pad by edge values for stability
    ypad = np.pad(y, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(y.shape[1]):
        sl = ypad[:, i : i + win]
        med = np.median(sl, axis=1)
        out[:, i] = y[:, i] - med
    return out


def _compute_recurrence_matrix(
    X: np.ndarray,
    *,
    max_points: int,
    eps_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if X.size == 0:
        return np.zeros((0, 0), dtype=float), np.arange(0, dtype=int), 0.0

    idx = _downsample_idx(len(X), max_points)
    Xs = np.asarray(X[idx], dtype=float)

    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    d = np.sqrt(np.maximum(d2, 0.0))

    tri = d[np.triu_indices(d.shape[0], k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        eps = 0.0
    else:
        eps = float(np.quantile(tri, float(eps_quantile)))

    if not np.isfinite(eps) or eps <= 0:
        eps = float(np.nanmedian(tri)) if tri.size else 0.0

    R = (d <= eps).astype(float)
    return R, idx, float(eps)


def _glow_markers_3d(
    *,
    go: Any,
    X: np.ndarray,
    name: str,
    color_core: str = "#ff3b3b",
    color_glow: str = "#ffcc00",
    size_core: int = 4,
    size_glow: int = 10,
    opacity_glow: float = 0.18,
) -> List[Any]:
    """Return two Scatter3d traces that look like a glow marker."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    glow = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=f"{name} (glow)",
        marker={
            "size": size_glow,
            "color": color_glow,
            "opacity": float(opacity_glow),
        },
        showlegend=False,
        hoverinfo="skip",
    )
    core = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        marker={
            "size": size_core,
            "color": color_core,
            "opacity": 0.95,
            "line": {"color": "#ffffff", "width": 1},
        },
    )
    return [glow, core]


def _find_rqa_scale_csvs(rqa_dir: Path) -> List[Path]:
    return sorted(rqa_dir.glob("multiscale_rqa_scale_*.csv"))


def _build_rqa_matrix(
    rqa_dir: Path,
    *,
    metric: str = "det_mean",
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """Build a (n_scales, n_windows) matrix from per-scale CSVs."""
    scale_paths = _find_rqa_scale_csvs(rqa_dir)
    if not scale_paths:
        return np.zeros((0, 0), dtype=float), [], None

    rows = []
    labels = []
    centers = None

    for p in scale_paths:
        df = pd.read_csv(p)
        if metric not in df.columns:
            continue
        v = df[metric].to_numpy(dtype=float)
        rows.append(v)
        labels.append(p.stem.replace("multiscale_rqa_scale_", ""))
        if centers is None and "center_time_s" in df.columns:
            centers = df["center_time_s"].to_numpy(dtype=float)

    if not rows:
        return np.zeros((0, 0), dtype=float), [], centers

    max_len = max(len(r) for r in rows)
    mat = np.full((len(rows), max_len), np.nan, dtype=float)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return mat, labels, centers


def generate_enhanced_viz(
    *,
    run_dir: Path,
    out_dir: Optional[Path] = None,
    input_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
    run_id: Optional[int] = None,
    signal_col: str = "foil_height_m",
    time_col: str = "time_s",
    embed_dim: int = 3,
    embed_lag: int = 3,
    max_points: int = 6000,
    rp_max_points: int = 650,
    rp_eps_quantile: float = 0.10,
    rqa_metric: str = "det_mean",
    rqa_detrend: str = "median",
    rqa_median_win: int = 9,
) -> EnhancedVizArtifacts:
    go, _make_subplots = _lazy_plotly()

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "viz_enhanced"
    _ensure_out_dir(out_dir)

    anomalies_path = run_dir / "anomalies.csv"
    manifest_path = run_dir / "manifest.json"
    anomalies = pd.read_csv(anomalies_path)
    manifest = _read_json(manifest_path) if manifest_path.exists() else None

    # Resolve input CSV (optional but used for nicer hover and robust signal selection)
    in_path: Optional[Path] = None
    if input_path:
        p = Path(input_path)
        if p.exists():
            in_path = p
        elif repo_root and (Path(repo_root) / p).exists():
            in_path = Path(repo_root) / p
        elif (run_dir / p).exists():
            in_path = run_dir / p

    if in_path is None and manifest is not None:
        params = manifest.get("params", {})
        m_in = params.get("input")
        if isinstance(m_in, str) and m_in.strip():
            p = Path(m_in)
            if p.exists():
                in_path = p
            elif repo_root and (Path(repo_root) / p).exists():
                in_path = Path(repo_root) / p
            elif (run_dir / p).exists():
                in_path = run_dir / p

    # pick run_id
    rid = int(run_id) if run_id is not None else None
    if rid is None:
        if manifest is not None:
            rc = manifest.get("params", {}).get("run_choice")
            if isinstance(rc, int) and rc > 0:
                rid = int(rc)
    if rid is None and "run_id" in anomalies.columns:
        rid = int(anomalies["run_id"].iloc[0])
    if rid is None:
        rid = 1

    tl = anomalies[anomalies["run_id"] == rid].reset_index(drop=True) if "run_id" in anomalies.columns else anomalies

    if time_col in tl.columns:
        t = tl[time_col].to_numpy(dtype=float)
    else:
        t = np.arange(len(tl), dtype=float)

    # Choose signal
    if signal_col in tl.columns:
        sig = tl[signal_col].to_numpy(dtype=float)
    else:
        # fallback on score_mean
        sig = tl.get("score_mean", pd.Series(np.zeros(len(tl)))).to_numpy(dtype=float)

    sigz = _zscore(sig)
    X = _takens_embedding_1d(sigz, dim=int(embed_dim), lag=int(embed_lag))

    # Align anomalies to embedding length
    burn = (int(embed_dim) - 1) * int(embed_lag)
    t2 = t[burn:] if burn < t.size else np.zeros(0, dtype=float)
    tl2 = tl.iloc[burn:].reset_index(drop=True) if burn < len(tl) else tl.iloc[:0].copy()

    idx = _downsample_idx(len(X), int(max_points))
    Xd = X[idx]
    td = t2[idx] if t2.size else np.arange(len(Xd), dtype=float)

    is_drop = tl2.get("is_drop", pd.Series(np.zeros(len(tl2)))).to_numpy(dtype=float) > 0.5
    is_alert = tl2.get("alert_dyn", pd.Series(np.zeros(len(tl2)))).to_numpy(dtype=float) > 0.5

    is_drop_d = is_drop[idx] if is_drop.size else np.zeros(len(Xd), dtype=bool)
    is_alert_d = is_alert[idx] if is_alert.size else np.zeros(len(Xd), dtype=bool)

    # Phase space 3D
    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter3d(
            x=Xd[:, 0],
            y=Xd[:, 1],
            z=Xd[:, 2],
            mode="lines",
            name="Trajectory",
            line={"width": 2},
            customdata=np.stack([td], axis=1),
            hovertemplate="t=%{customdata[0]:.2f}s<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )

    # Overlays
    if np.any(is_alert_d):
        Xa = Xd[is_alert_d]
        for tr in _glow_markers_3d(go=go, X=Xa, name="Alerts", color_core="#9b59ff", color_glow="#a6fffb"):
            fig_phase.add_trace(tr)

    if np.any(is_drop_d):
        Xr = Xd[is_drop_d]
        for tr in _glow_markers_3d(go=go, X=Xr, name="Foil drops", color_core="#ff3b3b", color_glow="#ffcc00"):
            fig_phase.add_trace(tr)

    fig_phase.update_layout(
        title="Enhanced 3D Phase Space, with glow overlays",
        scene={"xaxis_title": "Dim1", "yaxis_title": "Dim2", "zaxis_title": "Dim3"},
        **_dark_layout_kwargs(),
    )

    # Recurrence plot from embedding
    R, ridx, eps = _compute_recurrence_matrix(X, max_points=int(rp_max_points), eps_quantile=float(rp_eps_quantile))
    fig_rp = go.Figure(
        data=[
            go.Heatmap(
                z=R,
                colorscale="Greys",
                zmin=0.0,
                zmax=1.0,
                showscale=False,
            )
        ]
    )
    fig_rp.update_layout(
        title=f"Recurrence Plot (eps quantile={float(rp_eps_quantile):.2f}, eps={eps:.4f})",
        xaxis_title="Index",
        yaxis_title="Index",
        **_dark_layout_kwargs(),
    )

    # RQA heatmap (DET by default), detrended for visibility
    rqa_dir = run_dir / "rqa_multiscale"
    mat, scale_labels, centers = _build_rqa_matrix(rqa_dir, metric=str(rqa_metric))
    mat_dt = mat.copy()
    detr = str(rqa_detrend).lower().strip()
    if mat_dt.size:
        if detr == "linear":
            mat_dt = _linear_detrend(mat_dt)
        elif detr == "median":
            mat_dt = _rolling_median_detrend(mat_dt, win=int(rqa_median_win))
        elif detr == "none":
            pass
        else:
            raise ValueError("rqa_detrend must be one of: none, linear, median")

    xlab = centers if centers is not None and centers.size else np.arange(mat_dt.shape[1], dtype=float)
    ylab = scale_labels if scale_labels else [str(i) for i in range(mat_dt.shape[0])]

    fig_rqa = go.Figure(
        data=[
            go.Heatmap(
                z=mat_dt,
                x=xlab,
                y=ylab,
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": f"{rqa_metric} (detrended)"},
            )
        ]
    )
    fig_rqa.update_layout(
        title=f"Multiscale RQA heatmap ({rqa_metric}), detrend={detr}",
        xaxis_title="Center time (s)" if centers is not None else "Window index",
        yaxis_title="Scale (s)" if scale_labels else "Scale",
        **_dark_layout_kwargs(),
    )

    # Write artifacts
    phase_html = out_dir / "enhanced_phase_space_3d.html"
    rp_html = out_dir / "enhanced_recurrence_plot.html"
    rqa_html = out_dir / "enhanced_rqa_heatmap.html"
    dash_html = out_dir / "index.html"
    meta_json = out_dir / "metadata.json"

    _write_html(fig_phase, phase_html)
    _write_html(fig_rp, rp_html)
    _write_html(fig_rqa, rqa_html)

    meta = {
        "run_dir": str(run_dir),
        "run_id": int(rid),
        "input_path": str(in_path) if in_path is not None else None,
        "signal_col": signal_col,
        "embed_dim": int(embed_dim),
        "embed_lag": int(embed_lag),
        "rp_max_points": int(rp_max_points),
        "rp_eps_quantile": float(rp_eps_quantile),
        "rqa_metric": rqa_metric,
        "rqa_detrend": rqa_detrend,
        "rqa_median_win": int(rqa_median_win),
    }
    meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    dash = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ChaosTrace Enhanced Viz</title>
  <style>
    body {{ background: #06070a; color: #e8e8e8; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif; }}
    .wrap {{ max-width: 980px; margin: 32px auto; padding: 0 16px; }}
    a {{ color: #9b59ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .card {{ background: #0b0d12; border: 1px solid #1b2030; border-radius: 16px; padding: 16px; margin: 12px 0; }}
    .small {{ color: #aab; font-size: 13px; }}
    code {{ background: #111522; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ChaosTrace Enhanced Viz</h1>
    <p class="small">Run id: <code>{int(rid)}</code></p>

    <div class="card">
      <h2>Enhanced phase space 3D</h2>
      <p><a href="{phase_html.name}">Open enhanced_phase_space_3d.html</a></p>
      <p class="small">Glow overlays: foil drops in red, alerts in violet.</p>
    </div>

    <div class="card">
      <h2>Enhanced recurrence plot</h2>
      <p><a href="{rp_html.name}">Open enhanced_recurrence_plot.html</a></p>
    </div>

    <div class="card">
      <h2>Enhanced multiscale RQA heatmap</h2>
      <p><a href="{rqa_html.name}">Open enhanced_rqa_heatmap.html</a></p>
      <p class="small">Detrending improves contrast. Try rqa_detrend=linear or median.</p>
    </div>

    <div class="card">
      <h2>Metadata</h2>
      <p><a href="{meta_json.name}">Open metadata.json</a></p>
    </div>
  </div>
</body>
</html>
"""
    dash_html.write_text(dash, encoding="utf-8")

    return EnhancedVizArtifacts(
        out_dir=out_dir,
        phase3d_html=phase_html,
        recurrence_html=rp_html,
        rqa_heatmap_html=rqa_html,
        dashboard_html=dash_html,
        metadata_json=meta_json,
    )
