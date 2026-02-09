from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VizArtifacts:
    out_dir: Path
    phase3d_html: Path
    timeline_html: Path
    recurrence_html: Path
    dashboard_html: Path
    phase_animation_html: Optional[Path]
    metadata_json: Path


def _lazy_plotly() -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for cinematic visualizations. "
            "Install it with: pip install -r requirements_viz.txt"
        ) from e


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


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def _segments_from_mask(time_s: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    t = np.asarray(time_s, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if t.size == 0 or t.size != m.size:
        return []
    segs: List[Tuple[float, float]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(m):
        if v and not in_seg:
            in_seg = True
            start = i
        elif in_seg and not v:
            segs.append((float(t[start]), float(t[i - 1])))
            in_seg = False
    if in_seg:
        segs.append((float(t[start]), float(t[-1])))
    return segs


def _rising_edges(time_s: np.ndarray, mask: np.ndarray) -> List[float]:
    t = np.asarray(time_s, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if t.size == 0 or t.size != m.size:
        return []
    edges: List[float] = []
    prev = False
    for i, v in enumerate(m):
        if v and not prev:
            edges.append(float(t[i]))
        prev = bool(v)
    return edges


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_input_path(
    *,
    run_dir: Path,
    manifest: Optional[Dict[str, Any]],
    input_path: Optional[str],
    repo_root: Optional[Path],
) -> Optional[Path]:
    if input_path:
        p = Path(input_path)
        if p.exists():
            return p
        if repo_root and (repo_root / p).exists():
            return repo_root / p
        if (run_dir / p).exists():
            return run_dir / p

    if manifest:
        params = manifest.get("params", {})
        m_in = params.get("input")
        if isinstance(m_in, str) and m_in.strip():
            p = Path(m_in)
            if p.exists():
                return p
            if repo_root and (repo_root / p).exists():
                return repo_root / p
            if (run_dir / p).exists():
                return run_dir / p

    return None


def _pick_run_id(
    *,
    anomalies: pd.DataFrame,
    metrics: Optional[pd.DataFrame],
    manifest: Optional[Dict[str, Any]],
    run_id: Optional[int],
) -> int:
    if run_id is not None:
        return int(run_id)

    if manifest:
        params = manifest.get("params", {})
        rc = params.get("run_choice")
        if isinstance(rc, int) and rc > 0:
            return int(rc)

    if metrics is not None and not metrics.empty:
        if "alert_frac" in metrics.columns:
            # Choose median alert_frac: stable representative.
            m2 = metrics.sort_values("alert_frac").reset_index(drop=True)
            return int(m2.iloc[len(m2) // 2]["run_id"])
        return int(metrics.iloc[0]["run_id"])

    if "run_id" in anomalies.columns:
        return int(anomalies["run_id"].iloc[0])

    return 1


def _dark_layout_kwargs() -> Dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "#06070a",
        "plot_bgcolor": "#06070a",
        "font": {"family": "system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif", "size": 13},
    }


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_html(fig: Any, path: Path) -> None:
    path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def _maybe_write_png(fig: Any, path: Path, *, scale: float = 2.0) -> bool:
    # Requires kaleido.
    try:
        fig.write_image(str(path), scale=scale)
        return True
    except Exception:
        return False


def _compute_recurrence_matrix(
    X: np.ndarray,
    *,
    max_points: int,
    eps_quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.zeros((0, 0), dtype=float), np.arange(0, dtype=int)

    idx = _downsample_idx(len(X), max_points)
    Xs = np.asarray(X[idx], dtype=float)

    # Pairwise distances (vectorized).
    # For max_points=500 this is ~250k entries, ok.
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
    return R, idx


def generate_cinematic_suite(
    *,
    run_dir: Path,
    out_dir: Optional[Path] = None,
    input_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
    run_id: Optional[int] = None,
    color_by: str = "score",
    theme: str = "dark",
    max_points: int = 9000,
    rp_max_points: int = 500,
    rp_eps_quantile: float = 0.10,
    make_animation: bool = False,
    export_png: bool = False,
) -> VizArtifacts:
    go, make_subplots = _lazy_plotly()

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    anomalies_path = run_dir / "anomalies.csv"
    metrics_path = run_dir / "metrics.csv"
    manifest_path = run_dir / "manifest.json"

    anomalies = pd.read_csv(anomalies_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    manifest = _read_json(manifest_path) if manifest_path.exists() else None

    rid = _pick_run_id(anomalies=anomalies, metrics=metrics, manifest=manifest, run_id=run_id)

    if "run_id" in anomalies.columns:
        tl = anomalies[anomalies["run_id"] == rid].reset_index(drop=True)
    else:
        tl = anomalies.reset_index(drop=True)

    time_s = tl["time_s"].to_numpy(dtype=float) if "time_s" in tl.columns else np.arange(len(tl), dtype=float)

    # Downsample for speed.
    idx = _downsample_idx(len(tl), max_points)
    tl_ds = tl.iloc[idx].reset_index(drop=True)
    t_ds = time_s[idx]

    score = tl_ds.get("score_mean", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float)
    inv = tl_ds.get("score_invariant", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float)
    var = tl_ds.get("score_variant", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float)

    is_drop = tl_ds.get("is_drop", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float) > 0.5
    alert_dyn = tl_ds.get("alert_dyn", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float) > 0.5
    alert_fixed = tl_ds.get("alert_fixed", pd.Series(np.zeros(len(tl_ds)))).to_numpy(dtype=float) > 0.5

    # Choose alert channel.
    alert = alert_dyn if "alert_dyn" in tl_ds.columns else alert_fixed

    # Optional input for nicer timeline.
    input_file = _resolve_input_path(run_dir=run_dir, manifest=manifest, input_path=input_path, repo_root=repo_root)
    df_raw: Optional[pd.DataFrame] = None
    if input_file and input_file.exists():
        try:
            df_raw = pd.read_csv(input_file)
            if "time_s" in df_raw.columns:
                df_raw = df_raw.sort_values("time_s").reset_index(drop=True)
        except Exception:
            df_raw = None

    layout = _dark_layout_kwargs() if theme == "dark" else {"template": "plotly_white"}

    # Phase space 3D
    base_signal = None
    if df_raw is not None and "boat_speed" in df_raw.columns:
        base_signal = df_raw["boat_speed"].to_numpy(dtype=float)
        base_time = df_raw["time_s"].to_numpy(dtype=float)
    else:
        base_signal = tl.get("score_mean", pd.Series(np.zeros(len(tl)))).to_numpy(dtype=float)
        base_time = time_s

    # Keep phase space aligned with the chosen run (use full tl, not downsampled, then downsample embedding).
    dim = 3
    lag = 3
    X = _takens_embedding_1d(np.asarray(base_signal, dtype=float), dim=dim, lag=lag)
    if X.size == 0:
        X = np.zeros((0, 3), dtype=float)

    # Align times and colors to embedding length.
    t_emb = np.asarray(base_time, dtype=float)
    t_emb = t_emb[-len(X) :] if len(X) > 0 else np.asarray([], dtype=float)

    # Color.
    if color_by == "time":
        c = t_emb
        c_title = "time_s"
    elif color_by == "score":
        c_full = tl.get("score_mean", pd.Series(np.zeros(len(tl)))).to_numpy(dtype=float)
        c = c_full[-len(X) :] if len(X) > 0 else np.asarray([], dtype=float)
        c_title = "score_mean"
    elif color_by == "drop":
        d_full = tl.get("is_drop", pd.Series(np.zeros(len(tl)))).to_numpy(dtype=float)
        c = d_full[-len(X) :] if len(X) > 0 else np.asarray([], dtype=float)
        c_title = "is_drop"
    else:
        a_full = tl.get("alert_dyn", tl.get("alert_fixed", pd.Series(np.zeros(len(tl))))).to_numpy(dtype=float)
        c = a_full[-len(X) :] if len(X) > 0 else np.asarray([], dtype=float)
        c_title = "alert"

    idx_emb = _downsample_idx(len(X), min(max_points, 5000)) if len(X) else np.arange(0, dtype=int)
    Xd = X[idx_emb] if len(X) else X
    cd = c[idx_emb] if len(X) else c
    td = t_emb[idx_emb] if len(X) else t_emb

    phase_fig = go.Figure()

    # Ghost line (constant color) + colored points + glow layer
    if len(Xd) > 1:
        phase_fig.add_trace(
            go.Scatter3d(
                x=Xd[:, 0],
                y=Xd[:, 1],
                z=Xd[:, 2],
                mode="lines",
                line={"width": 2, "color": "rgba(220,220,220,0.15)"},
                hoverinfo="skip",
                name="trajectory",
            )
        )

    phase_fig.add_trace(
        go.Scatter3d(
            x=Xd[:, 0],
            y=Xd[:, 1],
            z=Xd[:, 2],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.75,
                "color": cd,
                "colorscale": "Inferno",
                "showscale": True,
                "colorbar": {"title": c_title},
            },
            customdata=np.column_stack([td, cd]) if len(Xd) else None,
            hovertemplate="time_s=%{customdata[0]:.3f}<br>color=%{customdata[1]:.3f}<extra></extra>",
            name="points",
        )
    )

    phase_fig.add_trace(
        go.Scatter3d(
            x=Xd[:, 0],
            y=Xd[:, 1],
            z=Xd[:, 2],
            mode="markers",
            marker={"size": 9, "opacity": 0.10, "color": cd, "colorscale": "Inferno", "showscale": False},
            hoverinfo="skip",
            name="glow",
        )
    )

    phase_fig.update_layout(
        title="Phase space 3D (Takens embedding)",
        **layout,
        scene={
            "xaxis": {"title": "x", "backgroundcolor": "#06070a", "gridcolor": "rgba(255,255,255,0.06)"},
            "yaxis": {"title": "y", "backgroundcolor": "#06070a", "gridcolor": "rgba(255,255,255,0.06)"},
            "zaxis": {"title": "z", "backgroundcolor": "#06070a", "gridcolor": "rgba(255,255,255,0.06)"},
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        legend={"orientation": "h"},
    )

    # Timeline
    timeline_fig = go.Figure()

    # Primary signals
    if df_raw is not None and "time_s" in df_raw.columns:
        t_raw = df_raw["time_s"].to_numpy(dtype=float)
        if "foil_height_m" in df_raw.columns:
            timeline_fig.add_trace(
                go.Scatter(
                    x=t_raw,
                    y=df_raw["foil_height_m"].to_numpy(dtype=float),
                    mode="lines",
                    line={"width": 2},
                    name="foil_height_m",
                    hovertemplate="time_s=%{x:.3f}<br>foil=%{y:.3f}<extra></extra>",
                    yaxis="y1",
                )
            )
        if "boat_speed" in df_raw.columns:
            timeline_fig.add_trace(
                go.Scatter(
                    x=t_raw,
                    y=df_raw["boat_speed"].to_numpy(dtype=float),
                    mode="lines",
                    line={"width": 2, "dash": "dot"},
                    name="boat_speed",
                    hovertemplate="time_s=%{x:.3f}<br>speed=%{y:.3f}<extra></extra>",
                    yaxis="y1",
                )
            )

    # Score on secondary axis
    timeline_fig.add_trace(
        go.Scatter(
            x=t_ds,
            y=score,
            mode="lines",
            line={"width": 2},
            name="score_mean",
            hovertemplate="time_s=%{x:.3f}<br>score=%{y:.3f}<extra></extra>",
            yaxis="y2",
        )
    )

    # Zones
    inv_zone = inv > 0.6
    var_zone = (var > 0.5) & (inv < 0.4)
    for t0, t1 in _segments_from_mask(t_ds, inv_zone):
        timeline_fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(90,200,120,0.10)", line_width=0)
    for t0, t1 in _segments_from_mask(t_ds, var_zone):
        timeline_fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(255,140,60,0.10)", line_width=0)

    # Event markers
    for x in _rising_edges(t_ds, is_drop):
        timeline_fig.add_vline(x=x, line_width=2, line_dash="dash", line_color="rgba(255,80,80,0.9)")
    for x in _rising_edges(t_ds, alert):
        timeline_fig.add_vline(x=x, line_width=2, line_dash="dot", line_color="rgba(255,220,120,0.9)")

    # Threshold
    thr = None
    if "threshold_dyn" in tl_ds.columns:
        thr = _safe_float(tl_ds["threshold_dyn"].iloc[0], default=0.55)
    elif "alert_threshold" in tl_ds.columns:
        thr = _safe_float(tl_ds["alert_threshold"].iloc[0], default=0.55)

    if thr is not None:
        timeline_fig.add_hline(y=thr, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)", yref="y2")

    timeline_fig.update_layout(
        title="Timeline (signals + zones + events)",
        **layout,
        xaxis={"title": "time_s", "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis={"title": "signals", "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis2={
            "title": "score",
            "overlaying": "y",
            "side": "right",
            "range": [-0.05, 1.05],
            "gridcolor": "rgba(255,255,255,0.06)",
        },
        legend={"orientation": "h"},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    # Recurrence plot (mini)
    R, idx_rp = _compute_recurrence_matrix(X, max_points=rp_max_points, eps_quantile=rp_eps_quantile)
    rp_fig = go.Figure()
    if R.size:
        rp_fig.add_trace(
            go.Heatmap(
                z=R,
                colorscale="Viridis",
                showscale=False,
            )
        )
        # Overlay drop onsets as vertical lines if we can map to embedding index.
        # Mapping: embedding corresponds to last len(X) times.
        if len(X) > 0:
            # Build a boolean onset series in raw time.
            if "is_drop" in tl.columns:
                drop_full = tl["is_drop"].to_numpy(dtype=float) > 0.5
                drop_on = np.zeros_like(drop_full, dtype=bool)
                prev = False
                for i, v in enumerate(drop_full):
                    if v and not prev:
                        drop_on[i] = True
                    prev = bool(v)
                drop_on = drop_on[-len(X) :]
                drop_on = drop_on[idx_rp] if drop_on.size == idx_rp.size else drop_on
                onset_idx = np.flatnonzero(drop_on)
                for k in onset_idx:
                    rp_fig.add_vline(x=float(k), line_width=1, line_color="rgba(255,80,80,0.85)")
                    rp_fig.add_hline(y=float(k), line_width=1, line_color="rgba(255,80,80,0.85)")

    rp_fig.update_layout(
        title="Recurrence plot (mini RP)",
        **layout,
        xaxis={"showticklabels": False},
        yaxis={"showticklabels": False, "scaleanchor": "x"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    # Animation (optional)
    anim_path: Optional[Path] = None
    anim_fig = None
    if make_animation and len(Xd) > 10:
        # Keep it small.
        n_frames = min(240, len(Xd))
        frame_idx = _downsample_idx(len(Xd), n_frames)

        head_x = Xd[frame_idx, 0]
        head_y = Xd[frame_idx, 1]
        head_z = Xd[frame_idx, 2]

        anim_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=Xd[:, 0],
                    y=Xd[:, 1],
                    z=Xd[:, 2],
                    mode="lines",
                    line={"width": 2, "color": "rgba(220,220,220,0.12)"},
                    hoverinfo="skip",
                ),
                go.Scatter3d(
                    x=[head_x[0]],
                    y=[head_y[0]],
                    z=[head_z[0]],
                    mode="markers",
                    marker={"size": 6, "opacity": 0.95, "color": "rgba(255,255,255,0.95)"},
                    hoverinfo="skip",
                ),
            ],
            layout=go.Layout(
                title="Phase trajectory animation (head point)",
                **layout,
                scene=phase_fig.layout.scene,
                margin={"l": 0, "r": 0, "t": 50, "b": 0},
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True},
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                            },
                        ],
                    }
                ],
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Scatter3d(x=Xd[:, 0], y=Xd[:, 1], z=Xd[:, 2]),
                        go.Scatter3d(x=[head_x[i]], y=[head_y[i]], z=[head_z[i]]),
                    ]
                )
                for i in range(len(frame_idx))
            ],
        )

    # Output paths
    out = _ensure_out_dir(out_dir or (run_dir / "viz_cinematic"))
    phase_path = out / "01_phase_space_3d.html"
    timeline_path = out / "02_timeline_interactive.html"
    rp_path = out / "03_recurrence_plot.html"
    dashboard_path = out / "04_dashboard.html"
    metadata_path = out / "metadata.json"

    _write_html(phase_fig, phase_path)
    _write_html(timeline_fig, timeline_path)
    _write_html(rp_fig, rp_path)

    if anim_fig is not None:
        anim_path = out / "05_phase_animation.html"
        _write_html(anim_fig, anim_path)

    if export_png:
        _maybe_write_png(phase_fig, out / "01_phase_space_3d.png", scale=2.5)
        _maybe_write_png(timeline_fig, out / "02_timeline_interactive.png", scale=2.5)
        _maybe_write_png(rp_fig, out / "03_recurrence_plot.png", scale=2.5)

    # Dashboard HTML: simple cinematic index with iframes.
    title = "ChaosTrace cinematic viz"
    picked = {
        "run_id": int(rid),
        "color_by": str(color_by),
        "theme": str(theme),
        "input_resolved": str(input_file) if input_file else None,
    }

    table_html = ""
    if metrics is not None and not metrics.empty:
        # Keep a compact subset.
        cols = [c for c in ["run_id", "f1", "precision", "recall", "alert_frac", "alert_frac_dyn"] if c in metrics.columns]
        if not cols:
            cols = ["run_id"]
        m2 = metrics.copy()
        # Prefer max f1 if available.
        if "f1" in m2.columns:
            m2 = m2.sort_values("f1", ascending=False)
        elif "alert_frac" in m2.columns:
            m2 = m2.sort_values("alert_frac", ascending=True)
        m2 = m2[cols].head(12)
        table_html = m2.to_html(index=False, escape=True)

    dashboard = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #06070a; color: #e6e6e6; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
    h1 {{ font-size: 20px; margin: 0 0 10px 0; }}
    .meta {{ font-size: 13px; opacity: 0.85; margin-bottom: 14px; }}
    a {{ color: #9ad1ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden; }}
    .card header {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 14px; }}
    iframe {{ width: 100%; height: 640px; border: 0; display: block; background: #06070a; }}
    .table {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid rgba(255,255,255,0.08); padding: 8px 10px; text-align: left; }}
    th {{ background: rgba(255,255,255,0.05); }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>ChaosTrace cinematic viz</h1>
    <div class=\"meta\">
      Run id: {picked['run_id']}<br/>
      Input: {picked['input_resolved']}<br/>
      Files: <a href=\"{phase_path.name}\">phase</a> | <a href=\"{timeline_path.name}\">timeline</a> | <a href=\"{rp_path.name}\">recurrence</a>{' | <a href="' + anim_path.name + '">animation</a>' if anim_path else ''}
    </div>

    <div class=\"card\">
      <header>Run comparison (top rows)</header>
      <div class=\"table\">{table_html}</div>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><header>Phase space 3D</header><iframe src=\"{phase_path.name}\"></iframe></div>
      <div class=\"card\"><header>Timeline</header><iframe src=\"{timeline_path.name}\"></iframe></div>
      <div class=\"card\"><header>Recurrence plot</header><iframe src=\"{rp_path.name}\"></iframe></div>
      {('<div class="card"><header>Phase animation</header><iframe src="' + anim_path.name + '"></iframe></div>') if anim_path else ''}
    </div>
  </div>
</body>
</html>"""

    dashboard_path.write_text(dashboard, encoding="utf-8")

    # Metadata
    meta = {
        "picked": picked,
        "run_dir": str(run_dir),
        "out_dir": str(out),
        "files": {
            "phase3d_html": phase_path.name,
            "timeline_html": timeline_path.name,
            "recurrence_html": rp_path.name,
            "dashboard_html": dashboard_path.name,
            "phase_animation_html": anim_path.name if anim_path else None,
        },
        "params": {
            "max_points": int(max_points),
            "rp_max_points": int(rp_max_points),
            "rp_eps_quantile": float(rp_eps_quantile),
            "make_animation": bool(make_animation),
            "export_png": bool(export_png),
        },
    }
    metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return VizArtifacts(
        out_dir=out,
        phase3d_html=phase_path,
        timeline_html=timeline_path,
        recurrence_html=rp_path,
        dashboard_html=dashboard_path,
        phase_animation_html=anim_path,
        metadata_json=metadata_path,
    )
