from __future__ import annotations

"""CLI: multiscale RQA over sliding windows.

Example
-------
python -m chaostrace.cli.rqa_multiscale \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/rqa_multiscale \
  --scales 3,5,10,20 \
  --series foil_height_m \
  --drop-threshold 0.30

Outputs
-------
- multiscale_rqa_scale_<Xs>.csv (per-scale)
- multiscale_det_lam.html (Plotly if available, else PNG)
- rqa_multiscale_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from chaostrace.data.ingest import load_timeseries
from chaostrace.features.windowing import estimate_sample_hz
from chaostrace.rqa.advanced import RQAAdvancedConfig, compute_rqa_advanced_from_series, rqa_early_warning_score
from chaostrace.rqa.multivariate import CrossRQAConfig, compute_cross_rqa


def _parse_scales(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty --scales")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multiscale RQA (windowed) for early warnings.")
    p.add_argument("--input", required=True, help="Input CSV/JSON (requires a time axis).")
    p.add_argument("--out", required=True, help="Output directory.")

    p.add_argument("--scales", default="3,5,10,20", help="Comma-separated window sizes (seconds).")
    p.add_argument("--step-frac", type=float, default=0.5, help="Step as fraction of window (0.5=50%).")

    p.add_argument("--series", default="foil_height_m", help="Series column for 1D RQA.")
    p.add_argument("--cross", action="store_true", help="Also compute cross-RQA series vs boat_speed.")
    p.add_argument("--cross-series", default="boat_speed", help="Second series for cross-RQA.")

    p.add_argument("--emb-dim", type=int, default=5)
    p.add_argument("--emb-lag", type=int, default=8)
    p.add_argument("--rr-target", type=float, default=0.02)
    p.add_argument("--theiler-window", type=int, default=None)
    p.add_argument("--l-min", type=int, default=2)
    p.add_argument("--v-min", type=int, default=2)

    p.add_argument("--drop-threshold", type=float, default=0.30, help="Foil drop threshold for overlays.")
    p.add_argument("--max-points", type=int, default=600, help="Max embedded points per window (downsample).")
    p.add_argument("--plotly", action="store_true", help="Force Plotly HTML (requires plotly).")
    return p


def _make_plot(
    out_dir: Path,
    *,
    df_by_scale: Dict[str, pd.DataFrame],
    time_s: np.ndarray,
    is_drop: np.ndarray,
    use_plotly: bool,
) -> str:
    """Create a multiscale DET/LAM plot. Returns filename."""
    if use_plotly:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for label, dfm in df_by_scale.items():
                fig.add_trace(
                    go.Scatter(
                        x=dfm["t_mid_s"],
                        y=dfm["det"],
                        mode="lines",
                        name=f"DET {label}",
                        opacity=0.9,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=dfm["t_mid_s"],
                        y=dfm["lam"],
                        mode="lines",
                        name=f"LAM {label}",
                        opacity=0.6,
                    )
                )

            # Drop overlays
            drop_ts = time_s[is_drop]
            if drop_ts.size:
                # Add a few vertical markers (avoid 2400 shapes)
                # Use change points only
                changes = np.flatnonzero(np.diff(is_drop.astype(int)) != 0) + 1
                drop_starts = [0] + changes.tolist()
                for idx in drop_starts:
                    if idx < len(is_drop) and is_drop[idx]:
                        fig.add_vline(x=float(time_s[idx]), line_width=1, line_dash="dot", opacity=0.25)

            fig.update_layout(
                template="plotly_dark",
                title="Multiscale RQA (DET/LAM)",
                xaxis_title="time_s",
                yaxis_title="metric",
                height=650,
            )
            out_name = "multiscale_det_lam.html"
            fig.write_html(str(out_dir / out_name), include_plotlyjs="cdn")
            return out_name
        except Exception:
            # Fall back to matplotlib if plotly is missing
            use_plotly = False

    import matplotlib.pyplot as plt

    plt.figure()
    for label, dfm in df_by_scale.items():
        plt.plot(dfm["t_mid_s"], dfm["det"], label=f"DET {label}")
        plt.plot(dfm["t_mid_s"], dfm["lam"], label=f"LAM {label}", alpha=0.7)
    plt.xlabel("time_s")
    plt.ylabel("metric")
    plt.title("Multiscale RQA (DET/LAM)")
    plt.legend()
    out_name = "multiscale_det_lam.png"
    plt.savefig(out_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close()
    return out_name


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    df = load_timeseries(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.series not in df.columns:
        raise ValueError(f"Missing --series column: {args.series!r}")
    if args.cross and args.cross_series not in df.columns:
        raise ValueError(f"Missing --cross-series column: {args.cross_series!r}")

    time_s = df["time_s"].to_numpy(dtype=float)
    hz = float(estimate_sample_hz(time_s))
    scales = _parse_scales(str(args.scales))

    foil = df["foil_height_m"].to_numpy(dtype=float)
    is_drop = np.isfinite(foil) & (foil < float(args.drop_threshold))

    adv_cfg = RQAAdvancedConfig(
        emb_dim=int(args.emb_dim),
        emb_lag=int(args.emb_lag),
        threshold_by="frr",
        rr_target=float(args.rr_target),
        epsilon=None,
        theiler_window=int(args.theiler_window) if args.theiler_window is not None else None,
        l_min=int(args.l_min),
        v_min=int(args.v_min),
        max_points=int(args.max_points),
        rng_seed=7,
        enable_network=True,
    )

    cross_cfg = CrossRQAConfig(
        emb_dim=int(args.emb_dim),
        emb_lag=int(args.emb_lag),
        rr_target=float(args.rr_target),
        theiler_window=int(args.theiler_window) if args.theiler_window is not None else None,
        l_min=int(args.l_min),
        v_min=int(args.v_min),
        max_points=int(args.max_points),
        rng_seed=7,
    )

    df_by_scale: Dict[str, pd.DataFrame] = {}
    summary: Dict[str, Any] = {
        "input": str(args.input),
        "hz": hz,
        "scales_s": scales,
        "series": str(args.series),
        "cross": bool(args.cross),
        "drop_threshold": float(args.drop_threshold),
    }

    for s in scales:
        window_n = int(max(20, round(float(s) * hz)))
        step_n = int(max(1, round(window_n * float(args.step_frac))))
        rows: List[Dict[str, Any]] = []
        for start in range(0, len(df) - window_n + 1, step_n):
            end = start + window_n
            x = df[str(args.series)].to_numpy(dtype=float)[start:end]
            m = compute_rqa_advanced_from_series(x, config=adv_cfg)
            m["t_mid_s"] = float(np.median(time_s[start:end]))
            m["scale_s"] = float(s)
            m["early_score"] = rqa_early_warning_score(m)
            if args.cross:
                a = df[str(args.series)].to_numpy(dtype=float)[start:end]
                b = df[str(args.cross_series)].to_numpy(dtype=float)[start:end]
                cm = compute_cross_rqa(a, b, cfg=cross_cfg)
                # Prefix to avoid collisions
                for k, v in cm.items():
                    m[k] = v
            rows.append(m)

        dfm = pd.DataFrame(rows)
        label = f"{int(s)}s" if float(s).is_integer() else f"{s:.1f}s"
        out_csv = out_dir / f"multiscale_rqa_scale_{label}.csv"
        dfm.to_csv(out_csv, index=False, float_format="%.6f")
        df_by_scale[label] = dfm

        # Basic stats per scale
        if not dfm.empty:
            summary[label] = {
                "rows": int(len(dfm)),
                "det_mean": float(pd.to_numeric(dfm["det"], errors="coerce").mean()),
                "lam_mean": float(pd.to_numeric(dfm["lam"], errors="coerce").mean()),
                "trend_mean": float(pd.to_numeric(dfm["trend"], errors="coerce").mean()),
            }

    use_plotly = bool(args.plotly)
    plot_file = _make_plot(out_dir, df_by_scale=df_by_scale, time_s=time_s, is_drop=is_drop, use_plotly=use_plotly)
    summary["plot"] = plot_file

    (out_dir / "rqa_multiscale_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
