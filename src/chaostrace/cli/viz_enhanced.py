from __future__ import annotations

import argparse
from pathlib import Path

from chaostrace.viz.enhanced import generate_enhanced_viz


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate enhanced interactive Plotly visualizations for a ChaosTrace run directory. "
            "Adds glow overlays for anomalies, plus detrended multiscale RQA heatmaps."
        )
    )
    p.add_argument("--run-dir", required=True, help="Run directory containing anomalies.csv/manifest.json")
    p.add_argument("--out", default=None, help="Output directory (default: <run-dir>/viz_enhanced)")
    p.add_argument("--input", default=None, help="Optional original input CSV (overrides manifest.params.input)")
    p.add_argument("--repo-root", default=None, help="Repo root to resolve manifest input path")
    p.add_argument("--run-id", type=int, default=None, help="Select run_id explicitly")

    p.add_argument("--signal-col", default="foil_height_m", help="Column to embed for phase space")
    p.add_argument("--time-col", default="time_s", help="Time column name")
    p.add_argument("--embed-dim", type=int, default=3)
    p.add_argument("--embed-lag", type=int, default=3)

    p.add_argument("--max-points", type=int, default=6000)
    p.add_argument("--rp-max-points", type=int, default=650)
    p.add_argument("--rp-eps-quantile", type=float, default=0.10)

    p.add_argument("--rqa-metric", default="det_mean", help="Metric column from multiscale RQA CSVs, e.g. det_mean, lam_mean")
    p.add_argument("--rqa-detrend", default="median", choices=["none", "linear", "median"])
    p.add_argument("--rqa-median-win", type=int, default=9)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else None
    input_path = str(args.input) if args.input else None
    repo_root = Path(args.repo_root) if args.repo_root else None

    generate_enhanced_viz(
        run_dir=run_dir,
        out_dir=out_dir,
        input_path=input_path,
        repo_root=repo_root,
        run_id=args.run_id,
        signal_col=str(args.signal_col),
        time_col=str(args.time_col),
        embed_dim=int(args.embed_dim),
        embed_lag=int(args.embed_lag),
        max_points=int(args.max_points),
        rp_max_points=int(args.rp_max_points),
        rp_eps_quantile=float(args.rp_eps_quantile),
        rqa_metric=str(args.rqa_metric),
        rqa_detrend=str(args.rqa_detrend),
        rqa_median_win=int(args.rqa_median_win),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
