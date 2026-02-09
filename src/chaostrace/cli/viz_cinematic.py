from __future__ import annotations

import argparse
from pathlib import Path

from chaostrace.viz.cinematic import generate_cinematic_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate cinematic, interactive Plotly visualizations for a ChaosTrace run directory. "
            "Outputs HTML (and optional PNG) artifacts."
        )
    )
    p.add_argument("--run-dir", required=True, help="Run directory containing metrics.csv/anomalies.csv/manifest.json")
    p.add_argument("--out", default=None, help="Output directory (default: <run-dir>/viz_cinematic)")
    p.add_argument("--input", default=None, help="Optional original input CSV (overrides manifest.params.input)")
    p.add_argument("--repo-root", default=None, help="Repo root to resolve manifest input path")
    p.add_argument("--run-id", type=int, default=None, help="Select run_id explicitly")

    p.add_argument("--max-points", type=int, default=4000, help="Max points for 3D phase plot")
    p.add_argument("--rp-max-points", type=int, default=600, help="Max points for recurrence plot")
    p.add_argument("--rp-eps-quantile", type=float, default=0.10, help="Distance quantile for RP epsilon")

    p.add_argument("--make-animation", action="store_true", help="Also generate an animated phase plot (heavier)")
    p.add_argument("--export-png", action="store_true", help="Export PNG snapshots (requires kaleido)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else None
    input_path = Path(args.input) if args.input else None
    repo_root = Path(args.repo_root) if args.repo_root else None

    generate_cinematic_suite(
        run_dir=run_dir,
        out_dir=out_dir,
        input_path=input_path,
        repo_root=repo_root,
        run_id=args.run_id,
        max_points=int(args.max_points),
        rp_max_points=int(args.rp_max_points),
        rp_eps_quantile=float(args.rp_eps_quantile),
        make_animation=bool(args.make_animation),
        export_png=bool(args.export_png),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
