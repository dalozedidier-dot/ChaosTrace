from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..data.ingest import load_timeseries
from ..data.synthetic import add_realistic_noise, tiny_perturb
from ..orchestrator.sweep import SweepConfig, sweep
from ..utils.manifest import write_manifest

def build_grid(runs: int) -> list[SweepConfig]:
    windows = [5.0, 10.0, 30.0]
    thresholds = [0.25, 0.30, 0.35]
    dims = [3, 4, 5]
    lags = [3, 5, 8]
    grid = []
    for w in windows:
        for th in thresholds:
            for d in dims:
                for lag in lags:
                    grid.append(SweepConfig(window_s=w, drop_threshold=th, emb_dim=d, emb_lag=lag))
    return grid[: max(runs, 0)]

def phase_portrait_3d(df: pd.DataFrame, out_png: Path) -> None:
    xs = df["boat_speed"].to_numpy(dtype=float)
    ys = df["foil_height_m"].to_numpy(dtype=float)
    zs = df["wind_speed"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=1.0)
    ax.set_xlabel("boat_speed")
    ax.set_ylabel("foil_height_m")
    ax.set_zlabel("wind_speed")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_timeseries(args.input)
    rng = np.random.default_rng(args.seed)
    if args.augment:
        df = add_realistic_noise(df, rng=rng)
        df = tiny_perturb(df, rng=rng, eps=1e-3)

    cfgs = build_grid(args.runs)
    metrics_df, anoms_df = sweep(df, cfgs=cfgs, seed=args.seed)

    (out_dir / "metrics.csv").write_text(metrics_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "anomalies.csv").write_text(anoms_df.to_csv(index=False), encoding="utf-8")

    fig_path = out_dir / "fig_phase.png"
    phase_portrait_3d(df, fig_path)

    write_manifest(
        out_dir,
        params={"input": str(Path(args.input)), "runs": args.runs, "seed": args.seed, "augment": bool(args.augment)},
        files=["metrics.csv", "anomalies.csv", "fig_phase.png"],
    )
    print(f"Wrote: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
