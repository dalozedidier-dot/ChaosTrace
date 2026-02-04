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
    # calibrated grid for early instabilities
    windows = [3.0, 5.0, 10.0, 20.0]
    thresholds = [0.10, 0.20, 0.30, 0.40]
    dims = [3, 4, 5]
    lags = [1, 3, 5, 8, 12]
    grid: list[SweepConfig] = []
    for w in windows:
        for th in thresholds:
            for d in dims:
                for lag in lags:
                    grid.append(SweepConfig(window_s=w, drop_threshold=th, emb_dim=d, emb_lag=lag))
    return grid[: max(runs, 0)]

def phase_portrait_3d_colored(df: pd.DataFrame, score: pd.Series | None, out_png: Path) -> None:
    xs = df["boat_speed"].to_numpy(dtype=float)
    ys = df["foil_height_m"].to_numpy(dtype=float)
    zs = df["wind_speed"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if score is None or len(score) != len(df):
        ax.plot(xs, ys, zs, linewidth=1.0)
    else:
        sc = score.to_numpy(dtype=float)
        hi = sc > 0.20
        lo = ~hi
        ax.scatter(xs[lo], ys[lo], zs[lo], s=4)
        ax.scatter(xs[hi], ys[hi], zs[hi], s=10, c="red")

    ax.set_xlabel("boat_speed")
    ax.set_ylabel("foil_height_m")
    ax.set_zlabel("wind_speed")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def timeline_overlay(df: pd.DataFrame, score: pd.Series | None, out_png: Path) -> None:
    t = df["time_s"].to_numpy(dtype=float)
    foil = df["foil_height_m"].to_numpy(dtype=float)
    spd = df["boat_speed"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, foil, linewidth=1.0, label="foil_height_m")
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("foil_height_m")

    ax2 = ax1.twinx()
    ax2.plot(t, spd, linewidth=1.0, label="boat_speed")
    ax2.set_ylabel("boat_speed")

    if score is not None and len(score) == len(df):
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(t, score.to_numpy(dtype=float), linewidth=1.0, label="score_mean")
        ax3.set_ylabel("score_mean")

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

    # Build a score series for coloring (approx): use anomalies if available by nearest time
score_series = None
if not anoms_df.empty:
    # use run_id=0 if present as representative for coloring demo
    a0 = anoms_df[anoms_df["run_id"] == 0].copy()
    if not a0.empty:
        a0 = a0.sort_values("time_s")
        score_series = pd.Series(
            np.interp(
                df["time_s"].to_numpy(dtype=float),
                a0["time_s"].to_numpy(dtype=float),
                a0["score_mean"].to_numpy(dtype=float),
                left=float(a0["score_mean"].iloc[0]),
                right=float(a0["score_mean"].iloc[-1]),
            )
        )

fig_path = out_dir / "fig_phase.png"
phase_portrait_3d_colored(df, score_series, fig_path)

fig_timeline = out_dir / "fig_timeline.png"
timeline_overlay(df, score_series, fig_timeline)

    write_manifest(
        out_dir,
        params={"input": str(Path(args.input)), "runs": args.runs, "seed": args.seed, "augment": bool(args.augment)},
        files=["metrics.csv", "anomalies.csv", "fig_phase.png", "fig_timeline.png"],
    )
    print(f"Wrote: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
