from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chaostrace.data.ingest import load_timeseries
from chaostrace.orchestrator.sweep import ALERT_THRESHOLD, build_grid, sweep
from chaostrace.phase.embedding import takens_embedding
from chaostrace.utils.manifest import write_manifest


def _parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _save_timeline(df: pd.DataFrame, tl: pd.DataFrame, outp: Path, *, threshold: float) -> tuple[Path, Path]:
    fig, ax1 = plt.subplots()
    t = df["time_s"].to_numpy(dtype=float)

    foil = df.get("foil_height_m", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    speed = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)

    ax1.plot(t, foil, label="foil_height_m")
    ax1.plot(t, speed, label="boat_speed")
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("signals")

    ax2 = ax1.twinx()
    ax2.plot(t, tl["score_mean"].to_numpy(dtype=float), label="score_mean")
    ax2.axhline(float(threshold), linestyle=":")
    ax2.set_ylabel("score")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    p1 = outp / "fig_timeline.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    foil_n = _norm01(foil)
    speed_n = _norm01(speed)
    inv = tl["score_invariant"].to_numpy(dtype=float)
    var = tl["score_variant"].to_numpy(dtype=float)

    ax.plot(t, foil_n, label="foil_norm")
    ax.plot(t, speed_n, label="speed_norm")
    ax.plot(t, inv, linewidth=3, label="score_invariant")
    ax.plot(t, var, linestyle="--", label="score_variant")

    inv_zone = inv > 0.6
    var_zone = (var > 0.5) & (inv < 0.4)
    ax.fill_between(t, 0.0, 1.0, where=inv_zone, alpha=0.08, label="invariant_zone")
    ax.fill_between(t, 0.0, 1.0, where=var_zone, alpha=0.10, label="variant_zone")

    ax.axhline(float(threshold), linestyle=":")
    ax.set_xlabel("time_s")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    p2 = outp / "fig_timeline_inv_var.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    return p1, p2


def _save_phase(df: pd.DataFrame, tl: pd.DataFrame, outp: Path, *, threshold: float) -> Path:
    x = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    X = takens_embedding(x, dim=3, lag=3)
    if len(X) == 0:
        fig = plt.figure()
        p = outp / "fig_phase.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p

    score = tl["score_mean"].to_numpy(dtype=float)
    score = score[-len(X) :]
    hi = score > float(threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[~hi, 0], X[~hi, 1], X[~hi, 2], s=3, alpha=0.5, c="0.6")
    ax.scatter(X[hi, 0], X[hi, 1], X[hi, 2], s=6, alpha=0.9, c="red")
    ax.set_title("Phase space (Takens embedding)")
    fig.tight_layout()
    p = outp / "fig_phase.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--runs", type=int, default=200, help="Number of sweep configurations to run")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for config sampling")
    ap.add_argument("--window-s", default="3,5,10,20")
    ap.add_argument("--drop-threshold", default="0.10,0.20,0.30,0.40")
    ap.add_argument("--emb-dim", default="3,4,5")
    ap.add_argument("--emb-lag", default="1,3,5,8,12")
    ap.add_argument("--preset", default="default", choices=["default", "invariants"])
    ap.add_argument("--plot-mode", default="dynamic", choices=["fixed", "dynamic"])
    args = ap.parse_args()

    df = load_timeseries(Path(args.input))

    cfgs = build_grid(
        window_s=_parse_list_floats(args.window_s),
        drop_threshold=_parse_list_floats(args.drop_threshold),
        emb_dim=_parse_list_ints(args.emb_dim),
        emb_lag=_parse_list_ints(args.emb_lag),
    )

    rng = np.random.default_rng(args.seed)
    if args.runs < len(cfgs):
        idx = rng.permutation(len(cfgs))[: args.runs]
        cfgs = [cfgs[i] for i in idx]

    metrics_df, timeline_df = sweep(df, cfgs, seed=args.seed)

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(outp / "metrics.csv", index=False, float_format="%.6f")
    timeline_df.to_csv(outp / "anomalies.csv", index=False, float_format="%.6f")

    run_choice = int(metrics_df.sort_values(["alert_frac_dyn", "alert_frac", "run_id"]).iloc[0]["run_id"])
    tl = timeline_df[timeline_df["run_id"] == run_choice].reset_index(drop=True)

    if args.plot_mode == "dynamic" and "alert_threshold_dyn" in metrics_df.columns:
        threshold = float(metrics_df.loc[metrics_df["run_id"] == run_choice, "alert_threshold_dyn"].iloc[0])
    else:
        threshold = float(ALERT_THRESHOLD)

    _save_phase(df, tl, outp, threshold=threshold)
    _save_timeline(df, tl, outp, threshold=threshold)

    manifest_params = {
        "input": args.input,
        "runs": int(args.runs),
        "seed": int(args.seed),
        "preset": args.preset,
        "plot_mode": args.plot_mode,
        "run_choice": int(run_choice),
        "grid": [asdict(c) for c in cfgs],
    }
    write_manifest(
        outp,
        params=manifest_params,
        files=["metrics.csv", "anomalies.csv", "fig_phase.png", "fig_timeline.png", "fig_timeline_inv_var.png"],
    )


if __name__ == "__main__":
    main()
