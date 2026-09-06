from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chaostrace.data.ingest import load_timeseries
from chaostrace.orchestrator.sweep import ALERT_THRESHOLD, build_grid, sweep
from chaostrace.utils.manifest import write_manifest
from chaostrace.viz.static import save_phase, save_timeline


def _parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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

    save_phase(df, tl, outp, threshold=threshold)
    save_timeline(df, tl, outp, threshold=threshold)

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
