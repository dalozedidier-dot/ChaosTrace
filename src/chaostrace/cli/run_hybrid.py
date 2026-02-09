from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chaostrace.data.ingest import load_timeseries
from chaostrace.features.windowing import estimate_sample_hz
from chaostrace.orchestrator.sweep import build_grid, sweep
from chaostrace.phase.embedding import takens_embedding
from chaostrace.utils.manifest import write_manifest
from chaostrace.hybrid.causal_var import compute_causal_drift
from chaostrace.hybrid.fusion import fuse_scores
from chaostrace.hybrid.matrix_profile import compute_matrix_profile
from chaostrace.hybrid.metrics import event_level_metrics, pointwise_prf


def _parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _save_timeline(df: pd.DataFrame, tl: pd.DataFrame, outp: Path, *, threshold: float) -> Path:
    fig, ax1 = plt.subplots()
    t = df["time_s"].to_numpy(dtype=float)
    foil = df.get("foil_height_m", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    speed = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)

    ax1.plot(t, foil, label="foil_height_m")
    ax1.plot(t, speed, label="boat_speed")
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("signals")

    ax2 = ax1.twinx()
    ax2.plot(t, tl["score_fused"].to_numpy(dtype=float), label="score_fused")
    ax2.axhline(float(threshold), linestyle=":")
    ax2.set_ylabel("score")

    alert = tl["alert"].to_numpy(dtype=int) > 0
    ax2.scatter(t[alert], tl.loc[alert, "score_fused"].to_numpy(dtype=float), s=8, alpha=0.7)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    p = outp / "fig_timeline_hybrid.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_phase(df: pd.DataFrame, tl: pd.DataFrame, outp: Path, *, threshold: float) -> Path:
    x = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    X = takens_embedding(x, dim=3, lag=3)
    if len(X) == 0:
        fig = plt.figure()
        p = outp / "fig_phase_hybrid.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p

    score = tl["score_fused"].to_numpy(dtype=float)
    score = score[-len(X) :]
    hi = score > float(threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[~hi, 0], X[~hi, 1], X[~hi, 2], s=3, alpha=0.5, c="0.6")
    ax.scatter(X[hi, 0], X[hi, 1], X[hi, 2], s=6, alpha=0.9, c="red")
    ax.set_title("Phase space (Takens embedding)\nHybrid thresholded points")
    fig.tight_layout()
    p = outp / "fig_phase_hybrid.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--runs", type=int, default=60, help="Number of chaos configs to evaluate before fusion")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window-s", default="3,5,10")
    ap.add_argument("--drop-threshold", default="0.30,0.40")
    ap.add_argument("--emb-dim", default="3,4,5")
    ap.add_argument("--emb-lag", default="5,8,12")

    ap.add_argument("--enable-mp", action="store_true", help="Enable Matrix Profile component (requires -e '.[mp]')")
    ap.add_argument("--mp-col", default="boat_speed")
    ap.add_argument("--enable-causal", action="store_true", help="Enable causal drift component")
    ap.add_argument("--causal-cols", default="boat_speed,foil_height_m")

    ap.add_argument("--model", default="", help="Optional DL model directory (requires -e '.[dl]')")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--early-window-s", type=float, default=2.0, help="Event-level early window")

    # Fusion thresholding controls (useful for early warnings without inflating thresholds near events)
    ap.add_argument("--baseline-s", type=float, default=20.0, help="Baseline length in seconds (early segment)")
    ap.add_argument("--baseline-frac", type=float, default=0.20, help="Baseline length as fraction of run duration")
    ap.add_argument("--baseline-percentile", type=float, default=99.5, help="Percentile used to set dynamic threshold")
    ap.add_argument("--threshold-min", type=float, default=0.10, help="Min clamp for dynamic threshold")
    ap.add_argument("--threshold-max", type=float, default=0.99, help="Max clamp for dynamic threshold")
    ap.add_argument("--merge-gap-s", type=float, default=0.20, help="Merge gap for postprocessed alert events")
    ap.add_argument("--min-duration-s", type=float, default=0.30, help="Min duration for alert events")

    args = ap.parse_args()

    df = load_timeseries(Path(args.input))
    df = df.sort_values("time_s").reset_index(drop=True)
    time_s = df["time_s"].to_numpy(dtype=float)
    hz = estimate_sample_hz(time_s)

    cfgs = build_grid(
        window_s=_parse_list_floats(args.window_s),
        drop_threshold=_parse_list_floats(args.drop_threshold),
        emb_dim=_parse_list_ints(args.emb_dim),
        emb_lag=_parse_list_ints(args.emb_lag),
    )

    rng = np.random.default_rng(int(args.seed))
    if int(args.runs) < len(cfgs):
        idx = rng.permutation(len(cfgs))[: int(args.runs)]
        cfgs = [cfgs[i] for i in idx]

    metrics_df, timeline_df = sweep(df, cfgs, seed=int(args.seed))

    best = None
    for run_id in metrics_df["run_id"].to_list():
        tl = timeline_df[timeline_df["run_id"] == run_id]
        prf = pointwise_prf(tl["alert_dyn"].to_numpy(dtype=int) > 0, tl["is_drop"].to_numpy(dtype=float) > 0.5)
        alert_frac_dyn = float(metrics_df.loc[metrics_df["run_id"] == run_id, "alert_frac_dyn"].iloc[0])
        key = (float(prf["f1"]), -alert_frac_dyn)
        if best is None or key > best[0]:
            best = (key, int(run_id), prf)
    assert best is not None
    _key, run_choice, prf_chaos = best

    tlc = timeline_df[timeline_df["run_id"] == run_choice].reset_index(drop=True)
    score_chaos = tlc["score_mean"].to_numpy(dtype=float)
    is_drop = tlc["is_drop"].to_numpy(dtype=float) > 0.5
    foil = df.get("foil_height_m", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)

    score_mp = None
    if bool(args.enable_mp):
        window_n = int(metrics_df.loc[metrics_df["run_id"] == run_choice, "window_n"].iloc[0])
        mp_res = compute_matrix_profile(df, col=str(args.mp_col), window_n=window_n)
        score_mp = mp_res.score

    score_causal = None
    if bool(args.enable_causal):
        cols = [c.strip() for c in str(args.causal_cols).split(",") if c.strip()]
        window_n = int(metrics_df.loc[metrics_df["run_id"] == run_choice, "window_n"].iloc[0])
        baseline_n = max(int(round(20.0 * hz)), window_n)
        cd = compute_causal_drift(df, cols=cols, window_n=window_n, baseline_n=baseline_n)
        score_causal = cd.score

    score_dl = None
    model_dir = str(args.model).strip()
    if model_dir:
        try:
            from chaostrace.hybrid.dl.infer import infer_series
        except Exception as e:
            raise RuntimeError("DL inference requires optional dependency 'torch'. Install with: pip install -e '.[dl]'") from e
        dl_res = infer_series(df, model_dir=Path(model_dir), device=str(args.device))
        score_dl = dl_res.score

    cfg_row = metrics_df.loc[metrics_df["run_id"] == run_choice].iloc[0]
    drop_threshold = float(cfg_row["drop_threshold"])

    fused = fuse_scores(
        time_s=time_s,
        score_chaos=score_chaos,
        is_drop=is_drop.astype(float),
        foil_height=foil,
        drop_threshold=drop_threshold,
        score_mp=score_mp,
        score_causal=score_causal,
        score_dl=score_dl,
        baseline_s=float(args.baseline_s),
        baseline_frac=float(args.baseline_frac),
        baseline_percentile=float(args.baseline_percentile),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        merge_gap_s=float(args.merge_gap_s),
        min_duration_s=float(args.min_duration_s),
    )


    alert = fused.alert_mask
    prf = pointwise_prf(alert, is_drop)
    ev = event_level_metrics(time_s, alert, is_drop, early_window_s=float(args.early_window_s))

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    tl_out = pd.DataFrame(
        {
            "time_s": time_s,
            "is_drop": is_drop.astype(int),
            "score_chaos": score_chaos,
            "score_fused": fused.score_fused,
            "alert": alert.astype(int),
            "threshold": float(fused.threshold),
        }
    )
    for k, s in fused.components.items():
        if k == "chaos":
            continue
        tl_out[f"score_{k}"] = np.asarray(s, dtype=float)

    tl_out.to_csv(outp / "anomalies_hybrid.csv", index=False, float_format="%.6f")

    m = {
        "run_choice": int(run_choice),
        "window_s": float(cfg_row["window_s"]),
        "window_n": int(cfg_row["window_n"]),
        "drop_threshold": float(cfg_row["drop_threshold"]),
        "emb_dim": int(cfg_row["emb_dim"]),
        "emb_lag": int(cfg_row["emb_lag"]),
        "enable_mp": bool(args.enable_mp),
        "enable_causal": bool(args.enable_causal),
        "has_dl": bool(model_dir),
        "threshold": float(fused.threshold),
        "tp": prf["tp"],
        "fp": prf["fp"],
        "fn": prf["fn"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "alert_frac": float(np.mean(alert.astype(float))),
        "threshold": float(fused.threshold),
        "drop_events": ev.drop_events,
        "alert_events": ev.alert_events,
        "matched_drop_events": ev.matched_drop_events,
        "drop_event_recall": ev.drop_event_recall,
        "alert_event_precision": ev.alert_event_precision,
        "lead_s_median": ev.lead_s_median,
        "lead_s_max": ev.lead_s_max,
        "weights": fused.weights,
    }
    (outp / "metrics_hybrid.json").write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")

    explain_path = outp / "explain_hybrid.jsonl"
    with explain_path.open("w", encoding="utf-8") as f:
        for i in np.flatnonzero(alert):
            row = {
                "time_s": float(time_s[int(i)]),
                "score_fused": float(fused.score_fused[int(i)]),
                "threshold": float(fused.threshold),
                "weights": fused.weights,
                "components": {k: float(np.asarray(v, dtype=float)[int(i)]) for k, v in fused.components.items()},
            }
            f.write(json.dumps(row) + "\n")

    fig_tl = _save_timeline(df, tl_out, outp, threshold=float(fused.threshold))
    fig_ph = _save_phase(df, tl_out, outp, threshold=float(fused.threshold))

    params = {
        "input": args.input,
        "seed": int(args.seed),
        "runs": int(args.runs),
        "grid": [asdict(c) for c in cfgs],
        "run_choice": int(run_choice),
        "hybrid": {
            "enable_mp": bool(args.enable_mp),
            "mp_col": str(args.mp_col),
            "enable_causal": bool(args.enable_causal),
            "causal_cols": str(args.causal_cols),
            "model": model_dir,
            "device": str(args.device),
            "early_window_s": float(args.early_window_s),
        },
    }
    write_manifest(
        outp,
        params=params,
        files=[
            "anomalies_hybrid.csv",
            "metrics_hybrid.json",
            "explain_hybrid.jsonl",
            Path(fig_tl).name,
            Path(fig_ph).name,
        ],
    )


if __name__ == "__main__":
    main()
