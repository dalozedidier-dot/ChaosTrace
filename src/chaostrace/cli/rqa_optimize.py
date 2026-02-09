from __future__ import annotations

"""CLI: optimize RQA parameters (grid or optuna).

This tool helps you stop hand-tuning (m, tau, rr_target) by producing a
reproducible search trace.

Two objectives are available:
- delta_det: maximize mean(DET stable) - mean(DET drop)
- f1: maximize pointwise F1 using an interpretable early-warning score

Example
-------
python -m chaostrace.cli.rqa_optimize \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/rqa_opt \
  --method grid \
  --objective delta_det
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from chaostrace.data.ingest import load_timeseries
from chaostrace.features.windowing import estimate_sample_hz
from chaostrace.rqa.advanced import RQAAdvancedConfig, compute_rqa_advanced_from_series, rqa_early_warning_score


@dataclass(frozen=True)
class Candidate:
    emb_dim: int
    emb_lag: int
    rr_target: float
    window_s: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimize RQA parameters (grid or optuna).")
    p.add_argument("--input", required=True, help="Input CSV/JSON.")
    p.add_argument("--out", required=True, help="Output directory.")

    p.add_argument("--method", choices=["grid", "optuna"], default="grid")
    p.add_argument("--objective", choices=["delta_det", "f1"], default="delta_det")

    p.add_argument("--series", default="foil_height_m", help="Series used for RQA.")
    p.add_argument("--drop-threshold", type=float, default=0.30)

    p.add_argument("--window-s", default="3,5,10", help="Comma-separated window sizes in seconds.")
    p.add_argument("--emb-dim", default="3,4,5,6", help="Comma-separated embedding dimensions.")
    p.add_argument("--emb-lag", default="5,8,12", help="Comma-separated embedding lags.")
    p.add_argument("--rr-target", default="0.02,0.03,0.04", help="Comma-separated RR targets (FRR).")

    p.add_argument("--step-frac", type=float, default=0.5)
    p.add_argument("--max-points", type=int, default=600)
    p.add_argument("--trials", type=int, default=40, help="Optuna trials (if method=optuna).")

    return p


def _parse_list(s: str, cast) -> List[Any]:
    out: List[Any] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast(part))
    return out


def _pointwise_prf(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & (~truth)))
    fn = int(np.sum((~pred) & truth))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def _evaluate_candidate(
    df: pd.DataFrame,
    *,
    hz: float,
    is_drop: np.ndarray,
    cand: Candidate,
    objective: str,
    step_frac: float,
    max_points: int,
) -> Dict[str, Any]:
    window_n = int(max(20, round(float(cand.window_s) * hz)))
    step_n = int(max(1, round(window_n * float(step_frac))))

    cfg = RQAAdvancedConfig(
        emb_dim=int(cand.emb_dim),
        emb_lag=int(cand.emb_lag),
        threshold_by="frr",
        rr_target=float(cand.rr_target),
        epsilon=None,
        theiler_window=None,
        l_min=2,
        v_min=2,
        max_points=int(max_points),
        rng_seed=7,
        enable_network=False,
    )

    det_vals: List[float] = []
    lam_vals: List[float] = []
    trend_vals: List[float] = []
    early_vals: List[float] = []
    drop_fracs: List[float] = []

    series = df["series"].to_numpy(dtype=float)
    for start in range(0, len(df) - window_n + 1, step_n):
        end = start + window_n
        x = series[start:end]
        m = compute_rqa_advanced_from_series(x, config=cfg)
        det_vals.append(float(m.get("det", 0.0)))
        lam_vals.append(float(m.get("lam", 0.0)))
        trend_vals.append(float(m.get("trend", 0.0)))
        early_vals.append(float(rqa_early_warning_score(m)))
        drop_fracs.append(float(np.mean(is_drop[start:end].astype(float))))

    det_arr = np.asarray(det_vals, dtype=float)
    lam_arr = np.asarray(lam_vals, dtype=float)
    trend_arr = np.asarray(trend_vals, dtype=float)
    early_arr = np.asarray(early_vals, dtype=float)
    drop_frac_arr = np.asarray(drop_fracs, dtype=float)

    # stable windows: very low drop fraction
    stable = drop_frac_arr <= 0.05
    drop = drop_frac_arr >= 0.50

    if objective == "delta_det":
        det_stable = float(np.mean(det_arr[stable])) if np.any(stable) else float(np.mean(det_arr))
        det_drop = float(np.mean(det_arr[drop])) if np.any(drop) else float(np.mean(det_arr))
        score = det_stable - det_drop
        return {
            "objective": "delta_det",
            "score": float(score),
            "det_stable": det_stable,
            "det_drop": det_drop,
            "lam_stable": float(np.mean(lam_arr[stable])) if np.any(stable) else float(np.mean(lam_arr)),
            "lam_drop": float(np.mean(lam_arr[drop])) if np.any(drop) else float(np.mean(lam_arr)),
            "trend_abs_mean": float(np.mean(np.abs(trend_arr))),
        }

    # objective == f1 (pointwise at window-level)
    # Convert window scores to a pointwise series by forward-filling to cover the window range.
    # This is simple and keeps causality (we only assign score to times within/after the window start).
    point_scores = np.zeros(len(df), dtype=float)
    count = np.zeros(len(df), dtype=float)

    idx = 0
    for start in range(0, len(df) - window_n + 1, step_n):
        end = start + window_n
        s = float(early_arr[idx])
        point_scores[start:end] += s
        count[start:end] += 1.0
        idx += 1

    count = np.where(count <= 0, 1.0, count)
    point_scores = point_scores / count

    # choose best threshold among quantiles
    qs = np.linspace(0.80, 0.99, 16)
    best = {"f1": -1.0, "thr": 0.5, "prec": 0.0, "rec": 0.0}
    for q in qs:
        thr = float(np.quantile(point_scores, q))
        pred = point_scores > thr
        prec, rec, f1 = _pointwise_prf(pred, is_drop)
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": thr, "prec": prec, "rec": rec}

    return {
        "objective": "f1",
        "score": float(best["f1"]),
        "threshold": float(best["thr"]),
        "precision": float(best["prec"]),
        "recall": float(best["rec"]),
        "f1": float(best["f1"]),
    }


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    df0 = load_timeseries(args.input)
    if args.series not in df0.columns:
        raise ValueError(f"Missing series column: {args.series!r}")

    df = df0.copy()
    df["series"] = pd.to_numeric(df[str(args.series)], errors="coerce")

    time_s = df["time_s"].to_numpy(dtype=float)
    hz = float(estimate_sample_hz(time_s))

    foil = df["foil_height_m"].to_numpy(dtype=float)
    is_drop = np.isfinite(foil) & (foil < float(args.drop_threshold))

    windows = _parse_list(str(args.window_s), float)
    emb_dims = _parse_list(str(args.emb_dim), int)
    emb_lags = _parse_list(str(args.emb_lag), int)
    rr_targets = _parse_list(str(args.rr_target), float)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    if args.method == "grid":
        for w in windows:
            for m in emb_dims:
                for tau in emb_lags:
                    for rr in rr_targets:
                        cand = Candidate(emb_dim=int(m), emb_lag=int(tau), rr_target=float(rr), window_s=float(w))
                        row = {
                            "window_s": float(w),
                            "emb_dim": int(m),
                            "emb_lag": int(tau),
                            "rr_target": float(rr),
                        }
                        row.update(
                            _evaluate_candidate(
                                df,
                                hz=hz,
                                is_drop=is_drop,
                                cand=cand,
                                objective=str(args.objective),
                                step_frac=float(args.step_frac),
                                max_points=int(args.max_points),
                            )
                        )
                        results.append(row)
                        if best_row is None or float(row["score"]) > float(best_row["score"]):
                            best_row = dict(row)
    else:
        try:
            import optuna
        except Exception as e:
            raise RuntimeError("Optuna is not installed. Install with: pip install 'chaostrace[opt]'") from e

        def objective(trial: "optuna.Trial") -> float:
            w = trial.suggest_categorical("window_s", windows)
            m = trial.suggest_categorical("emb_dim", emb_dims)
            tau = trial.suggest_categorical("emb_lag", emb_lags)
            rr = trial.suggest_categorical("rr_target", rr_targets)
            cand = Candidate(emb_dim=int(m), emb_lag=int(tau), rr_target=float(rr), window_s=float(w))
            out = _evaluate_candidate(
                df,
                hz=hz,
                is_drop=is_drop,
                cand=cand,
                objective=str(args.objective),
                step_frac=float(args.step_frac),
                max_points=int(args.max_points),
            )
            trial.set_user_attr("detail", out)
            return float(out["score"])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.trials))

        for tr in study.trials:
            detail = tr.user_attrs.get("detail", {})
            row = dict(tr.params)
            row.update(detail)
            results.append(row)

        best = study.best_trial
        best_row = dict(best.params)
        best_row.update(best.user_attrs.get("detail", {}))

    df_res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    df_res.to_csv(out_dir / "rqa_param_optimization.csv", index=False, float_format="%.6f")

    best_out: Dict[str, Any] = {
        "input": str(args.input),
        "hz": hz,
        "objective": str(args.objective),
        "method": str(args.method),
        "best": best_row or {},
    }
    (out_dir / "best_rqa_params.json").write_text(
        json.dumps(best_out, indent=2, sort_keys=True), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
