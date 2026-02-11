from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from chaostrace.features.windowing import estimate_sample_hz
from chaostrace.hybrid.metrics import event_level_metrics, pointwise_prf
from chaostrace.orchestrator.sweep import SweepConfig, build_grid, sweep


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError("CSV must contain a 'time_s' column (seconds).")
    df = df.sort_values("time_s").reset_index(drop=True)
    return df


def _require_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col!r}")
    return df[col].to_numpy(dtype=float)


def _renorm_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, float(v)) for v in w.values()))
    if s <= 1e-12:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}


def _default_weights(*, has_dl: bool, has_mp: bool, has_causal: bool) -> Dict[str, float]:
    w = {
        "chaos": 0.65,
        "dl": 0.20 if has_dl else 0.0,
        "mp": 0.10 if has_mp else 0.0,
        "causal": 0.05 if has_causal else 0.0,
    }
    return _renorm_weights(w)


def _event_f1(prec: float, rec: float) -> float:
    p = float(prec)
    r = float(rec)
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _dynamic_threshold(
    scores: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    percentile: float,
    thr_min: float,
    thr_max: float,
) -> float:
    """Compute a robust threshold from a baseline segment.

    Important: if the user's thr_min is above the maximum achievable score, we clamp the
    threshold to the maximum score so alerts are still possible (otherwise you'd get
    a silent run with 0 alerts).
    """
    s = np.asarray(scores, dtype=float)
    m = np.asarray(baseline_mask, dtype=bool)
    if s.shape != m.shape:
        raise ValueError("scores and baseline_mask must have same shape")

    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return float(thr_min)

    base = s[m]
    base = base[np.isfinite(base)]
    if base.size == 0:
        thr = float(np.percentile(finite, float(percentile)))
    else:
        thr = float(np.percentile(base, float(percentile)))

    if not np.isfinite(thr):
        thr = float(thr_min)

    # Standard clip first.
    thr = float(np.clip(thr, float(thr_min), float(thr_max)))

    # If thr is above what scores can reach, clamp to max score (minus epsilon).
    max_valid = float(np.max(finite))
    eps = 1e-9
    max_valid = max(max_valid - eps, 0.0)
    thr_floor = min(float(thr_min), max_valid)
    thr = min(thr, max_valid)
    thr = max(thr, thr_floor)

    return float(thr)


def _postprocess_alerts(
    alert: np.ndarray,
    time_s: np.ndarray,
    *,
    merge_gap_s: float,
    min_duration_s: float,
) -> Tuple[np.ndarray, int]:
    """Merge short gaps and drop too-short alert events."""
    a = np.asarray(alert, dtype=bool).copy()
    t = np.asarray(time_s, dtype=float)
    if a.shape != t.shape:
        raise ValueError("alert and time_s must have same shape")

    # Identify events
    events: List[Tuple[int, int]] = []
    in_ev = False
    s_idx = 0
    for i, v in enumerate(a):
        if v and not in_ev:
            in_ev = True
            s_idx = i
        if in_ev and (not v or i == len(a) - 1):
            e_idx = i if not v else i + 1
            events.append((s_idx, e_idx))
            in_ev = False

    if not events:
        return a, 0

    # Merge close events separated by small gaps in time.
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = events[0]
    for s, e in events[1:]:
        gap = float(t[s] - t[cur_e - 1]) if cur_e - 1 < len(t) else float("inf")
        if gap <= float(merge_gap_s):
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Drop too-short events.
    keep: List[Tuple[int, int]] = []
    for s, e in merged:
        dur = float(t[e - 1] - t[s]) if e - 1 < len(t) else 0.0
        if dur >= float(min_duration_s):
            keep.append((s, e))

    a[:] = False
    for s, e in keep:
        a[s:e] = True

    return a, len(keep)


def _mask_baseline(time_s: np.ndarray, baseline_n: int) -> np.ndarray:
    m = np.zeros_like(time_s, dtype=bool)
    n = int(np.clip(int(baseline_n), 0, len(m)))
    if n > 0:
        m[:n] = True
    return m


def _compute_is_drop_from_foil(foil_height_m: np.ndarray, *, drop_threshold: float) -> np.ndarray:
    foil = np.asarray(foil_height_m, dtype=float)
    return (foil < float(drop_threshold)).astype(bool)


def _safe_load_model_scores(
    model_dir: Path,
    *,
    input_csv: str,
    cols: List[str],
    device: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.dl_infer import infer_scores

        scores = infer_scores(
            model_dir=model_dir,
            input_csv=input_csv,
            cols=cols,
            device=device,
        )
        return np.asarray(scores, dtype=float), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _safe_load_mp_scores(
    input_csv: str,
    *,
    col: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.mp_baseline import mp_score

        s = mp_score(input_csv=input_csv, col=col)
        return np.asarray(s, dtype=float), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _safe_load_causal_scores(
    input_csv: str,
    *,
    cols: List[str],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.causal_baseline import causal_score

        s = causal_score(input_csv=input_csv, cols=cols)
        return np.asarray(s, dtype=float), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


@dataclass(frozen=True)
class Picked:
    run_id: int
    idx: int
    metrics: Dict[str, Any]


def _pick_grid_row(
    grid_rows: List[Dict[str, Any]],
    *,
    pick_mode: str,
) -> int:
    if not grid_rows:
        raise ValueError("grid_rows is empty")

    if pick_mode == "max_f1_event":
        key = lambda i: (-float(grid_rows[i].get("f1_event", 0.0)), float(grid_rows[i].get("alert_frac", 1.0)))
    elif pick_mode == "max_lead":
        key = lambda i: (
            -float(grid_rows[i].get("lead_s_max", 0.0)),
            -float(grid_rows[i].get("f1_event", 0.0)),
            float(grid_rows[i].get("alert_frac", 1.0)),
        )
    else:
        raise ValueError(f"Unknown pick_mode: {pick_mode!r}")

    best = min(range(len(grid_rows)), key=key)
    return int(best)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run hybrid early-warning scoring and evaluate event metrics.")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--runs", type=int, default=40, help="Number of chaos sweep configs to try")
    p.add_argument("--seed", type=int, default=7, help="Seed for sweep RNG")

    p.add_argument("--enable-mp", action="store_true", help="Enable simple Markov/MP baseline score component")
    p.add_argument("--mp-col", default="foil_height_m", help="Column used by MP baseline")

    p.add_argument("--enable-causal", action="store_true", help="Enable causal baseline component")
    p.add_argument("--causal-cols", default="boat_speed,foil_height_m", help="Comma-separated causal columns")

    p.add_argument("--model", default="", help="DL model directory (trained by train_hybrid)")
    p.add_argument("--device", default="cpu", help="Device for DL inference (cpu)")

    p.add_argument("--pick", default="max_f1_event", choices=["max_f1_event", "max_lead"], help="Grid pick mode")

    p.add_argument("--baseline-s", type=float, default=10.0, help="Baseline length in seconds")
    p.add_argument("--baseline-frac", type=float, default=0.0, help="Baseline length as fraction of series")
    p.add_argument("--baseline-percentile", type=float, default=99.5, help="Percentile on baseline to set threshold")
    p.add_argument("--threshold-min", type=float, default=0.05, help="Minimum threshold clamp")
    p.add_argument("--threshold-max", type=float, default=0.97, help="Maximum threshold clamp")

    p.add_argument("--gate-dl", type=float, default=0.0, help="If >0, DL must exceed this to allow alerts")
    p.add_argument("--gate-causal", type=float, default=0.0, help="If >0, causal must exceed this to allow alerts")
    p.add_argument("--gate-chaos", type=float, default=0.0, help="If >0, chaos must exceed this to allow alerts")

    p.add_argument("--merge-gap-s", type=float, default=0.30, help="Merge alert gaps smaller than this (seconds)")
    p.add_argument("--min-duration-s", type=float, default=0.10, help="Drop alert events shorter than this (seconds)")

    p.add_argument("--early-window-s", type=float, default=2.0, help="Early-warning window before drop (seconds)")
    return p


def _select_cfgs(args: argparse.Namespace) -> List[SweepConfig]:
    grid = build_grid(
        window_s=[3, 5, 10, 20],
        drop_threshold=[0.10, 0.20, 0.30, 0.40],
        emb_dim=[3, 4, 5],
        emb_lag=[1, 3, 5, 8, 12],
    )

    # Deterministic subset selection for CI speed.
    rng = np.random.default_rng(int(args.seed))
    n = int(max(1, int(args.runs)))
    if n >= len(grid):
        return list(grid)

    idx = rng.choice(np.arange(len(grid), dtype=int), size=n, replace=False)
    return [grid[int(i)] for i in idx]


def _apply_gating(
    fused: np.ndarray,
    *,
    score_chaos: np.ndarray,
    score_dl: Optional[np.ndarray],
    score_causal: Optional[np.ndarray],
    gate_chaos: float,
    gate_dl: float,
    gate_causal: float,
) -> np.ndarray:
    m = np.ones_like(fused, dtype=bool)

    if float(gate_chaos) > 0.0:
        m &= np.asarray(score_chaos, dtype=float) >= float(gate_chaos)

    if score_dl is not None and float(gate_dl) > 0.0:
        m &= np.asarray(score_dl, dtype=float) >= float(gate_dl)

    if score_causal is not None and float(gate_causal) > 0.0:
        m &= np.asarray(score_causal, dtype=float) >= float(gate_causal)

    return np.asarray(fused, dtype=float) * m.astype(float)


def _fuse_components(
    score_chaos: np.ndarray,
    *,
    score_mp: Optional[np.ndarray],
    score_causal: Optional[np.ndarray],
    score_dl: Optional[np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fused = np.clip(float(weights.get("chaos", 0.0)) * np.asarray(score_chaos, dtype=float), 0.0, 1.0)
    components: Dict[str, np.ndarray] = {"chaos": np.asarray(score_chaos, dtype=float)}

    if score_mp is not None and float(weights.get("mp", 0.0)) > 0:
        components["mp"] = np.asarray(score_mp, dtype=float)
        fused += float(weights["mp"]) * components["mp"]

    if score_causal is not None and float(weights.get("causal", 0.0)) > 0:
        components["causal"] = np.asarray(score_causal, dtype=float)
        fused += float(weights["causal"]) * components["causal"]

    if score_dl is not None and float(weights.get("dl", 0.0)) > 0:
        components["dl"] = np.asarray(score_dl, dtype=float)
        fused += float(weights["dl"]) * components["dl"]

    fused = np.clip(fused, 0.0, 1.0)
    return fused, components


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    df = _load_csv(args.input)
    t = df["time_s"].to_numpy(dtype=float)
    hz = float(estimate_sample_hz(t))

    foil = _require_col(df, "foil_height_m")
    cfgs = _select_cfgs(args)

    # Run chaos suite for all cfgs at once
    _metrics_df, tl_df = sweep(df, cfgs, seed=int(args.seed))
    if tl_df.empty:
        raise RuntimeError("Sweep produced empty timeline output")

    # Baseline sizing
    baseline_n = int(round(float(args.baseline_s) * hz))
    if float(args.baseline_frac) > 0.0:
        baseline_n = max(baseline_n, int(round(float(args.baseline_frac) * len(df))))
    baseline_n = int(np.clip(baseline_n, 1, len(df)))

    causal_cols = [c.strip() for c in str(args.causal_cols).split(",") if c.strip()]
    mp_col = str(args.mp_col).strip()

    model_dir = Path(str(args.model)) if args.model else None

    grid_rows: List[Dict[str, Any]] = []

    # Optional components
    score_dl: Optional[np.ndarray] = None
    dl_error: Optional[str] = None
    if model_dir is not None and model_dir.exists():
        score_dl, dl_error = _safe_load_model_scores(
            model_dir,
            input_csv=str(args.input),
            cols=["boat_speed", "foil_height_m"],
            device=str(args.device),
        )

    score_mp: Optional[np.ndarray] = None
    mp_error: Optional[str] = None
    if bool(args.enable_mp):
        score_mp, mp_error = _safe_load_mp_scores(str(args.input), col=mp_col)

    score_causal: Optional[np.ndarray] = None
    causal_error: Optional[str] = None
    if bool(args.enable_causal):
        score_causal, causal_error = _safe_load_causal_scores(str(args.input), cols=causal_cols)

    # Ground truth is drop threshold from picked cfg.
    # We will recompute is_drop per cfg so it is coherent with cfg.drop_threshold.
    outp = Path(str(args.out))
    outp.mkdir(parents=True, exist_ok=True)

    # Build a baseline mask once (time based), used for thresholding the fused score.
    baseline_mask = _mask_baseline(t, baseline_n)

    has_dl = score_dl is not None
    has_mp = score_mp is not None
    has_causal = score_causal is not None
    weights = _default_weights(has_dl=has_dl, has_mp=has_mp, has_causal=has_causal)

    pick_mode = str(args.pick)

    # Evaluate each chaos cfg row with coherent recompute.
    for run_id, cfg in enumerate(cfgs):
        # Extract chaos score timeline for this cfg.
        # tl_df contains one row per timepoint per cfg; sweep adds run_id.
        sub = tl_df[tl_df["run_id"] == run_id]
        if sub.empty:
            continue

        # Ensure aligned ordering by time_s.
        sub = sub.sort_values("time_s").reset_index(drop=True)

        score_chaos = sub["score_mean"].to_numpy(dtype=float)
        if len(score_chaos) != len(df):
            # If mismatch, align by time; but in our pipeline it should match exactly.
            # For safety, interpolate chaos score onto input times.
            score_chaos = np.interp(t, sub["time_s"].to_numpy(dtype=float), score_chaos)

        fused, components = _fuse_components(
            score_chaos,
            score_mp=score_mp,
            score_causal=score_causal,
            score_dl=score_dl,
            weights=weights,
        )

        fused = _apply_gating(
            fused,
            score_chaos=score_chaos,
            score_dl=score_dl,
            score_causal=score_causal,
            gate_chaos=float(args.gate_chaos),
            gate_dl=float(args.gate_dl),
            gate_causal=float(args.gate_causal),
        )

        # Threshold from baseline
        thr = _dynamic_threshold(
            fused,
            baseline_mask,
            percentile=float(args.baseline_percentile),
            thr_min=float(args.threshold_min),
            thr_max=float(args.threshold_max),
        )

        alert = fused >= float(thr)

        # Postprocess alert events
        alert, alert_events = _postprocess_alerts(
            alert,
            t,
            merge_gap_s=float(args.merge_gap_s),
            min_duration_s=float(args.min_duration_s),
        )

        # Recompute is_drop for this cfg
        is_drop = _compute_is_drop_from_foil(foil, drop_threshold=float(cfg.drop_threshold))

        # Pointwise metrics
        prf = pointwise_prf(is_drop.astype(int), alert.astype(int))

        # Event-level metrics
        early_window_s = float(args.early_window_s)
        ev = event_level_metrics(time_s=t, is_drop=is_drop, alert=alert, early_window_s=early_window_s)
        f1_event = _event_f1(ev.alert_event_precision, ev.drop_event_recall)

        grid_rows.append(
            {
                "run_id": int(run_id),
                "window_s": float(cfg.window_s),
                "drop_threshold": float(cfg.drop_threshold),
                "emb_dim": int(cfg.emb_dim),
                "emb_lag": int(cfg.emb_lag),
                "threshold": float(thr),
                "alert_events": int(alert_events),
                "alert_frac": float(np.mean(alert.astype(float))),
                "f1_event": float(f1_event),
                "drop_events": int(ev.drop_events),
                "matched_drop_events": int(ev.matched_drop_events),
                "lead_s_median": float(ev.lead_s_median),
                "lead_s_max": float(ev.lead_s_max),
            }
            | {k: float(v) for k, v in prf.items()}
        )

    if not grid_rows:
        raise RuntimeError("No grid rows evaluated (empty grid_rows)")

    picked_idx = _pick_grid_row(grid_rows, pick_mode=pick_mode)
    picked_run = grid_rows[picked_idx]

    picked = Picked(
        run_id=int(picked_run["run_id"]),
        idx=int(picked_idx),
        metrics=dict(picked_run),
    )

    # Recompute and persist artifacts for the picked run_id.
    cfg = cfgs[int(picked.run_id)]
    sub = tl_df[tl_df["run_id"] == int(picked.run_id)].sort_values("time_s").reset_index(drop=True)
    score_chaos = sub["score_mean"].to_numpy(dtype=float)
    if len(score_chaos) != len(df):
        score_chaos = np.interp(t, sub["time_s"].to_numpy(dtype=float), score_chaos)

    fused, components = _fuse_components(
        score_chaos,
        score_mp=score_mp,
        score_causal=score_causal,
        score_dl=score_dl,
        weights=weights,
    )

    fused = _apply_gating(
        fused,
        score_chaos=score_chaos,
        score_dl=score_dl,
        score_causal=score_causal,
        gate_chaos=float(args.gate_chaos),
        gate_dl=float(args.gate_dl),
        gate_causal=float(args.gate_causal),
    )

    thr = _dynamic_threshold(
        fused,
        baseline_mask,
        percentile=float(args.baseline_percentile),
        thr_min=float(args.threshold_min),
        thr_max=float(args.threshold_max),
    )

    alert = fused >= float(thr)
    alert, alert_events = _postprocess_alerts(
        alert,
        t,
        merge_gap_s=float(args.merge_gap_s),
        min_duration_s=float(args.min_duration_s),
    )

    is_drop = _compute_is_drop_from_foil(foil, drop_threshold=float(cfg.drop_threshold))

    # Metrics
    prf = pointwise_prf(is_drop.astype(int), alert.astype(int))
    ev = event_level_metrics(time_s=t, is_drop=is_drop, alert=alert, early_window_s=float(args.early_window_s))
    f1_event = _event_f1(ev.alert_event_precision, ev.drop_event_recall)

    # Persist anomalies timeline
    win_n = int(round(float(cfg.window_s) * hz))
    out_df = pd.DataFrame(
        {
            "time_s": t,
            "is_drop": is_drop.astype(int),
            "score_chaos": np.asarray(score_chaos, dtype=float),
            "score_fused": np.asarray(fused, dtype=float),
            "threshold": float(thr),
            "alert": alert.astype(int),
        }
    )
    for k, v in components.items():
        if k == "chaos":
            continue
        out_df[f"score_{k}"] = np.asarray(v, dtype=float)

    out_df.to_csv(outp / "anomalies_hybrid.csv", index=False, float_format="%.6f")

    metrics_out: Dict[str, Any] = {
        "picked_mode": str(pick_mode),
        "picked_run_id": int(picked.run_id),
        "window_s": float(cfg.window_s),
        "window_n": int(win_n),
        "drop_threshold": float(cfg.drop_threshold),
        "emb_dim": int(cfg.emb_dim),
        "emb_lag": int(cfg.emb_lag),
        "enable_mp": bool(args.enable_mp),
        "enable_causal": bool(args.enable_causal),
        "model_dir": str(model_dir) if model_dir is not None else None,
        "baseline_n": int(baseline_n),
        "baseline_s": float(args.baseline_s),
        "baseline_frac": float(args.baseline_frac),
        "baseline_percentile": float(args.baseline_percentile),
        "threshold": float(thr),
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "gate_dl": float(args.gate_dl),
        "gate_causal": float(args.gate_causal),
        "gate_chaos": float(args.gate_chaos),
        "merge_gap_s": float(args.merge_gap_s),
        "min_duration_s": float(args.min_duration_s),
        "alert_events": int(alert_events),
        "alert_frac": float(np.mean(alert.astype(float))),
        "weights": weights,
        # Pointwise metrics
        **{k: float(v) for k, v in prf.items()},
        # Event-level metrics
        "drop_events": int(ev.drop_events),
        "matched_drop_events": int(ev.matched_drop_events),
        "drop_event_recall": float(ev.drop_event_recall),
        "alert_event_precision": float(ev.alert_event_precision),
        "f1_event": float(f1_event),
        "lead_s_median": float(ev.lead_s_median),
        "lead_s_max": float(ev.lead_s_max),
        # Component activity
        "mp_active": bool(score_mp is not None),
        "mp_error": mp_error,
        "causal_active": bool(score_causal is not None),
        "causal_error": causal_error,
        "dl_active": bool(score_dl is not None),
        "dl_error": dl_error,
    }
    (outp / "metrics_hybrid.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8")

    # Explain (per alert timepoint)
    with (outp / "explain_hybrid.jsonl").open("w", encoding="utf-8") as f:
        for i in np.flatnonzero(alert):
            row = {
                "time_s": float(t[int(i)]),
                "score_fused": float(fused[int(i)]),
                "threshold": float(thr),
                "weights": weights,
                "components": {k: float(np.asarray(v, dtype=float)[int(i)]) for k, v in components.items()},
            }
            f.write(json.dumps(row) + "\n")

    # Debug grid
    (outp / "grid_metrics.json").write_text(json.dumps(grid_rows, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
