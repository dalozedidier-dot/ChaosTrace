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
        elif in_ev and not v:
            events.append((s_idx, i - 1))
            in_ev = False
    if in_ev:
        events.append((s_idx, len(a) - 1))

    if not events:
        return a, 0

    # Merge gaps shorter than merge_gap_s
    merged: List[Tuple[int, int]] = [events[0]]
    for s, e in events[1:]:
        ps, pe = merged[-1]
        gap = float(t[s] - t[pe])
        if gap <= float(merge_gap_s):
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Filter short events
    out = np.zeros_like(a, dtype=bool)
    kept = 0
    for s, e in merged:
        dur = float(t[e] - t[s])
        if dur >= float(min_duration_s):
            out[s : e + 1] = True
            kept += 1
    return out, kept


def _compute_mp_score(df: pd.DataFrame, *, col: str, window_n: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.matrix_profile import compute_matrix_profile

        return compute_matrix_profile(df, col=col, window_n=window_n).score, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _compute_causal_score(
    df: pd.DataFrame, *, cols: List[str], window_n: int, baseline_n: int
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.causal_var import compute_causal_drift

        return compute_causal_drift(df, cols=cols, window_n=window_n, baseline_n=baseline_n).score, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _compute_dl_score(df: pd.DataFrame, *, model_dir: Path, device: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.dl.infer import infer_series

        out = infer_series(df, model_dir=model_dir, device=device)
        return out.score, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@dataclass(frozen=True)
class PickedRun:
    run_id: int
    cfg: SweepConfig
    window_n: int
    score_chaos: np.ndarray
    is_drop: np.ndarray
    foil_height: np.ndarray
    metrics_pick: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run hybrid detection: chaos + optional Matrix Profile + optional causal drift + optional DL."
    )
    p.add_argument("--input", required=True, help="CSV input with time_s and telemetry columns.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--runs", type=int, default=30, help="Number of sweep configs sampled from internal grid.")
    p.add_argument("--seed", type=int, default=7)

    # Optional fixed config override
    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--drop-threshold", type=float, default=None)
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--emb-lag", type=int, default=None)

    # Optional hybrid components
    p.add_argument("--enable-mp", action="store_true", help="Enable Matrix Profile component (requires stumpy).")
    p.add_argument("--mp-col", default="boat_speed", help="Column to use for Matrix Profile score.")
    p.add_argument("--require-mp", action="store_true", help="Fail if MP is requested but not available.")

    p.add_argument("--enable-causal", action="store_true", help="Enable causal drift component.")
    p.add_argument("--causal-cols", default="boat_speed,foil_height_m", help="Columns used for causal drift (CSV).")
    p.add_argument("--require-causal", action="store_true", help="Fail if causal drift is requested but not available.")

    p.add_argument("--model", default=None, help="Model directory produced by train_hybrid (optional).")
    p.add_argument("--device", default="cpu", help="DL device (cpu).")
    p.add_argument("--require-dl", action="store_true", help="Fail if a model is provided but DL inference fails.")

    # Event-level early warning window
    p.add_argument("--early-window-s", type=float, default=2.0, help="Event-level early window in seconds.")

    # Dynamic threshold and post-processing knobs
    p.add_argument("--baseline-s", type=float, default=10.0, help="Seconds used as baseline segment from start.")
    p.add_argument("--baseline-frac", type=float, default=0.0, help="Fraction of series used as baseline (0 disables).")
    p.add_argument("--baseline-percentile", type=float, default=99.5)
    p.add_argument("--threshold-min", type=float, default=0.55)
    p.add_argument("--threshold-max", type=float, default=0.97)
    p.add_argument("--gate-dl", type=float, default=0.0, help="Component gate for DL score (0 disables).")
    p.add_argument("--gate-causal", type=float, default=0.0, help="Component gate for causal score (0 disables).")
    p.add_argument("--gate-chaos", type=float, default=0.0, help="Component gate for chaos score (0 disables).")
    p.add_argument("--merge-gap-s", type=float, default=0.20)
    p.add_argument("--min-duration-s", type=float, default=0.30)


    # Optional composite incoherence score (per timepoint, baseline-excess aggregated)
    p.add_argument(
        "--incoherence",
        action="store_true",
        help=(
            "Add a composite incoherence score column (aggregated baseline-excess of component metrics). "
            "This does not change alerting unless you explicitly use the column downstream."
        ),
    )
    p.add_argument(
        "--inco-weights",
        default=None,
        help=(
            "Optional JSON dict of weights for incoherence components, e.g. "
            "'{\"markov\":0.45,\"delta\":0.15,\"lyap\":0.15,\"rqa_break\":0.25}'. "
            "Unknown keys are ignored. If omitted, uniform weights are used."
        ),
    )
    p.add_argument(
        "--inco-percentile",
        type=float,
        default=None,
        help="Percentile for per-component baseline thresholds (default: --baseline-percentile).",
    )
    p.add_argument(
        "--inco-only-chaos",
        action="store_true",
        help=(
            "If set, incoherence uses only chaos sub-metrics (delta, markov, lyap, rqa_break, laminar_break) "
            "and ignores optional MP/causal/DL components."
        ),
    )

    p.add_argument(
        "--pick",
        default="auto",
        choices=["auto", "min_alert_frac", "max_f1", "max_f1_event", "max_ew", "min_fp"],
        help=(
            "How to pick the best config when multiple runs are evaluated. "
            "Use max_ew for serious early-warning (lead + event F1)."
        ),
    )
    return p


def _select_cfgs(args: argparse.Namespace) -> List[SweepConfig]:
    # If user specified all core config fields, run a single config.
    if all(getattr(args, k) is not None for k in ("window_s", "drop_threshold", "emb_dim", "emb_lag")):
        return [
            SweepConfig(
                window_s=float(args.window_s),
                drop_threshold=float(args.drop_threshold),
                emb_dim=int(args.emb_dim),
                emb_lag=int(args.emb_lag),
            )
        ]

    # Default grid (small but meaningful)
    cfgs = build_grid(
        window_s=[3.0, 5.0, 10.0],
        drop_threshold=[0.3, 0.4],
        emb_dim=[3, 5],
        emb_lag=[8, 12],
    )

    rng = np.random.default_rng(int(args.seed))
    n = int(args.runs)
    if n <= 0 or n >= len(cfgs):
        return cfgs
    idx = rng.choice(np.arange(len(cfgs)), size=n, replace=False)
    return [cfgs[int(i)] for i in idx]


def _resolve_pick_mode(args: argparse.Namespace, grid_rows: List[Dict[str, Any]]) -> str:
    if str(args.pick) != "auto":
        return str(args.pick)

    has_drops = any(int(r.get("drop_events", 0)) > 0 for r in grid_rows)
    if not has_drops:
        return "min_alert_frac"

    # If a model is provided, default to early-warning mode.
    if args.model is not None:
        return "max_ew"

    return "max_f1_event"


def _pick_best(grid_rows: List[Dict[str, Any]], mode: str) -> int:
    if not grid_rows:
        return 0

    def key(i: int) -> Tuple[float, float, float, float]:
        r = grid_rows[i]

        # Core metrics
        f1 = float(r.get("f1", 0.0))
        fp = float(r.get("fp", 1e18))
        alert_frac = float(r.get("alert_frac", 1.0))

        # Event-level early warning metrics
        f1_event = float(r.get("f1_event", 0.0))
        lead_max = float(r.get("lead_s_max", 0.0))
        drop_events = int(r.get("drop_events", 0))
        alert_events = int(r.get("alert_events", 0))

        # Hard-penalize silent runs on drop datasets.
        silent = 1.0 if (drop_events > 0 and alert_events == 0) else 0.0

        if mode == "max_f1":
            return (silent, -f1, fp, alert_frac)

        if mode == "min_fp":
            return (silent, fp, -f1, alert_frac)

        if mode == "max_f1_event":
            # Maximize event F1, then prefer positive lead and fewer alert events.
            lead_bad = 1.0 if lead_max <= 0.0 else 0.0
            return (silent + lead_bad, -f1_event, -lead_max, alert_events, fp, alert_frac)

        if mode == "max_ew":
            # Early-warning selection must not sacrifice precision.
            # We prioritize event-level F1 first, then positive lead, then lead magnitude.
            lead_bad = 1.0 if lead_max <= 0.0 else 0.0
            # Tie-breakers: fewer alert events and fewer FP.
            return (silent + lead_bad, -f1_event, -lead_max, alert_events, fp, alert_frac)

        # min_alert_frac
        return (silent, alert_frac, fp, -f1)

    best = min(range(len(grid_rows)), key=key)
    return int(best)



def _robust_01(x: np.ndarray, *, p: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    with np.errstate(all="ignore"):
        scale = float(np.nanpercentile(x, float(p))) if np.isfinite(x).any() else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return np.clip(x / scale, 0.0, 1.0)


def _fill0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, 0.0)


def _compute_chaos_inco_components(df: pd.DataFrame, *, cfg: SweepConfig, window_n: int) -> Dict[str, np.ndarray]:
    """Compute per-timepoint sub-metrics used to explain chaos incoherence.

    Output components are scaled to [0, 1] and oriented so larger means 'more incoherent'.
    """
    # Local imports to keep CLI import light.
    from chaostrace.analyzers.delta_stats import delta_stats
    from chaostrace.analyzers.lyapunov_like import lyapunov_like
    from chaostrace.analyzers.markov_drop import markov_drop
    from chaostrace.analyzers.null_trace import null_trace
    from chaostrace.analyzers.rqa_light import rqa_light

    res_null = null_trace(df, col="foil_height_m", win=int(window_n), eps=0.01)
    res_delta = delta_stats(df, a="boat_speed", b="foil_height_m")
    res_markov = markov_drop(df, drop_threshold=float(cfg.drop_threshold))
    res_rqa = rqa_light(df, col="boat_speed", dim=int(cfg.emb_dim), lag=int(cfg.emb_lag), eps=0.5)
    res_lyap = lyapunov_like(df, col="boat_speed", dim=int(cfg.emb_dim), lag=int(cfg.emb_lag), max_t=20)

    laminar = np.clip(_fill0(res_null.timeline["score"].to_numpy(dtype=float)), 0.0, 1.0)
    delta01 = _robust_01(_fill0(res_delta.timeline["score"].to_numpy(dtype=float)))
    markov01 = _robust_01(_fill0(res_markov.timeline["score"].to_numpy(dtype=float)))

    rqa_det = res_rqa.metrics.get("det_proxy", float("nan"))
    rqa_rr = res_rqa.metrics.get("rr", float("nan"))
    rqa_inv = float(rqa_det) if np.isfinite(rqa_det) else float(rqa_rr)
    if not np.isfinite(rqa_inv):
        rqa_inv = 0.0
    rqa_inv = float(np.clip(rqa_inv, 0.0, 1.0))
    rqa_break = np.full(len(df), 1.0 - rqa_inv, dtype=float)

    lyap_val = res_lyap.metrics.get("lyap_like", float("nan"))
    lyap_abs = float(abs(lyap_val)) if np.isfinite(lyap_val) else 0.0
    lyap01 = float(1.0 - np.exp(-lyap_abs))
    lyap01 = float(np.clip(lyap01, 0.0, 1.0))
    lyap_series = np.full(len(df), lyap01, dtype=float)

    laminar_break = 1.0 - laminar

    return {
        "delta": delta01,
        "markov": markov01,
        "lyap": lyap_series,
        "rqa_break": rqa_break,
        "laminar_break": laminar_break,
    }


def _compute_incoherence_series(
    components: Dict[str, np.ndarray],
    baseline_mask: np.ndarray,
    *,
    weights: Optional[Dict[str, float]] = None,
    percentile: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Aggregate a composite incoherence series as baseline-excess.

    inco[t] = sum_k w_k * max(0, comp_k[t] - thr_k)
    where thr_k is a percentile computed on the baseline segment.
    """
    if not components:
        return np.zeros(0, dtype=float), {}, {}

    # Ensure consistent shape.
    n = len(next(iter(components.values())))
    for k, v in list(components.items()):
        vv = np.asarray(v, dtype=float)
        if vv.shape[0] != n:
            raise ValueError(f"Incoherence component {k!r} has wrong length: {vv.shape[0]} != {n}")
        components[k] = vv

    keys = list(components.keys())

    # Parse / normalize weights.
    if weights is None:
        w = {k: 1.0 / len(keys) for k in keys}
    else:
        w = {}
        for k in keys:
            try:
                w[k] = float(weights.get(k, 0.0))
            except Exception:
                w[k] = 0.0
        w = _renorm_weights(w)
        # If all provided weights are 0, fall back to uniform.
        if sum(w.values()) <= 1e-12:
            w = {k: 1.0 / len(keys) for k in keys}

    m = np.asarray(baseline_mask, dtype=bool)
    if m.shape[0] != n:
        raise ValueError("baseline_mask must have same length as components")

    thr: Dict[str, float] = {}
    for k in keys:
        s = components[k]
        base = s[m]
        base = base[np.isfinite(base)]
        ref = base if base.size else s[np.isfinite(s)]
        thr[k] = float(np.percentile(ref, float(percentile))) if ref.size else 0.0

    inco = np.zeros(n, dtype=float)
    for k in keys:
        inco += float(w.get(k, 0.0)) * np.maximum(0.0, components[k] - float(thr[k]))

    inco = np.where(np.isfinite(inco), inco, 0.0)
    return inco, thr, w



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
    per_run_cache: List[PickedRun] = []

    # First pass: evaluate each config and collect per-run summary rows.
    for rid in sorted(tl_df["run_id"].unique()):
        rid_int = int(rid)
        tl = tl_df[tl_df["run_id"] == rid_int].reset_index(drop=True)

        cfg = cfgs[rid_int - 1]
        win_n = max(int(round(float(cfg.window_s) * hz)), 5)

        score_chaos = tl["score_mean"].to_numpy(dtype=float)
        is_drop = foil < float(cfg.drop_threshold)

        score_mp, mp_error = (None, None)
        if bool(args.enable_mp):
            score_mp, mp_error = _compute_mp_score(df, col=mp_col, window_n=win_n)
            if bool(args.require_mp) and score_mp is None:
                raise RuntimeError(f"Matrix Profile requested but unavailable: {mp_error}")

        score_causal, causal_error = (None, None)
        if bool(args.enable_causal):
            score_causal, causal_error = _compute_causal_score(
                df, cols=causal_cols, window_n=max(win_n, 8), baseline_n=baseline_n
            )
            if bool(args.require_causal) and score_causal is None:
                raise RuntimeError(f"Causal drift requested but unavailable: {causal_error}")

        score_dl, dl_error = (None, None)
        if model_dir is not None:
            score_dl, dl_error = _compute_dl_score(df.assign(is_drop=is_drop.astype(float)), model_dir=model_dir, device=str(args.device))
            if bool(args.require_dl) and score_dl is None:
                raise RuntimeError(f"DL model provided but inference failed: {dl_error}")

        weights = _default_weights(has_dl=score_dl is not None, has_mp=score_mp is not None, has_causal=score_causal is not None)
        fused, components = _fuse_components(
            score_chaos,
            score_mp=score_mp,
            score_causal=score_causal,
            score_dl=score_dl,
            weights=weights,
        )

        baseline_mask = (np.arange(len(df)) < baseline_n) & (~is_drop)
    incoherence = None
    inco_thresholds: Dict[str, float] = {}
    inco_weights: Dict[str, float] = {}
    inco_percentile = float(args.inco_percentile) if args.inco_percentile is not None else float(args.baseline_percentile)

    if bool(args.incoherence):
        # Build a richer component set than score_mean alone (delta/markov/lyap/RQA/laminar),
        # then optionally add MP/causal/DL components when enabled.
        inco_components = _compute_chaos_inco_components(df, cfg=cfg, window_n=win_n)

        if not bool(args.inco_only_chaos):
            if score_mp is not None:
                inco_components["mp"] = np.asarray(score_mp, dtype=float)
            if score_causal is not None:
                inco_components["causal"] = np.asarray(score_causal, dtype=float)
            if score_dl is not None:
                inco_components["dl"] = np.asarray(score_dl, dtype=float)

        user_w: Optional[Dict[str, float]] = None
        if args.inco_weights is not None:
            try:
                w_raw = json.loads(str(args.inco_weights))
                if isinstance(w_raw, dict):
                    user_w = {str(k): float(v) for k, v in w_raw.items()}
            except Exception:
                user_w = None

        incoherence, inco_thresholds, inco_weights = _compute_incoherence_series(
            inco_components,
            baseline_mask,
            weights=user_w,
            percentile=inco_percentile,
        )

        thr = _dynamic_threshold(
            fused,
            baseline_mask,
            percentile=float(args.baseline_percentile),
            thr_min=float(args.threshold_min),
            thr_max=float(args.threshold_max),
        )
        alert_raw = fused > float(thr)
        # Optional component gating to reduce spurious alert islands:
        # keep an alert point only if at least one component is "strong enough".
        gate_any = (float(args.gate_chaos) > 0.0) or (float(args.gate_dl) > 0.0) or (float(args.gate_causal) > 0.0)
        if gate_any:
            support = np.asarray(score_chaos, dtype=float) >= float(args.gate_chaos)
            if score_dl is not None:
                support = support | (np.asarray(score_dl, dtype=float) >= float(args.gate_dl))
            if score_causal is not None:
                support = support | (np.asarray(score_causal, dtype=float) >= float(args.gate_causal))
            alert_raw = alert_raw & support
        alert, alert_events = _postprocess_alerts(
            alert_raw,
            t,
            merge_gap_s=float(args.merge_gap_s),
            min_duration_s=float(args.min_duration_s),
        )

        prf = pointwise_prf(alert, is_drop)
        ev = event_level_metrics(t, alert, is_drop, early_window_s=float(args.early_window_s))

        prec_event = float(ev.alert_event_precision)
        rec_event = float(ev.drop_event_recall)
        f1_event = _event_f1(prec_event, rec_event)

        row: Dict[str, Any] = {
            "run_id": rid_int,
            "window_s": float(cfg.window_s),
            "window_n": int(win_n),
            "drop_threshold": float(cfg.drop_threshold),
            "emb_dim": int(cfg.emb_dim),
            "emb_lag": int(cfg.emb_lag),
            "threshold": float(thr),
            "alert_events": int(alert_events),
            "alert_frac": float(np.mean(alert.astype(float))),
            **{k: float(v) for k, v in prf.items()},
            "drop_events": int(ev.drop_events),
            "matched_drop_events": int(ev.matched_drop_events),
            "drop_event_recall": float(ev.drop_event_recall),
            "alert_event_precision": float(ev.alert_event_precision),
            "f1_event": float(f1_event),
            "lead_s_median": float(ev.lead_s_median),
            "lead_s_max": float(ev.lead_s_max),
            "mp_active": bool(score_mp is not None),
            "mp_error": mp_error,
            "causal_active": bool(score_causal is not None),
            "causal_error": causal_error,
            "dl_active": bool(score_dl is not None),
            "dl_error": dl_error,
            "weights": weights,
        }

        grid_rows.append(row)
        per_run_cache.append(
            PickedRun(
                run_id=rid_int,
                cfg=cfg,
                window_n=win_n,
                score_chaos=score_chaos,
                is_drop=is_drop,
                foil_height=foil,
                metrics_pick=row,
            )
        )

    pick_mode = _resolve_pick_mode(args, grid_rows)
    best_i = _pick_best(grid_rows, pick_mode)
    picked = per_run_cache[int(best_i)]

    # Recompute for picked run (to write full outputs deterministically).
    cfg = picked.cfg
    win_n = picked.window_n
    score_chaos = picked.score_chaos
    is_drop = picked.is_drop

    score_mp, mp_error = (None, None)
    if bool(args.enable_mp):
        score_mp, mp_error = _compute_mp_score(df, col=mp_col, window_n=win_n)
        if bool(args.require_mp) and score_mp is None:
            raise RuntimeError(f"Matrix Profile requested but unavailable: {mp_error}")

    score_causal, causal_error = (None, None)
    if bool(args.enable_causal):
        score_causal, causal_error = _compute_causal_score(df, cols=causal_cols, window_n=max(win_n, 8), baseline_n=baseline_n)
        if bool(args.require_causal) and score_causal is None:
            raise RuntimeError(f"Causal drift requested but unavailable: {causal_error}")

    score_dl, dl_error = (None, None)
    if model_dir is not None:
        score_dl, dl_error = _compute_dl_score(df.assign(is_drop=is_drop.astype(float)), model_dir=model_dir, device=str(args.device))
        if bool(args.require_dl) and score_dl is None:
            raise RuntimeError(f"DL model provided but inference failed: {dl_error}")

    weights = _default_weights(has_dl=score_dl is not None, has_mp=score_mp is not None, has_causal=score_causal is not None)
    fused, components = _fuse_components(
        score_chaos,
        score_mp=score_mp,
        score_causal=score_causal,
        score_dl=score_dl,
        weights=weights,
    )

    baseline_mask = (np.arange(len(df)) < baseline_n) & (~is_drop)
    thr = _dynamic_threshold(
        fused,
        baseline_mask,
        percentile=float(args.baseline_percentile),
        thr_min=float(args.threshold_min),
        thr_max=float(args.threshold_max),
    )
    alert_raw = fused > float(thr)
    # Apply component gating (same as during grid evaluation) to avoid spurious alert islands.
    gate_any = (float(args.gate_chaos) > 0.0) or (float(args.gate_dl) > 0.0) or (float(args.gate_causal) > 0.0)
    if gate_any:
        support = np.asarray(score_chaos, dtype=float) >= float(args.gate_chaos)
        if score_dl is not None:
            support = support | (np.asarray(score_dl, dtype=float) >= float(args.gate_dl))
        if score_causal is not None:
            support = support | (np.asarray(score_causal, dtype=float) >= float(args.gate_causal))
        alert_raw = alert_raw & support
    alert, alert_events = _postprocess_alerts(
        alert_raw,
        t,
        merge_gap_s=float(args.merge_gap_s),
        min_duration_s=float(args.min_duration_s),
    )

    prf = pointwise_prf(alert, is_drop)
    ev = event_level_metrics(t, alert, is_drop, early_window_s=float(args.early_window_s))
    prec_event = float(ev.alert_event_precision)
    rec_event = float(ev.drop_event_recall)
    f1_event = _event_f1(prec_event, rec_event)

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    # Timeline CSV
    base_cols: Dict[str, Any] = {
        "time_s": t,
        "is_drop": is_drop.astype(int),
        "score_chaos": np.asarray(score_chaos, dtype=float),
        "score_fused": np.asarray(fused, dtype=float),
        "threshold": float(thr),
        "alert": alert.astype(int),
    }
    if incoherence is not None:
        base_cols["incoherence"] = np.asarray(incoherence, dtype=float)

    out_df = pd.DataFrame(base_cols)
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
        "incoherence_enabled": bool(args.incoherence),
        "inco_only_chaos": bool(args.inco_only_chaos),
        "inco_percentile": float(inco_percentile),
        "inco_weights": inco_weights,
        "inco_thresholds": inco_thresholds,

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
                "incoherence": float(incoherence[int(i)]) if incoherence is not None else None,
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
