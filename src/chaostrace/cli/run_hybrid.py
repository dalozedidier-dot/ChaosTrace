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


def _dynamic_threshold(
    scores: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    percentile: float,
    thr_min: float,
    thr_max: float,
) -> float:
    s = np.asarray(scores, dtype=float)
    m = np.asarray(baseline_mask, dtype=bool)
    if s.shape != m.shape:
        raise ValueError("scores and baseline_mask must have same shape")

    base = s[m]
    base = base[np.isfinite(base)]
    if base.size == 0:
        thr = float(np.nanpercentile(s[np.isfinite(s)], percentile)) if np.isfinite(s).any() else thr_min
    else:
        thr = float(np.percentile(base, percentile))

    if not np.isfinite(thr):
        thr = float(thr_min)
    return float(np.clip(thr, thr_min, thr_max))


def _postprocess_alerts(
    alert: np.ndarray,
    time_s: np.ndarray,
    *,
    merge_gap_s: float,
    min_duration_s: float,
) -> Tuple[np.ndarray, int]:
    """Merge short gaps and drop too short alert events."""
    a = np.asarray(alert, dtype=bool).copy()
    t = np.asarray(time_s, dtype=float)
    if a.shape != t.shape:
        raise ValueError("alert and time_s must have same shape")

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

    merged: List[Tuple[int, int]] = [events[0]]
    for s, e in events[1:]:
        ps, pe = merged[-1]
        gap = float(t[s] - t[pe])
        if gap <= float(merge_gap_s):
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    out = np.zeros_like(a, dtype=bool)
    kept = 0
    for s, e in merged:
        dur = float(t[e] - t[s])
        if dur >= float(min_duration_s):
            out[s : e + 1] = True
            kept += 1
    return out, kept


def _exc_brief(e: BaseException) -> str:
    msg = str(e).strip()
    if not msg:
        msg = e.__class__.__name__
    return f"{e.__class__.__name__}: {msg}"


def _compute_mp_score(df: pd.DataFrame, *, col: str, window_n: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.matrix_profile import compute_matrix_profile

        return compute_matrix_profile(df, col=col, window_n=window_n).score, None
    except Exception as e:
        return None, _exc_brief(e)


def _compute_causal_score(
    df: pd.DataFrame,
    *,
    cols: List[str],
    window_n: int,
    baseline_n: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.causal_var import compute_causal_drift

        return compute_causal_drift(df, cols=cols, window_n=window_n, baseline_n=baseline_n).score, None
    except Exception as e:
        return None, _exc_brief(e)


def _compute_dl_score(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    device: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from chaostrace.hybrid.dl.infer import infer_series

        out = infer_series(df, model_dir=model_dir, device=device)
        return out.score, None
    except Exception as e:
        return None, _exc_brief(e)


@dataclass(frozen=True)
class PickedRun:
    run_id: int
    cfg: SweepConfig
    window_n: int
    score_chaos: np.ndarray
    is_drop: np.ndarray
    metrics_pick: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run hybrid detection: chaos + optional MP + optional causal + optional DL."
        )
    )
    p.add_argument("--input", required=True, help="CSV input with time_s and telemetry columns.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--runs", type=int, default=30, help="Number of sweep configs sampled from internal grid.")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--drop-threshold", type=float, default=None)
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--emb-lag", type=int, default=None)

    p.add_argument("--enable-mp", action="store_true", help="Enable Matrix Profile component (requires stumpy).")
    p.add_argument("--require-mp", action="store_true", help="Fail if MP was requested but unavailable.")
    p.add_argument("--mp-col", default="boat_speed", help="Column to use for Matrix Profile score.")

    p.add_argument("--enable-causal", action="store_true", help="Enable causal drift component.")
    p.add_argument("--require-causal", action="store_true", help="Fail if causal was requested but unavailable.")
    p.add_argument("--causal-cols", default="boat_speed,foil_height_m", help="Columns used for causal drift (CSV).")

    p.add_argument("--model", default=None, help="Model directory produced by train_hybrid (optional).")
    p.add_argument("--require-dl", action="store_true", help="Fail if --model is provided but DL is unavailable.")
    p.add_argument("--device", default="cpu", help="DL device (cpu).")

    p.add_argument("--early-window-s", type=float, default=2.0, help="Event-level early window in seconds.")

    p.add_argument("--baseline-s", type=float, default=10.0, help="Seconds used as baseline segment from start.")
    p.add_argument("--baseline-frac", type=float, default=0.0, help="Fraction of series used as baseline (0 disables).")
    p.add_argument("--baseline-percentile", type=float, default=99.5)
    p.add_argument("--threshold-min", type=float, default=0.55)
    p.add_argument("--threshold-max", type=float, default=0.97)
    p.add_argument("--merge-gap-s", type=float, default=0.20)
    p.add_argument("--min-duration-s", type=float, default=0.30)

    p.add_argument(
        "--pick",
        default="min_alert_frac",
        choices=["min_alert_frac", "max_f1", "min_fp"],
        help="How to pick the best config when multiple runs are evaluated.",
    )
    return p


def _select_cfgs(args: argparse.Namespace) -> List[SweepConfig]:
    if all(getattr(args, k) is not None for k in ("window_s", "drop_threshold", "emb_dim", "emb_lag")):
        return [
            SweepConfig(
                window_s=float(args.window_s),
                drop_threshold=float(args.drop_threshold),
                emb_dim=int(args.emb_dim),
                emb_lag=int(args.emb_lag),
            )
        ]

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


def _pick_best(grid_rows: List[Dict[str, Any]], mode: str) -> int:
    if not grid_rows:
        return 0

    def key(i: int) -> Tuple[float, float, float]:
        r = grid_rows[i]
        f1 = float(r.get("f1", 0.0))
        fp = float(r.get("fp", 1e18))
        alert_frac = float(r.get("alert_frac", 1.0))
        if mode == "max_f1":
            return (-f1, fp, alert_frac)
        if mode == "min_fp":
            return (fp, -f1, alert_frac)
        return (alert_frac, fp, -f1)

    return int(min(range(len(grid_rows)), key=key))


def _fuse_components(
    score_chaos: np.ndarray,
    *,
    score_mp: Optional[np.ndarray],
    score_causal: Optional[np.ndarray],
    score_dl: Optional[np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fused = np.clip(float(weights["chaos"]) * np.asarray(score_chaos, dtype=float), 0.0, 1.0)
    comps: Dict[str, np.ndarray] = {"chaos": np.asarray(score_chaos, dtype=float)}

    if score_mp is not None and float(weights.get("mp", 0.0)) > 0.0:
        comps["mp"] = np.asarray(score_mp, dtype=float)
        fused += float(weights["mp"]) * comps["mp"]
    if score_causal is not None and float(weights.get("causal", 0.0)) > 0.0:
        comps["causal"] = np.asarray(score_causal, dtype=float)
        fused += float(weights["causal"]) * comps["causal"]
    if score_dl is not None and float(weights.get("dl", 0.0)) > 0.0:
        comps["dl"] = np.asarray(score_dl, dtype=float)
        fused += float(weights["dl"]) * comps["dl"]

    return np.clip(fused, 0.0, 1.0), comps


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    df = _load_csv(args.input)
    t = df["time_s"].to_numpy(dtype=float)
    hz = float(estimate_sample_hz(t))

    foil = _require_col(df, "foil_height_m")
    cfgs = _select_cfgs(args)

    metrics_df, tl_df = sweep(df, cfgs, seed=int(args.seed))
    if tl_df.empty:
        raise RuntimeError("Sweep produced empty timeline output")

    baseline_n = int(round(float(args.baseline_s) * hz))
    if float(args.baseline_frac) > 0.0:
        baseline_n = max(baseline_n, int(round(float(args.baseline_frac) * len(df))))
    baseline_n = int(np.clip(baseline_n, 1, len(df)))

    causal_cols = [c.strip() for c in str(args.causal_cols).split(",") if c.strip()]
    mp_col = str(args.mp_col).strip()

    model_dir = Path(str(args.model)) if args.model else None
    if model_dir is not None and not model_dir.exists():
        raise FileNotFoundError(f"--model directory not found: {model_dir}")

    grid_rows: List[Dict[str, Any]] = []
    per_run_cache: List[PickedRun] = []

    for rid in sorted(tl_df["run_id"].unique()):
        rid_int = int(rid)
        tl = tl_df[tl_df["run_id"] == rid_int].reset_index(drop=True)

        cfg = cfgs[rid_int - 1]
        win_n = max(int(round(float(cfg.window_s) * hz)), 5)

        score_chaos = tl["score_mean"].to_numpy(dtype=float)
        is_drop = foil < float(cfg.drop_threshold)

        score_mp, mp_err = (None, None)
        if bool(args.enable_mp):
            score_mp, mp_err = _compute_mp_score(df, col=mp_col, window_n=win_n)
            if score_mp is None and bool(args.require_mp):
                raise RuntimeError(f"MP requested but unavailable. {mp_err or ''}")

        score_causal, causal_err = (None, None)
        if bool(args.enable_causal):
            score_causal, causal_err = _compute_causal_score(
                df, cols=causal_cols, window_n=max(win_n, 8), baseline_n=baseline_n
            )
            if score_causal is None and bool(args.require_causal):
                raise RuntimeError(f"Causal requested but unavailable. {causal_err or ''}")

        score_dl, dl_err = (None, None)
        if model_dir is not None:
            score_dl, dl_err = _compute_dl_score(
                df.assign(is_drop=is_drop.astype(float)), model_dir=model_dir, device=str(args.device)
            )
            if score_dl is None and bool(args.require_dl):
                raise RuntimeError(f"DL requested but unavailable. {dl_err or ''}")

        weights = _default_weights(
            has_dl=score_dl is not None,
            has_mp=score_mp is not None,
            has_causal=score_causal is not None,
        )
        fused, comps = _fuse_components(
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
        alert, alert_events = _postprocess_alerts(
            alert_raw, t, merge_gap_s=float(args.merge_gap_s), min_duration_s=float(args.min_duration_s)
        )

        prf = pointwise_prf(alert, is_drop)
        ev = event_level_metrics(t, alert, is_drop, early_window_s=float(args.early_window_s))

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
            "lead_s_median": float(ev.lead_s_median),
            "lead_s_max": float(ev.lead_s_max),
            "mp_active": bool(score_mp is not None),
            "mp_error": str(mp_err) if score_mp is None and mp_err else "",
            "causal_active": bool(score_causal is not None),
            "causal_error": str(causal_err) if score_causal is None and causal_err else "",
            "dl_active": bool(score_dl is not None),
            "dl_error": str(dl_err) if score_dl is None and dl_err else "",
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
                metrics_pick=row,
            )
        )

    best_i = _pick_best(grid_rows, str(args.pick))
    picked = per_run_cache[int(best_i)]
    cfg = picked.cfg
    win_n = picked.window_n
    score_chaos = picked.score_chaos
    is_drop = picked.is_drop

    score_mp, mp_err = (None, None)
    if bool(args.enable_mp):
        score_mp, mp_err = _compute_mp_score(df, col=mp_col, window_n=win_n)
        if score_mp is None and bool(args.require_mp):
            raise RuntimeError(f"MP requested but unavailable. {mp_err or ''}")

    score_causal, causal_err = (None, None)
    if bool(args.enable_causal):
        score_causal, causal_err = _compute_causal_score(
            df,
            cols=causal_cols,
            window_n=max(win_n, 8),
            baseline_n=baseline_n,
        )
        if score_causal is None and bool(args.require_causal):
            raise RuntimeError(f"Causal requested but unavailable. {causal_err or ''}")

    score_dl, dl_err = (None, None)
    if model_dir is not None:
        score_dl, dl_err = _compute_dl_score(
            df.assign(is_drop=is_drop.astype(float)),
            model_dir=model_dir,
            device=str(args.device),
        )
        if score_dl is None and bool(args.require_dl):
            raise RuntimeError(f"DL requested but unavailable. {dl_err or ''}")

    weights = _default_weights(
        has_dl=score_dl is not None,
        has_mp=score_mp is not None,
        has_causal=score_causal is not None,
    )
    fused, comps = _fuse_components(
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
    alert, alert_events = _postprocess_alerts(
        alert_raw, t, merge_gap_s=float(args.merge_gap_s), min_duration_s=float(args.min_duration_s)
    )

    prf = pointwise_prf(alert, is_drop)
    ev = event_level_metrics(t, alert, is_drop, early_window_s=float(args.early_window_s))

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

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
    for k, v in comps.items():
        if k == "chaos":
            continue
        out_df[f"score_{k}"] = np.asarray(v, dtype=float)

    out_df.to_csv(outp / "anomalies_hybrid.csv", index=False, float_format="%.6f")

    metrics_out: Dict[str, Any] = {
        "picked_mode": str(args.pick),
        "picked_run_id": int(picked.run_id),
        "window_s": float(cfg.window_s),
        "window_n": int(win_n),
        "drop_threshold": float(cfg.drop_threshold),
        "emb_dim": int(cfg.emb_dim),
        "emb_lag": int(cfg.emb_lag),
        "enable_mp": bool(args.enable_mp),
        "enable_causal": bool(args.enable_causal),
        "model_dir": str(model_dir) if model_dir is not None else "",
        "mp_active": bool(score_mp is not None),
        "mp_error": str(mp_err) if score_mp is None and mp_err else "",
        "causal_active": bool(score_causal is not None),
        "causal_error": str(causal_err) if score_causal is None and causal_err else "",
        "dl_active": bool(score_dl is not None),
        "dl_error": str(dl_err) if score_dl is None and dl_err else "",
        "baseline_n": int(baseline_n),
        "baseline_s": float(args.baseline_s),
        "baseline_frac": float(args.baseline_frac),
        "baseline_percentile": float(args.baseline_percentile),
        "threshold": float(thr),
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "merge_gap_s": float(args.merge_gap_s),
        "min_duration_s": float(args.min_duration_s),
        "alert_events": int(alert_events),
        "alert_frac": float(np.mean(alert.astype(float))),
        "weights": weights,
        **{k: float(v) for k, v in prf.items()},
        "drop_events": int(ev.drop_events),
        "matched_drop_events": int(ev.matched_drop_events),
        "drop_event_recall": float(ev.drop_event_recall),
        "alert_event_precision": float(ev.alert_event_precision),
        "lead_s_median": float(ev.lead_s_median),
        "lead_s_max": float(ev.lead_s_max),
    }
    (outp / "metrics_hybrid.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8")

    with (outp / "explain_hybrid.jsonl").open("w", encoding="utf-8") as f:
        for i in np.flatnonzero(alert):
            row = {
                "time_s": float(t[int(i)]),
                "score_fused": float(fused[int(i)]),
                "threshold": float(thr),
                "weights": weights,
                "components": {k: float(np.asarray(v, dtype=float)[int(i)]) for k, v in comps.items()},
            }
            f.write(json.dumps(row) + "\n")

    (outp / "grid_metrics.json").write_text(json.dumps(grid_rows, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
