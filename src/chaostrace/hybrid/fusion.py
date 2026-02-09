from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FusionOutput:
    threshold: float
    alert_mask: np.ndarray
    alert_events: int
    score_fused: np.ndarray
    components: dict[str, np.ndarray]
    weights: dict[str, float]


def _renorm_weights(w: dict[str, float]) -> dict[str, float]:
    s = float(sum(max(0.0, float(v)) for v in w.values()))
    if s <= 1e-12:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}


def default_weights(*, has_dl: bool, has_mp: bool, has_causal: bool) -> dict[str, float]:
    """Default fusion weights (renormalized)."""
    w = {
        "chaos": 0.65,
        "dl": 0.20 if has_dl else 0.0,
        "mp": 0.10 if has_mp else 0.0,
        "causal": 0.05 if has_causal else 0.0,
    }
    return _renorm_weights(w)


def fuse_scores(
    *,
    time_s: np.ndarray,
    score_chaos: np.ndarray,
    is_drop: np.ndarray,
    foil_height: np.ndarray,
    drop_threshold: float,
    score_mp: Optional[np.ndarray] = None,
    score_causal: Optional[np.ndarray] = None,
    score_dl: Optional[np.ndarray] = None,
    weights: Optional[dict[str, float]] = None,
    baseline_margin: float = 0.05,
    baseline_percentile: float = 99.5,
    threshold_min: float = 0.55,
    threshold_max: float = 0.97,
    merge_gap_s: float = 0.20,
    min_duration_s: float = 0.30,
) -> FusionOutput:
    """Fuse multiple score streams into a final alert mask with dynamic thresholding."""
    from chaostrace.orchestrator.sweep import dynamic_threshold, postprocess_alerts

    t = np.asarray(time_s, dtype=float)
    chaos = np.asarray(score_chaos, dtype=float)
    d = np.asarray(is_drop, dtype=float)
    foil = np.asarray(foil_height, dtype=float)

    if chaos.shape != t.shape:
        raise ValueError("score_chaos must match time_s shape")
    if d.shape != t.shape or foil.shape != t.shape:
        raise ValueError("is_drop and foil_height must have same shape as time_s")

    comps: dict[str, np.ndarray] = {"chaos": chaos}
    has_mp = score_mp is not None
    has_causal = score_causal is not None
    has_dl = score_dl is not None

    if score_mp is not None:
        s = np.asarray(score_mp, dtype=float)
        if s.shape != t.shape:
            raise ValueError("score_mp must match time_s shape")
        comps["mp"] = s
    if score_causal is not None:
        s = np.asarray(score_causal, dtype=float)
        if s.shape != t.shape:
            raise ValueError("score_causal must match time_s shape")
        comps["causal"] = s
    if score_dl is not None:
        s = np.asarray(score_dl, dtype=float)
        if s.shape != t.shape:
            raise ValueError("score_dl must match time_s shape")
        comps["dl"] = s

    if weights is None:
        w = default_weights(has_dl=has_dl, has_mp=has_mp, has_causal=has_causal)
    else:
        w = _renorm_weights(weights)

    fused = np.zeros_like(chaos, dtype=float)
    for k, s in comps.items():
        fused += float(w.get(k, 0.0)) * np.asarray(s, dtype=float)
    fused = np.clip(fused, 0.0, 1.0)

    baseline_mask = (d < 0.5) & (foil > (float(drop_threshold) + float(baseline_margin)))
    thr = dynamic_threshold(
        fused,
        baseline_mask,
        percentile=float(baseline_percentile),
        min_thr=float(threshold_min),
        max_thr=float(threshold_max),
    )
    alert_raw = fused > float(thr)
    alert_pp, alert_events = postprocess_alerts(
        alert_raw,
        t,
        merge_gap_s=float(merge_gap_s),
        min_duration_s=float(min_duration_s),
    )

    return FusionOutput(
        threshold=float(thr),
        alert_mask=alert_pp,
        alert_events=int(alert_events),
        score_fused=fused,
        components=comps,
        weights=w,
    )
