from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def pointwise_prf(alert: np.ndarray, is_drop: np.ndarray) -> Dict[str, float]:
    a = np.asarray(alert, dtype=bool)
    d = np.asarray(is_drop, dtype=bool)
    if a.shape != d.shape:
        raise ValueError("alert and is_drop must have same shape")
    tp = int(np.sum(a & d))
    fp = int(np.sum(a & ~d))
    fn = int(np.sum(~a & d))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _events_from_mask(time_s: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Return contiguous (start_time, end_time) segments where mask is True."""
    t = np.asarray(time_s, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if t.shape != m.shape:
        raise ValueError("time_s and mask must have same shape")

    events: List[Tuple[float, float]] = []
    if len(m) == 0:
        return events

    in_ev = False
    start_idx = 0
    for i, v in enumerate(m):
        if v and not in_ev:
            in_ev = True
            start_idx = int(i)
            continue
        if in_ev and not v:
            end_idx = int(i - 1)
            events.append((float(t[start_idx]), float(t[end_idx])))
            in_ev = False

    if in_ev:
        events.append((float(t[start_idx]), float(t[-1])))

    return events


@dataclass(frozen=True)
class EventMetrics:
    drop_events: int
    alert_events: int
    matched_drop_events: int
    drop_event_recall: float
    alert_event_precision: float
    lead_s_median: float
    lead_s_max: float


def event_level_metrics(
    time_s: np.ndarray,
    alert: np.ndarray,
    is_drop: np.ndarray,
    *,
    early_window_s: float = 2.0,
) -> EventMetrics:
    """Compute event-level metrics and early-warning lead times.

    A drop event is a contiguous segment where is_drop is True.
    An alert event is a contiguous segment where alert is True.

    A drop event is considered "matched" if there exists an alert event
    whose START time is within [t_drop_start - early_window_s, t_drop_start].

    Precision is computed over alert events: an alert event is "good" if it matches
    at least one drop event under the same rule.
    """
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(alert, dtype=bool)
    d = np.asarray(is_drop, dtype=bool)

    drop_events = _events_from_mask(t, d)
    alert_events = _events_from_mask(t, a)

    # Match drops -> alerts (greedy: pick earliest alert in window)
    matched_drops = 0
    leads: List[float] = []

    used_alert = [False] * len(alert_events)
    for (td0, _td1) in drop_events:
        best_j = None
        best_lead = None
        for j, (ta0, _ta1) in enumerate(alert_events):
            if used_alert[j]:
                continue
            lead = td0 - ta0
            if lead < 0:
                continue
            if lead <= float(early_window_s):
                if best_lead is None or lead > best_lead:
                    best_lead = float(lead)
                    best_j = j
        if best_j is not None:
            used_alert[best_j] = True
            matched_drops += 1
            leads.append(float(best_lead))

    # Alert-event precision: any alert that has a drop within window ahead?
    good_alerts = 0
    for (ta0, _ta1) in alert_events:
        ok = False
        for (td0, _td1) in drop_events:
            lead = td0 - ta0
            if 0.0 <= lead <= float(early_window_s):
                ok = True
                break
        good_alerts += 1 if ok else 0

    drop_event_recall = matched_drops / len(drop_events) if drop_events else 0.0
    alert_event_precision = good_alerts / len(alert_events) if alert_events else 0.0

    if leads:
        lead_med = float(np.median(np.asarray(leads, dtype=float)))
        lead_max = float(np.max(np.asarray(leads, dtype=float)))
    else:
        lead_med = 0.0
        lead_max = 0.0

    return EventMetrics(
        drop_events=int(len(drop_events)),
        alert_events=int(len(alert_events)),
        matched_drop_events=int(matched_drops),
        drop_event_recall=float(drop_event_recall),
        alert_event_precision=float(alert_event_precision),
        lead_s_median=float(lead_med),
        lead_s_max=float(lead_max),
    )
