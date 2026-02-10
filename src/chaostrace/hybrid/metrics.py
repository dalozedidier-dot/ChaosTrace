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
    t = np.asarray(time_s, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if t.shape != m.shape:
        raise ValueError("time_s and mask must have same shape")

    events: List[Tuple[float, float]] = []
    in_ev = False
    start = 0
    for i, v in enumerate(m):
        if v and not in_ev:
            in_ev = True
            start = i
        if in_ev and (not v or i == len(m) - 1):
            end_idx = i if not v else i
            events.append((float(t[start]), float(t[end_idx])))
            in_ev = False
    return events


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a1 >= b0) and (a0 <= b1)


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
    """Event-level metrics with overlap-aware matching and early-warning leads.

    Definitions
    - drop event: contiguous segment where is_drop is True
    - alert event: contiguous segment where alert is True

    Matching
    A drop event is considered detected if any alert event overlaps it.
    Separately, a drop event has an early warning lead if there exists an alert event
    whose start time ta0 is within [td0 - early_window_s, td0].

    This avoids reporting 0 event recall when alerts start slightly after td0 but still
    overlap the drop.
    """
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(alert, dtype=bool)
    d = np.asarray(is_drop, dtype=bool)

    drop_events = _events_from_mask(t, d)
    alert_events = _events_from_mask(t, a)

    # Drop detection by overlap
    matched_drops = 0
    for (td0, td1) in drop_events:
        if any(_overlaps(ta0, ta1, td0, td1) for (ta0, ta1) in alert_events):
            matched_drops += 1

    # Early-warning leads (only when alert starts before drop start)
    leads: List[float] = []
    for (td0, _td1) in drop_events:
        best_lead = None
        for (ta0, _ta1) in alert_events:
            lead = float(td0 - ta0)
            if 0.0 <= lead <= float(early_window_s):
                if best_lead is None or lead > best_lead:
                    best_lead = lead
        if best_lead is not None:
            leads.append(float(best_lead))

    # Alert-event precision: good if overlaps any drop, or if it is an early warning for a drop.
    good_alerts = 0
    for (ta0, ta1) in alert_events:
        ok = False
        for (td0, td1) in drop_events:
            if _overlaps(ta0, ta1, td0, td1):
                ok = True
                break
            lead = float(td0 - ta0)
            if 0.0 <= lead <= float(early_window_s):
                ok = True
                break
        if ok:
            good_alerts += 1

    drop_event_recall = matched_drops / len(drop_events) if drop_events else 0.0
    alert_event_precision = good_alerts / len(alert_events) if alert_events else 0.0

    lead_s_median = float(np.median(leads)) if leads else 0.0
    lead_s_max = float(np.max(leads)) if leads else 0.0

    return EventMetrics(
        drop_events=len(drop_events),
        alert_events=len(alert_events),
        matched_drop_events=int(matched_drops),
        drop_event_recall=float(drop_event_recall),
        alert_event_precision=float(alert_event_precision),
        lead_s_median=float(lead_s_median),
        lead_s_max=float(lead_s_max),
    )
