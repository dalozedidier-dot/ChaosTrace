from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, TypedDict

State = Literal["stable", "transition", "unstable"]


class WindowReport(TypedDict, total=False):
    ts: float
    score: float
    state: State


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """Configuration for live scoring.

    window_s: rolling time window size in seconds.
    stable_threshold / unstable_threshold: score thresholds for state labeling.
    max_gap_s: if consecutive samples exceed this gap (in seconds), they count as a data hole.
    """

    window_s: int = 180
    stable_threshold: float = 0.15
    unstable_threshold: float = 0.30
    max_gap_s: float = 300.0


class RollingWindow:
    """Rolling time-window over (timestamp, value) pairs."""

    def __init__(self, window_s: int = 180) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.window_s = window_s
        self._buf: list[tuple[datetime, float]] = []

    def push(self, ts: datetime, value: float) -> None:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        self._buf.append((ts, value))

    def _trim(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        cutoff = now - timedelta(seconds=self.window_s)

        idx: int | None = None
        for i, (ts, _v) in enumerate(self._buf):
            if ts >= cutoff:
                idx = i
                break

        if idx is None:
            self._buf.clear()
            return

        if idx > 0:
            del self._buf[:idx]

    def values(self, now: datetime) -> list[float]:
        self._trim(now)
        return [v for _ts, v in self._buf]

    def samples(self, now: datetime) -> list[tuple[datetime, float]]:
        """Return trimmed (timestamp, value) samples."""
        self._trim(now)
        return list(self._buf)


def analyze_window(values: list[float], cfg: LiveConfig | None = None) -> WindowReport:
    """Compute a lightweight stability score + state for the last window."""
    if cfg is None:
        cfg = LiveConfig()

    if not values:
        return {"score": 0.0, "state": "stable"}

    w = values[-max(1, cfg.window_s) :]
    score = min(1.0, abs(statistics.pstdev(w)))

    state: State
    if score < cfg.stable_threshold:
        state = "stable"
    elif score > cfg.unstable_threshold:
        state = "unstable"
    else:
        state = "transition"

    return {"ts": time.time(), "score": score, "state": state}
