from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from redis.typing import EncodableT

from .live import LiveConfig, RollingWindow, analyze_window
from .utils import make_redis

StreamFields = dict[str, str]
Message = tuple[str, StreamFields]
XReadResponse = list[tuple[str, list[Message]]]


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except (TypeError, ValueError):
        return datetime(1970, 1, 1, tzinfo=UTC)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_key(station_id: str, variable: str) -> str:
    return f"meteovoid:latest:{station_id}:{variable}"


def _default_start_id() -> str:
    """Default Redis Streams start-id.

    Order of precedence:
    1) METEOVOID_START_ID env var if set
    2) In CI, default to '0-0' so seeded messages are consumed
    3) Locally, default to '$' (tail)
    """
    override = os.getenv("METEOVOID_START_ID")
    if override:
        return override
    if os.getenv("GITHUB_ACTIONS") or os.getenv("CI"):
        return "0-0"
    return "$"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(round(0.95 * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return float(xs[k])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _meteo_interpretation(
    state: str,
    score: float,
    n_points: int,
    missing_time_frac: float,
    gap_count: int,
) -> dict[str, Any]:
    flags: list[str] = []
    phrases: list[str] = []

    if n_points < 10:
        flags.append("low_data")
        phrases.append("Fenêtre peu alimentée, interprétation prudente.")

    if gap_count > 0:
        flags.append("data_gaps")
        phrases.append("Trous de données détectés, possible perte de transmission ou capteur intermittent.")

    if missing_time_frac >= 0.10:
        flags.append("missing_significant")
        phrases.append("Part importante de la fenêtre sans données, fiabilité dégradée.")

    if state == "unstable":
        flags.append("gusts_erratic")
        phrases.append("Variabilité forte, rafales erratiques ou rupture de régime probable.")
    elif state == "transition":
        flags.append("watch")
        phrases.append("Variabilité en hausse, surveillance recommandée.")
    else:
        phrases.append("Signal globalement régulier sur la fenêtre.")

    # A tiny, explicit mapping so it stays predictable.
    severity = "low"
    if state == "unstable" or missing_time_frac >= 0.10:
        severity = "high"
    elif state == "transition" or gap_count > 0:
        severity = "medium"

    return {
        "interpretation": " ".join(phrases).strip(),
        "flags": flags,
        "severity": severity,
        "score_hint": "plus le score est haut, plus la variabilité est forte",
        "state_hint": "stable, transition, unstable",
    }


def process_observation(
    fields: Mapping[str, Any],
    msg_id: str = "",
    cfg: LiveConfig | None = None,
    windows: dict[tuple[str, str], RollingWindow] | None = None,
    ts_ingest: float | None = None,
) -> dict[str, Any] | None:
    """Process a single observation message and return an enriched report."""
    cfg = cfg or LiveConfig()
    windows = windows if windows is not None else {}

    station_id_raw = fields.get("station_id")
    variable_raw = fields.get("variable")
    if station_id_raw is None or variable_raw is None:
        return None

    station_id = str(station_id_raw).strip()
    variable = str(variable_raw).strip()
    value = _to_float(fields.get("value"))
    ts = _parse_ts(str(fields.get("ts", "0")))

    if not station_id or not variable or value is None:
        return None

    key = (station_id, variable)
    win = windows.setdefault(key, RollingWindow(window_s=cfg.window_s))
    win.push(ts, float(value))

    samples = win.samples(ts)
    values = [v for _t, v in samples]

    base = dict(analyze_window(values, cfg))

    # basic stats
    n = len(values)
    v_min = float(min(values)) if values else 0.0
    v_max = float(max(values)) if values else 0.0
    v_mean = float(sum(values) / n) if n else 0.0
    v_p95 = _p95(values)

    # holes / gaps
    deltas: list[float] = []
    for i in range(1, len(samples)):
        deltas.append((samples[i][0] - samples[i - 1][0]).total_seconds())

    dt_median = _median(deltas) if deltas else 0.0
    gap_threshold = float(cfg.max_gap_s)
    gaps = [d for d in deltas if d > gap_threshold]
    gap_count = len(gaps)
    gap_max = float(max(gaps)) if gaps else 0.0
    gap_total = float(sum(gaps)) if gaps else 0.0

    # estimate missing time (beyond a reference interval)
    dt_ref = max(dt_median, 1.0)
    missing_time_s = float(sum(max(0.0, d - dt_ref) for d in gaps)) if gaps else 0.0
    missing_time_frac = float(min(1.0, missing_time_s / float(max(1, cfg.window_s))))

    score = float(base.get("score", 0.0))
    state = str(base.get("state", "stable"))

    meteo = _meteo_interpretation(
        state=state,
        score=score,
        n_points=n,
        missing_time_frac=missing_time_frac,
        gap_count=gap_count,
    )

    report: dict[str, Any] = {}
    report.update(base)
    report.update(
        {
            "station_id": station_id,
            "variable": variable,
            "stream_id": msg_id,
            "ts_ingest": float(ts_ingest if ts_ingest is not None else time.time()),
            "stats": {
                "n_points": int(n),
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "p95": v_p95,
                "dt_median_s": float(dt_median),
                "gap_threshold_s": gap_threshold,
                "gap_count": int(gap_count),
                "gap_max_s": gap_max,
                "gap_total_s": gap_total,
                "missing_time_s": missing_time_s,
                "missing_time_frac": missing_time_frac,
            },
            "meteo": meteo,
        }
    )
    return report


def run_live_worker(
    redis_url: str,
    in_stream: str,
    out_stream: str,
    cfg: LiveConfig | None = None,
    start_id: str | None = None,
    max_messages: int | None = None,
    max_idle_s: float | None = None,
) -> None:
    r = make_redis(redis_url)
    cfg = cfg or LiveConfig()

    windows: dict[tuple[str, str], RollingWindow] = {}

    last_id = start_id if start_id is not None else _default_start_id()
    processed = 0
    last_progress = time.time()

    while True:
        resp_any = r.xread({in_stream: last_id}, block=1000, count=200)
        resp = cast(XReadResponse, resp_any)

        if not resp:
            if max_idle_s is not None and (time.time() - last_progress) > float(max_idle_s):
                return
            continue

        for _stream_name, messages in resp:
            for msg_id, fields in messages:
                last_id = msg_id

                report = process_observation(fields, msg_id=msg_id, cfg=cfg, windows=windows)
                if report is None:
                    continue

                last_progress = time.time()
                processed += 1

                station_id = cast(str, report["station_id"])
                variable = cast(str, report["variable"])

                payload = json.dumps(report, separators=(",", ":"), sort_keys=True)

                r.set(_latest_key(station_id, variable), payload)

                out_fields: dict[EncodableT, EncodableT] = {
                    "station_id": station_id,
                    "variable": variable,
                    "payload": payload,
                }
                r.xadd(out_stream, out_fields)

                if max_messages is not None and processed >= int(max_messages):
                    return
