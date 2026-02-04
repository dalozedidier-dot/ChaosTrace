from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

REQUIRED_COLS = [
    "time_s",
    "boat_speed",
    "heading_deg",
    "wind_speed",
    "wind_angle_deg",
    "foil_height_m",
    "foil_rake_deg",
    "daggerboard_depth_m",
    "vmg",
    "pitch_deg",
    "roll_deg",
]

def load_timeseries(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported input: {p.suffix}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values("time_s").reset_index(drop=True)
