from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Minimal requirement for ChaosTrace: a time axis.
# Everything else is optional and can be missing in real-world CSV exports.
TIME_ALIASES = ("time_s", "time", "t", "timestamp_s", "seconds")

FOIL_ALIASES = ("foil_height_m", "foil_height", "foil_m", "foil")
SPEED_ALIASES = ("boat_speed", "boat_speed_mps", "speed_mps", "speed", "v")

OPTIONAL_COLS_DEFAULTS: dict[str, float] = {
    "foil_height_m": np.nan,
    "boat_speed": np.nan,
    "heading_deg": np.nan,
    "wind_speed": np.nan,
    "wind_angle_deg": np.nan,
    "foil_rake_deg": np.nan,
    "daggerboard_depth_m": np.nan,
    "vmg": np.nan,
    "pitch_deg": np.nan,
    "roll_deg": np.nan,
}


def _first_present(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_timeseries(path: str | Path) -> pd.DataFrame:
    """
    Load a timeseries file (CSV or JSON) into a canonical dataframe.

    Canonical columns guaranteed:
      - time_s

    Canonical columns best-effort (created if possible, otherwise present as NaN):
      - foil_height_m
      - boat_speed
      - heading_deg, wind_speed, wind_angle_deg, foil_rake_deg, daggerboard_depth_m, vmg, pitch_deg, roll_deg

    This is intentionally permissive to support "real data ingestion" where many
    sensors/fields might be missing. Downstream analyzers should degrade gracefully.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        data: Any = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported input: {p.suffix}")

    if df.empty:
        raise ValueError("Empty input dataset")

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Find/rename time column
    time_col = _first_present(df, TIME_ALIASES)
    if time_col is None:
        raise ValueError("Missing required column: time_s (or an alias)")

    if time_col != "time_s":
        df = df.rename(columns={time_col: "time_s"})

    # Alias important signals to canonical names if needed
    foil_col = _first_present(df, FOIL_ALIASES)
    if foil_col is not None and foil_col != "foil_height_m":
        df = df.rename(columns={foil_col: "foil_height_m"})

    speed_col = _first_present(df, SPEED_ALIASES)
    if speed_col is not None and speed_col != "boat_speed":
        df = df.rename(columns={speed_col: "boat_speed"})

    # Add missing optional columns
    for c, default in OPTIONAL_COLS_DEFAULTS.items():
        if c not in df.columns:
            df[c] = default

    # Coerce numeric types for known fields
    _coerce_numeric(df, ["time_s", *list(OPTIONAL_COLS_DEFAULTS.keys())])

    # If VMG is missing but we have boat_speed + wind_angle_deg, compute a simple proxy
    if "vmg" in df.columns and df["vmg"].isna().all():
        if "boat_speed" in df.columns and "wind_angle_deg" in df.columns:
            ang = np.deg2rad(df["wind_angle_deg"].to_numpy(dtype=float))
            spd = df["boat_speed"].to_numpy(dtype=float)
            vmg = spd * np.cos(ang)
            df["vmg"] = vmg

    # Sort on time, drop duplicates, reset index
    df = df.sort_values("time_s", kind="mergesort").drop_duplicates(subset=["time_s"]).reset_index(drop=True)

    return df
