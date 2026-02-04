from __future__ import annotations

import numpy as np
import pandas as pd

def add_realistic_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    wind_shear_sigma: float = 0.25,
    wave_short_sigma: float = 0.02,
    sensor_sigma: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if sensor_sigma is None:
        sensor_sigma = {
            "boat_speed": 0.05,
            "foil_height_m": 0.01,
            "vmg": 0.05,
            "pitch_deg": 0.05,
            "roll_deg": 0.05,
        }

    n = len(out)
    shear = rng.normal(0.0, wind_shear_sigma, size=n).cumsum() / max(n, 1)
    out["wind_speed"] = out["wind_speed"] + shear

    out["pitch_deg"] = out["pitch_deg"] + rng.normal(0.0, wave_short_sigma, size=n)
    out["roll_deg"] = out["roll_deg"] + rng.normal(0.0, wave_short_sigma, size=n)
    out["foil_height_m"] = out["foil_height_m"] + rng.normal(0.0, wave_short_sigma, size=n)

    for col, sig in sensor_sigma.items():
        if col in out.columns:
            out[col] = out[col] + rng.normal(0.0, sig, size=n)

    return out

def tiny_perturb(df: pd.DataFrame, rng: np.random.Generator, eps: float = 1e-3) -> pd.DataFrame:
    out = df.copy()
    for col in ["wind_speed", "wind_angle_deg", "foil_rake_deg", "daggerboard_depth_m"]:
        out[col] = out[col] + rng.normal(0.0, eps, size=len(out))
    return out
