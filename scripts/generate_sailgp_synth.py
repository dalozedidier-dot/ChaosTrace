from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _base_series(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth baseline with mild noise."""
    boat_speed = 37.0 + 0.6 * np.sin(t / 12.0) + rng.normal(0.0, 0.05, size=len(t))
    foil_height = 0.58 + 0.02 * np.sin(t / 6.0) + rng.normal(0.0, 0.01, size=len(t))

    wind_shear = rng.normal(0.0, 0.15, size=len(t)).cumsum() / max(len(t), 1)
    wave_height = 0.02 * np.sin(t / 2.0) + rng.normal(0.0, 0.02, size=len(t))
    return foil_height, boat_speed, wind_shear, wave_height


def generate_profile(
    *,
    profile: str,
    seed: int = 7,
    dt_s: float = 0.05,
    duration_s: float = 120.0,
) -> pd.DataFrame:
    """Generate synthetic SailGP-like timeseries.

    Profiles:
      - stable: smooth, no foil drops
      - drops: two forced foil drops (58-62s and 92-93s)
      - chaotic: multiple random drops + stronger perturbations
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)

    foil, speed, wind_shear, wave = _base_series(t, rng)

    if profile == "stable":
        foil = foil + 0.005 * rng.normal(0.0, 1.0, size=len(t))
        speed = speed + 0.05 * rng.normal(0.0, 1.0, size=len(t))

    elif profile == "drops":
        foil = foil.copy()

        # Forced drops, per spec:
        # - around 55-65s: 2-4 seconds at ~0.05m (here 58-62s)
        # - around 90-95s: 1 second at ~0.10m (here 92-93s)
        m1 = (t >= 58.0) & (t <= 62.0)
        foil[m1] = 0.05 + 0.01 * rng.normal(0.0, 1.0, size=int(m1.sum()))

        m2 = (t >= 92.0) & (t <= 93.0)
        foil[m2] = 0.10 + 0.01 * rng.normal(0.0, 1.0, size=int(m2.sum()))

        # Amplify transitions (speed jitter as accel proxy) and wind perturbation.
        speed = speed + rng.normal(0.0, 0.12, size=len(t))
        wind_shear = wind_shear + rng.normal(0.0, 0.05, size=len(t)).cumsum() / max(len(t), 1)

    elif profile == "chaotic":
        foil = foil.copy()

        # Multi-drop bursts (random) + stronger perturbations
        p = 0.0025  # per-sample at 20Hz ~6 bursts per 120s
        drop_starts = np.where(rng.random(len(t)) < p)[0]
        for i in drop_starts:
            k = int(rng.integers(10, 60))  # 0.5s to 3s
            foil[i : i + k] = np.maximum(0.0, foil[i : i + k] - rng.uniform(0.4, 0.7))

        speed = speed + rng.normal(0.0, 0.20, size=len(t))
        wind_shear = wind_shear + rng.normal(0.0, 0.10, size=len(t)).cumsum() / max(len(t), 1)
        wave = wave + 0.04 * rng.normal(0.0, 1.0, size=len(t))

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return pd.DataFrame(
        {
            "time_s": t,
            "foil_height_m": foil,
            "boat_speed": speed,
            "wind_shear": wind_shear,
            "wave_height": wave,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="stable", choices=["stable", "drops", "chaotic"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="test_data/sample_timeseries_stable.csv")
    args = ap.parse_args()

    df = generate_profile(profile=args.profile, seed=args.seed)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)


if __name__ == "__main__":
    main()
