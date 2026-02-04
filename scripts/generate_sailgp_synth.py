#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def brownian(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return rng.normal(0.0, sigma, size=n).cumsum()

def generate(
    n: int,
    hz: float,
    seed: int,
    foil_drop_prob_per_s: float,
    drop_threshold: float = 0.30,
    wind_shear_sigma: float = 0.35,
    wave_sigma: float = 0.04,
    foil_brown_sigma: float = 0.003,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz

    # Wind: base + sinus + cumulative shear (random walk) + gust noise
    base_wind = 12.0 + 1.8 * np.sin(2 * np.pi * t / 120.0)
    shear = brownian(rng, n, sigma=wind_shear_sigma) / max(n, 1)
    gust = rng.normal(0.0, 0.25, size=n)
    wind_speed = base_wind + shear + gust

    wind_angle = 30.0 + 10.0 * np.sin(2 * np.pi * t / 60.0) + rng.normal(0.0, 0.7, size=n)

    # Boat speed coupled to wind + low frequency oscillations + sensor noise
    boat_speed = 34.0 + 4.5 * np.sin(2 * np.pi * t / 40.0) + 0.25 * wind_speed + rng.normal(0.0, 0.35, size=n)

    # Foil dynamics: baseline + waves (sinus + stochastic) + brownian on change
    wave = wave_sigma * (np.sin(2 * np.pi * t / 2.8) + 0.6 * np.sin(2 * np.pi * t / 1.4))
    foil_rw = brownian(rng, n, sigma=foil_brown_sigma) / max(hz, 1.0)
    foil_height = 0.85 + 0.06 * np.sin(2 * np.pi * t / 25.0) + wave + foil_rw + rng.normal(0.0, 0.012, size=n)

    # Inject drop events (prob per second -> prob per sample)
    p_sample = np.clip(foil_drop_prob_per_s / hz, 0.0, 0.25)
    i = 0
    while i < n:
        if rng.random() < p_sample:
            drop_len = int(rng.integers(int(0.25 * hz), int(1.2 * hz) + 1))
            drop_len = max(drop_len, 2)
            end = min(n, i + drop_len)

            # ramp down to below threshold, then noisy floor, then exponential recovery
            ramp = np.linspace(0.0, 0.75, end - i)
            foil_height[i:end] -= ramp

            floor = drop_threshold * rng.uniform(0.4, 0.9)
            foil_height[i:end] = np.minimum(foil_height[i:end], floor + rng.normal(0.0, 0.01, size=end - i))

            rec_len = int(rng.integers(int(0.3 * hz), int(1.5 * hz) + 1))
            rec_end = min(n, end + rec_len)
            if rec_end > end:
                tau = rng.uniform(0.4, 1.4)
                x = np.linspace(0.0, 3.0, rec_end - end)
                foil_height[end:rec_end] += 0.18 * np.exp(-x / tau)

            i = rec_end
        else:
            i += 1

    foil_height = np.clip(foil_height, 0.05, 1.2)

    foil_rake = 4.0 + 0.6 * np.sin(2 * np.pi * t / 33.0) + rng.normal(0.0, 0.06, size=n)
    dagger = 1.8 + 0.22 * np.sin(2 * np.pi * t / 45.0) + rng.normal(0.0, 0.03, size=n)
    heading = (90.0 + 15.0 * np.sin(2 * np.pi * t / 200.0) + rng.normal(0.0, 0.25, size=n)) % 360.0
    vmg = boat_speed * np.cos(np.deg2rad(wind_angle)) + rng.normal(0.0, 0.25, size=n)

    # Pitch/Roll: waves + noise (more chaotic when foil close to threshold)
    instab = np.clip((drop_threshold + 0.05 - foil_height) / (drop_threshold + 0.05), 0.0, 1.0)
    pitch = 2.0 + 0.9 * np.sin(2 * np.pi * t / 15.0) + wave_sigma * 2.0 * rng.normal(0.0, 1.0, size=n) + 0.6 * instab * rng.normal(0.0, 1.0, size=n)
    roll = 1.0 + 0.7 * np.sin(2 * np.pi * t / 12.0) + wave_sigma * 2.0 * rng.normal(0.0, 1.0, size=n) + 0.5 * instab * rng.normal(0.0, 1.0, size=n)

    return pd.DataFrame({
        "time_s": t,
        "boat_speed": boat_speed,
        "heading_deg": heading,
        "wind_speed": wind_speed,
        "wind_angle_deg": wind_angle,
        "foil_height_m": foil_height,
        "foil_rake_deg": foil_rake,
        "daggerboard_depth_m": dagger,
        "vmg": vmg,
        "pitch_deg": pitch,
        "roll_deg": roll,
    })

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--drop-threshold", type=float, default=0.30)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = int(args.seconds * args.hz)

    # 3 regimes by seed/prob
    stable = generate(n, args.hz, seed=7,  foil_drop_prob_per_s=0.00, drop_threshold=args.drop_threshold)
    one_two = generate(n, args.hz, seed=42, foil_drop_prob_per_s=0.02, drop_threshold=args.drop_threshold)
    chaotic = generate(n, args.hz, seed=1337, foil_drop_prob_per_s=0.05, drop_threshold=args.drop_threshold)

    stable.to_csv(out / "sample_timeseries_stable.csv", index=False)
    one_two.to_csv(out / "sample_timeseries_1_2_drops.csv", index=False)
    chaotic.to_csv(out / "sample_timeseries_chaotic.csv", index=False)

    print(f"Wrote 3 files to: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
