from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _random_walk(rng: np.random.Generator, n: int, step_sd: float) -> np.ndarray:
    steps = rng.normal(0.0, step_sd, size=n)
    return np.cumsum(steps)


def _wave_field(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    # Stochastic waves: sum of sinusoids + noise
    f1 = rng.uniform(0.05, 0.12)
    f2 = rng.uniform(0.12, 0.25)
    p1 = rng.uniform(0.0, 2 * np.pi)
    p2 = rng.uniform(0.0, 2 * np.pi)
    a1 = rng.uniform(0.10, 0.25)
    a2 = rng.uniform(0.05, 0.18)
    w = a1 * np.sin(2 * np.pi * f1 * t + p1) + a2 * np.sin(2 * np.pi * f2 * t + p2)
    w += rng.normal(0.0, 0.03, size=t.size)
    return w


def _inject_drops(
    rng: np.random.Generator,
    t: np.ndarray,
    base_foil: np.ndarray,
    profile: str,
    drop_threshold: float,
    burn_in_s: float = 30.0,
) -> np.ndarray:
    """
    Create clear low-foil segments so Markov + drop detectors have something to chew on.
    """
    foil = base_foil.copy()

    duration = float(t[-1] - t[0]) if t.size else 0.0

    burn_in_s = float(max(0.0, burn_in_s))
    burn_in_t = float(t[0] + burn_in_s)

    if profile == "stable":
        return foil

    if profile == "1_2_drops":
        ndrops = int(rng.integers(1, 3))
        t_start = max(burn_in_s, 0.15 * duration)
        times = rng.uniform(t_start, 0.85 * duration, size=ndrops)
        times.sort()
        for tc in times:
            depth = rng.uniform(0.02, 0.08)  # deep enough under typical thresholds
            width = rng.uniform(1.0, 2.5)    # seconds
            # Smooth drop: Gaussian dip
            foil -= (rng.uniform(0.35, 0.55)) * np.exp(-0.5 * ((t - tc) / (width / 2.0)) ** 2)
            # Clamp below threshold for a short plateau near center
            mask = np.abs(t - tc) < (0.25 * width)
            foil[mask] = np.minimum(foil[mask], depth)
        return np.clip(foil, 0.0, 1.0)

    # chaotic: many drops + regime shifts
    # Create a time-varying hazard that increases in "bad" periods
    hazard = 0.0005 + 0.0015 * (np.sin(2 * np.pi * t / max(1.0, duration)) ** 2)
    hazard += 0.0005 * (rng.random(size=t.size))
    events = (rng.random(size=t.size) < hazard) & (t >= burn_in_t)

    idxs = np.where(events)[0]
    for i in idxs:
        tc = float(t[i])
        width = float(rng.uniform(0.4, 1.8))
        depth = float(rng.uniform(0.01, 0.10))
        foil -= float(rng.uniform(0.25, 0.60)) * np.exp(-0.5 * ((t - tc) / (width / 2.0)) ** 2)
        mask = np.abs(t - tc) < (0.20 * width)
        foil[mask] = np.minimum(foil[mask], depth)

    # Add a few longer regime dips
    for _ in range(int(rng.integers(1, 4))):
        tc = float(rng.uniform(max(burn_in_s, 0.2 * duration), 0.8 * duration))
        width = float(rng.uniform(3.0, 8.0))
        foil -= float(rng.uniform(0.05, 0.20)) * np.exp(-0.5 * ((t - tc) / (width / 2.0)) ** 2)

    foil = np.clip(foil, 0.0, 1.0)
    return foil


def generate_sailgp_synth(
    out_csv: Path,
    profile: str,
    duration_s: float = 120.0,
    hz: float = 20.0,
    seed: int = 7,
    drop_threshold: float = 0.20,
) -> None:
    rng = np.random.default_rng(int(seed))

    n = int(round(duration_s * hz))
    n = max(100, n)
    t = np.arange(n, dtype=float) / float(hz)

    # Wind shear: cumulative random walk, plus slow drift
    wind_shear = _random_walk(rng, n, step_sd=0.02)
    wind_shear += 0.2 * np.sin(2 * np.pi * t / max(30.0, duration_s))

    # Waves
    waves = _wave_field(rng, t)

    # Base foil: stable hover around ~0.6 with disturbances (wind + waves)
    foil = 0.60 + 0.10 * np.tanh(0.8 * wind_shear) + 0.08 * waves
    foil += rng.normal(0.0, 0.01, size=n)

    foil = _inject_drops(rng, t, foil, profile=profile, drop_threshold=drop_threshold, burn_in_s=30.0)

    # Brownian-like perturbation on foil dynamics for chaotic profile
    if profile == "chaotic":
        foil += 0.015 * _random_walk(rng, n, step_sd=0.06)
        foil = np.clip(foil, 0.0, 1.0)

    # Boat speed: anti-correlated with bad foil + wind/waves
    base_speed = 38.0 + 2.5 * np.tanh(0.6 * wind_shear) - 6.0 * (foil < drop_threshold).astype(float)
    base_speed += 1.5 * waves
    base_speed += rng.normal(0.0, 0.25, size=n)

    # Add recovery inertia: smooth speed with a low-pass filter
    speed = base_speed.copy()
    alpha = 0.08 if profile != "stable" else 0.05
    for i in range(1, n):
        speed[i] = (1 - alpha) * speed[i - 1] + alpha * speed[i]

    df = pd.DataFrame(
        {
            "time_s": t,
            "foil_height_m": foil.astype(float),
            "boat_speed": speed.astype(float),
            "wind_shear": wind_shear.astype(float),
            "wave_height": waves.astype(float),
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> int:
    p = argparse.ArgumentParser(prog="generate_sailgp_synth")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--profile", choices=["stable", "1_2_drops", "chaotic"], default="stable")
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drop-threshold", type=float, default=0.20)
    args = p.parse_args()

    generate_sailgp_synth(
        out_csv=Path(args.out),
        profile=str(args.profile),
        duration_s=float(args.duration_s),
        hz=float(args.hz),
        seed=int(args.seed),
        drop_threshold=float(args.drop_threshold),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
