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


def _apply_precursor(
    t: np.ndarray,
    *,
    t0: float,
    pre_drop_s: float,
    foil: np.ndarray,
    speed: np.ndarray,
    wind_shear: np.ndarray,
    wave: np.ndarray,
    rng: np.random.Generator,
    foil_depth: float = 0.10,
    speed_dip: float = 0.60,
    shear_bump: float = 0.08,
    extra_noise: float = 0.02,
) -> None:
    """Inject a causal pre-drop precursor (early-warning signal) before a forced drop.

    This is intentionally subtle: it should not cross typical `drop_threshold` values, but
    should create measurable non-stationarity (variance + drift + coupling) that chaos/RQA/DL
    can pick up 1-2 seconds early.
    """
    if pre_drop_s <= 0:
        return

    start = float(t0) - float(pre_drop_s)
    if start < float(t[0]):
        start = float(t[0])

    m = (t >= start) & (t < float(t0))
    if not np.any(m):
        return

    # phase ramps from 0 -> 1 in the precursor window
    phase = (t[m] - float(start)) / max(float(t0) - float(start), 1e-9)

    # foil: gentle downward drift + increasing jitter (but do NOT make it a drop yet)
    foil[m] = foil[m] - float(foil_depth) * phase + rng.normal(0.0, float(extra_noise), size=int(np.sum(m))) * phase

    # speed: small dip + jitter
    speed[m] = speed[m] - float(speed_dip) * phase + rng.normal(0.0, 0.06, size=int(np.sum(m))) * phase

    # wind and waves: mild regime change
    wind_shear[m] = wind_shear[m] + float(shear_bump) * phase + rng.normal(0.0, 0.02, size=int(np.sum(m))).cumsum() / max(int(np.sum(m)), 1)
    wave[m] = wave[m] + rng.normal(0.0, 0.03, size=int(np.sum(m))) * phase


def generate_profile(
    *,
    profile: str,
    seed: int = 7,
    dt_s: float = 0.05,
    duration_s: float = 120.0,
    pre_drop_s: float = 0.0,
    pre_drop_depth: float = 0.10,
) -> pd.DataFrame:
    """Generate synthetic SailGP-like timeseries.

    Profiles:
      - stable: smooth, no foil drops
      - drops: two forced foil drops (58-62s and 92-93s)
      - chaotic: multiple random drops + stronger perturbations

    Early-warning option:
      - if pre_drop_s > 0, inject a subtle precursor before forced drops/bursts so that
        early-warning lead time is physically plausible in offline evaluation.
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
        t10, t1a = 58.0, 62.0
        t20, t2a = 92.0, 93.0

        # Optional pre-drop precursor before each forced drop.
        _apply_precursor(
            t,
            t0=t10,
            pre_drop_s=float(pre_drop_s),
            foil=foil,
            speed=speed,
            wind_shear=wind_shear,
            wave=wave,
            rng=rng,
            foil_depth=float(pre_drop_depth),
        )
        _apply_precursor(
            t,
            t0=t20,
            pre_drop_s=float(pre_drop_s),
            foil=foil,
            speed=speed,
            wind_shear=wind_shear,
            wave=wave,
            rng=rng,
            foil_depth=float(pre_drop_depth) * 0.7,
            speed_dip=0.45,
        )

        m1 = (t >= t10) & (t <= t1a)
        foil[m1] = 0.05 + 0.01 * rng.normal(0.0, 1.0, size=int(m1.sum()))

        m2 = (t >= t20) & (t <= t2a)
        foil[m2] = 0.10 + 0.01 * rng.normal(0.0, 1.0, size=int(m2.sum()))

        # Amplify transitions (speed jitter as accel proxy) and wind perturbation.
        speed = speed + rng.normal(0.0, 0.12, size=len(t))
        wind_shear = wind_shear + rng.normal(0.0, 0.05, size=len(t)).cumsum() / max(len(t), 1)

    elif profile == "chaotic":
        foil = foil.copy()

        # Multi-drop bursts (random) + stronger perturbations
        p = 0.0025  # per-sample at 20Hz ~6 bursts per 120s
        drop_starts = np.where(rng.random(len(t)) < p)[0]
        pre_n = int(round(float(pre_drop_s) / float(dt_s))) if pre_drop_s > 0 else 0

        for i in drop_starts:
            # optional precursor window before this burst
            if pre_n > 0:
                j0 = max(0, i - pre_n)
                # fabricate a pseudo time t0 at sample i
                t0 = float(t[i])
                _apply_precursor(
                    t[j0 : i + 1],
                    t0=t0,
                    pre_drop_s=float(pre_drop_s),
                    foil=foil[j0 : i + 1],
                    speed=speed[j0 : i + 1],
                    wind_shear=wind_shear[j0 : i + 1],
                    wave=wave[j0 : i + 1],
                    rng=rng,
                    foil_depth=float(pre_drop_depth) * 0.8,
                    speed_dip=0.50,
                    shear_bump=0.06,
                    extra_noise=0.03,
                )  # slices are views, edits propagate

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
    ap.add_argument(
        "--pre-drop-s",
        type=float,
        default=0.0,
        help="Optional precursor duration in seconds before forced drops/bursts.",
    )
    ap.add_argument(
        "--pre-drop-depth",
        type=float,
        default=0.10,
        help="Approx foil drift depth during precursor (meters).",
    )
    args = ap.parse_args()

    df = generate_profile(
        profile=args.profile,
        seed=args.seed,
        pre_drop_s=float(args.pre_drop_s),
        pre_drop_depth=float(args.pre_drop_depth),
    )
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)


if __name__ == "__main__":
    main()
