from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chaostrace.orchestrator.sweep import build_grid, sweep


def _parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, params: dict, files: list[Path]) -> Path:
    manifest = {
        "params": params,
        "files": [
            {
                "path": str(f.relative_to(out_dir)),
                "sha256": _sha256_file(f),
                "bytes": f.stat().st_size,
            }
            for f in files
        ],
    }
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _best_run_id(metrics_df: pd.DataFrame) -> int:
    if metrics_df.empty or "score_mean" not in metrics_df.columns:
        return 1
    idx = int(metrics_df["score_mean"].astype(float).idxmax())
    return int(metrics_df.loc[idx, "run_id"])


def _timeline_overlay_plot(
    out_path: Path,
    df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    run_id: int,
) -> None:
    import matplotlib.pyplot as plt  # local import for faster CLI start

    sub = timeline_df[timeline_df["run_id"] == run_id].copy()
    if sub.empty:
        return

    t = sub["time_s"].to_numpy(dtype=float)
    score = sub["score_mean"].to_numpy(dtype=float)

    foil_col = "foil_height_m" if "foil_height_m" in df.columns else None
    speed_col = "boat_speed" if "boat_speed" in df.columns else None
    if speed_col is None:
        for c in ["boat_speed_mps", "speed_mps", "speed", "v"]:
            if c in df.columns:
                speed_col = c
                break

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    if foil_col and foil_col in df.columns:
        ax1.plot(df["time_s"].to_numpy(dtype=float), df[foil_col].to_numpy(dtype=float), label=foil_col)

    if speed_col and speed_col in df.columns:
        ax1.plot(df["time_s"].to_numpy(dtype=float), df[speed_col].to_numpy(dtype=float), label=speed_col)

    ax1.set_xlabel("time_s")
    ax1.set_ylabel("signals")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, score, label="score_mean")
    ax2.set_ylabel("score_mean")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _phase_space_plot(
    out_path: Path,
    df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    run_id: int,
) -> None:
    import matplotlib.pyplot as plt  # local import
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = timeline_df[timeline_df["run_id"] == run_id].copy()
    if sub.empty:
        return

    # Choose 3 axes. Prefer speed + foil + derivative proxy.
    t = sub["time_s"].to_numpy(dtype=float)
    score = sub["score_mean"].to_numpy(dtype=float)

    foil = df["foil_height_m"].to_numpy(dtype=float) if "foil_height_m" in df.columns else np.zeros_like(t)
    speed_col = None
    for c in ["boat_speed", "boat_speed_mps", "speed_mps", "speed", "v"]:
        if c in df.columns:
            speed_col = c
            break
    speed = df[speed_col].to_numpy(dtype=float) if speed_col else np.zeros_like(t)

    dt = np.gradient(t)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nanmedian(dt[np.isfinite(dt) & (dt > 0)]) if np.any(np.isfinite(dt) & (dt > 0)) else 1.0
    accel = np.gradient(speed) / dt

    # Color rule requested: score_mean > 0.20 in red.
    mask = score > 0.20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(speed[~mask], foil[~mask], accel[~mask], s=6)
    ax.scatter(speed[mask], foil[mask], accel[mask], s=10)

    ax.set_xlabel(speed_col or "speed")
    ax.set_ylabel("foil_height_m")
    ax.set_zlabel("accel")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(prog="chaostrace.run_sweep")
    parser.add_argument("--input", required=True, help="CSV input (must contain time_s)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (configs) to execute")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--augment", action="store_true", help="Optional light augmentation/noise")

    parser.add_argument("--window-s", default="3,5,10,20", help="Comma list, seconds")
    parser.add_argument("--drop-threshold", default="0.10,0.20,0.30,0.40", help="Comma list")
    parser.add_argument("--emb-dim", default="3,4,5", help="Comma list")
    parser.add_argument("--emb-lag", default="1,3,5,8,12", help="Comma list")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "time_s" not in df.columns:
        raise SystemExit("Input CSV must contain a time_s column")

    df = df.sort_values("time_s", kind="mergesort").reset_index(drop=True)

    rng = np.random.default_rng(int(args.seed))

    if bool(args.augment):
        # Very light augmentation: small gaussian noise on numeric columns (except time_s)
        for c in df.columns:
            if c == "time_s":
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                x = df[c].to_numpy(dtype=float)
                sd = float(np.nanstd(x))
                if np.isfinite(sd) and sd > 0:
                    df[c] = x + rng.normal(0.0, 0.01 * sd, size=x.shape[0])

    window_s = _parse_list_int(args.window_s)
    drop_threshold = _parse_list_float(args.drop_threshold)
    emb_dim = _parse_list_int(args.emb_dim)
    emb_lag = _parse_list_int(args.emb_lag)

    grid = build_grid(window_s=window_s, drop_threshold=drop_threshold, emb_dim=emb_dim, emb_lag=emb_lag)
    if not grid:
        raise SystemExit("Empty sweep grid")

    runs = int(args.runs)
    if runs <= 0:
        raise SystemExit("--runs must be > 0")

    # Deterministic selection of cfgs
    if runs <= len(grid):
        idx = rng.choice(len(grid), size=runs, replace=False)
    else:
        idx = rng.choice(len(grid), size=runs, replace=True)

    cfgs = [grid[int(i)] for i in idx]

    metrics_df, timeline_df = sweep(df, cfgs, seed=int(args.seed))

    metrics_path = out_dir / "metrics.csv"
    anomalies_path = out_dir / "anomalies.csv"
    fig_phase = out_dir / "fig_phase.png"
    fig_timeline = out_dir / "fig_timeline.png"

    metrics_df.to_csv(metrics_path, index=False)

    anomalies_df = timeline_df[["run_id", "time_s", "score_mean"]].copy()
    anomalies_df.to_csv(anomalies_path, index=False)

    best_run = _best_run_id(metrics_df)

    _phase_space_plot(fig_phase, df, timeline_df, best_run)
    _timeline_overlay_plot(fig_timeline, df, timeline_df, best_run)

    params = {
        "input": str(input_path),
        "out": str(out_dir),
        "runs": runs,
        "seed": int(args.seed),
        "augment": bool(args.augment),
        "grid": {
            "window_s": window_s,
            "drop_threshold": drop_threshold,
            "emb_dim": emb_dim,
            "emb_lag": emb_lag,
        },
        "best_run_id": int(best_run),
    }

    produced = [metrics_path, anomalies_path, fig_phase, fig_timeline]
    manifest_path = write_manifest(out_dir, params=params, files=produced)

    # Make sure manifest exists and is last
    _ = manifest_path

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
