from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chaostrace.orchestrator.sweep import build_grid, sweep


def _parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


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


def _minmax01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    lo = float(np.nanmin(a)) if np.any(np.isfinite(a)) else 0.0
    hi = float(np.nanmax(a)) if np.any(np.isfinite(a)) else 1.0
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
        return np.zeros_like(a, dtype=float)
    y = (a - lo) / (hi - lo)
    y[~np.isfinite(y)] = 0.0
    return y


def _timeline_overlay_plot(
    out_path: Path,
    df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    run_id: int,
    alert_threshold: float,
) -> None:
    """
    Visualisation designed to be readable:
    - speed and foil are normalized to 0..1 (same axis)
    - score_mean on right axis
    - horizontal line at alert_threshold
    """
    import matplotlib.pyplot as plt  # local import

    sub = timeline_df[timeline_df["run_id"] == run_id].copy()
    if sub.empty:
        return

    t = sub["time_s"].to_numpy(dtype=float)
    score = sub["score_mean"].to_numpy(dtype=float)

    foil_col = "foil_height_m" if "foil_height_m" in df.columns else None
    speed_col = None
    for c in ["boat_speed", "boat_speed_mps", "speed_mps", "speed", "v"]:
        if c in df.columns:
            speed_col = c
            break

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    if foil_col and foil_col in df.columns:
        foil = df[foil_col].to_numpy(dtype=float)
        ax1.plot(df["time_s"].to_numpy(dtype=float), _minmax01(foil), label=f"{foil_col} (norm)")

    if speed_col and speed_col in df.columns:
        speed = df[speed_col].to_numpy(dtype=float)
        ax1.plot(df["time_s"].to_numpy(dtype=float), _minmax01(speed), label=f"{speed_col} (norm)")

    ax1.set_xlabel("time_s")
    ax1.set_ylabel("signals (normalized)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, score, label="score_mean")
    ax2.axhline(float(alert_threshold), linestyle="--", linewidth=1.0, label="alert_threshold")
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
    alert_threshold: float,
) -> None:
    import matplotlib.pyplot as plt  # local import
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = timeline_df[timeline_df["run_id"] == run_id].copy()
    if sub.empty:
        return

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
    dt_mask = (~np.isfinite(dt)) | (dt <= 0)
    if np.any(dt_mask):
        good = dt[np.isfinite(dt) & (dt > 0)]
        fill = float(np.nanmedian(good)) if good.size else 1.0
        dt = dt.copy()
        dt[dt_mask] = fill
    accel = np.gradient(speed) / dt

    mask = score > float(alert_threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Two clouds: normal vs alert. The alert cloud is drawn last for visibility.
    ax.scatter(speed[~mask], foil[~mask], accel[~mask], s=6)
    ax.scatter(speed[mask], foil[mask], accel[mask], s=10, c="red")

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
    parser.add_argument("--runs", type=int, default=3, help="Number of configs to execute")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--augment", action="store_true", help="Optional light augmentation/noise")

    parser.add_argument("--baseline-seconds", type=float, default=30.0, help="Baseline window for normalization")
    parser.add_argument("--no-dynamic-weights", action="store_true", help="Disable variance-scaled weights")

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

    metrics_df, timeline_df = sweep(
        df,
        cfgs,
        seed=int(args.seed),
        baseline_seconds=float(args.baseline_seconds),
        use_dynamic_weights=not bool(args.no_dynamic_weights),
    )

    metrics_path = out_dir / "metrics.csv"
    anomalies_path = out_dir / "anomalies.csv"
    fig_phase = out_dir / "fig_phase.png"
    fig_timeline = out_dir / "fig_timeline.png"

    metrics_df.to_csv(metrics_path, index=False)

    anomalies_df = timeline_df[["run_id", "time_s", "score_mean"]].copy()
    anomalies_df.to_csv(anomalies_path, index=False)

    best_run = _best_run_id(metrics_df)

    # Use the threshold computed for the best run when plotting
    thr = 0.55
    if not metrics_df.empty and "alert_threshold_used" in metrics_df.columns:
        row = metrics_df[metrics_df["run_id"] == best_run]
        if not row.empty:
            thr = float(row["alert_threshold_used"].iloc[0])

    _phase_space_plot(fig_phase, df, timeline_df, best_run, thr)
    _timeline_overlay_plot(fig_timeline, df, timeline_df, best_run, thr)

    params = {
        "input": str(input_path),
        "out": str(out_dir),
        "runs": runs,
        "seed": int(args.seed),
        "augment": bool(args.augment),
        "baseline_seconds": float(args.baseline_seconds),
        "use_dynamic_weights": not bool(args.no_dynamic_weights),
        "grid": {
            "window_s": window_s,
            "drop_threshold": drop_threshold,
            "emb_dim": emb_dim,
            "emb_lag": emb_lag,
        },
        "best_run_id": int(best_run),
        "best_run_alert_threshold": float(thr),
    }

    produced = [metrics_path, anomalies_path, fig_phase, fig_timeline]
    _ = write_manifest(out_dir, params=params, files=produced)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
