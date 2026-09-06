from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from chaostrace.phase.embedding import takens_embedding

FOIL = "#0072B2"
SPEED = "#009E73"
SCORE = "#D55E00"
INVARIANT = "#56B4E9"
VARIANT = "#E69F00"
ALERT = "#CC3311"
MUTED = "#9AA0A6"


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _apply_style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _shade_events(ax, t: np.ndarray, mask: np.ndarray, *, color: str, label: str | None) -> None:
    if mask.size != t.size or not np.any(mask):
        return
    labeled = False
    in_run = False
    start = 0
    for i, flag in enumerate(mask):
        if flag and not in_run:
            start = i
            in_run = True
        elif not flag and in_run:
            ax.axvspan(
                t[start],
                t[i - 1],
                color=color,
                alpha=0.12,
                label=label if not labeled else None,
                zorder=0,
            )
            labeled = True
            in_run = False
    if in_run:
        ax.axvspan(
            t[start],
            t[-1],
            color=color,
            alpha=0.12,
            label=label if not labeled else None,
            zorder=0,
        )


def save_timeline(
    df: pd.DataFrame,
    tl: pd.DataFrame,
    outp: Path,
    *,
    threshold: float,
) -> tuple[Path, Path]:
    t = df["time_s"].to_numpy(dtype=float)
    foil = df.get("foil_height_m", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    speed = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    score = tl["score_mean"].to_numpy(dtype=float)
    drop = None
    if "is_drop" in df.columns:
        drop = df["is_drop"].to_numpy() > 0.5
    elif "is_drop" in tl.columns:
        drop = tl["is_drop"].to_numpy() > 0.5

    fig, ax1 = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
    ax1.plot(t, foil, color=FOIL, lw=1.4, label="foil height (m)")
    ax1.plot(t, speed, color=SPEED, lw=1.2, alpha=0.9, label="boat speed")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("signals")
    _apply_style(ax1)

    ax2 = ax1.twinx()
    ax2.plot(t, score, color=SCORE, lw=1.8, label="chaos score")
    ax2.axhline(float(threshold), color=MUTED, ls="--", lw=1.0, label=f"threshold {threshold:.2f}")
    ax2.set_ylabel("score")
    ax2.spines["top"].set_visible(False)
    if drop is not None:
        _shade_events(ax1, t, drop.astype(bool), color=ALERT, label="drop")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)
    ax1.set_title("Signals and chaos score")
    p1 = outp / "fig_timeline.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
    foil_n = _norm01(foil)
    speed_n = _norm01(speed)
    inv = tl["score_invariant"].to_numpy(dtype=float)
    var = tl["score_variant"].to_numpy(dtype=float)

    ax.plot(t, foil_n, color=FOIL, lw=1.0, alpha=0.7, label="foil (norm)")
    ax.plot(t, speed_n, color=SPEED, lw=1.0, alpha=0.7, label="speed (norm)")
    ax.plot(t, inv, color=INVARIANT, lw=2.4, label="invariant score")
    ax.plot(t, var, color=VARIANT, lw=1.6, ls="--", label="variant score")
    ax.axhline(float(threshold), color=MUTED, ls=":", lw=1.0, label=f"threshold {threshold:.2f}")

    inv_zone = inv > 0.6
    var_zone = (var > 0.5) & (inv < 0.4)
    _shade_events(ax, t, inv_zone, color=INVARIANT, label="invariant zone")
    _shade_events(ax, t, var_zone, color=VARIANT, label="variant zone")
    if drop is not None:
        _shade_events(ax, t, drop.astype(bool), color=ALERT, label="drop")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized units")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Invariant vs variant score")
    _apply_style(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    p2 = outp / "fig_timeline_inv_var.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p1, p2


def save_phase(
    df: pd.DataFrame,
    tl: pd.DataFrame,
    outp: Path,
    *,
    threshold: float,
) -> Path:
    x = df.get("boat_speed", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    X = takens_embedding(x, dim=3, lag=3)
    p = outp / "fig_phase.png"
    if len(X) == 0:
        fig = plt.figure(figsize=(7.2, 6.2))
        fig.suptitle("Phase space (empty embedding)")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return p

    score = tl["score_mean"].to_numpy(dtype=float)
    score = score[-len(X) :]
    hi = score > float(threshold)

    fig = plt.figure(figsize=(7.4, 6.4), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        X[~hi, 0],
        X[~hi, 1],
        X[~hi, 2],
        s=8,
        alpha=0.35,
        c=score[~hi],
        cmap="viridis",
        linewidths=0,
    )
    if np.any(hi):
        ax.scatter(
            X[hi, 0],
            X[hi, 1],
            X[hi, 2],
            s=18,
            alpha=0.95,
            c=ALERT,
            linewidths=0,
            label=f"score > {threshold:.2f}",
        )
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$x(t+\tau)$")
    ax.set_zlabel(r"$x(t+2\tau)$")
    ax.set_title("Takens embedding of boat speed")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p
