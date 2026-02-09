from __future__ import annotations

"""Advanced (but still lightweight) RQA metrics.

This module offers "real" RQA metrics beyond simple proxies:
- RR, DET, LAM
- L_max, DIV = 1 / L_max
- TREND (non-stationarity proxy)
- V_max, V_ENT (entropy of vertical line lengths)
- RPDE_proxy (period density entropy proxy)
- Recurrence network metrics (mean degree, clustering, avg path length, diameter)

Optional dependencies
---------------------
If installed, `chaostrace[rqa]` can enable richer computations via:
- pyrqa
- pyunicorn

This file intentionally keeps a NumPy/SciPy implementation as default, to
remain CI-friendly and deterministic.
"""

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.stats

import networkx as nx

from chaostrace.rqa.params import fixed_recurrence_rate_epsilon, takens_embedding_1d


@dataclass(frozen=True)
class RQAAdvancedConfig:
    """Configuration for advanced RQA computations."""

    emb_dim: int = 5
    emb_lag: int = 8

    threshold_by: str = "frr"  # "frr" or "fixed"
    rr_target: float = 0.02
    epsilon: Optional[float] = None

    theiler_window: Optional[int] = None
    l_min: int = 2
    v_min: int = 2

    max_points: int = 600
    rng_seed: int = 7
    enable_network: bool = True
    max_network_points: int = 300


def _downsample_points(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    stride = int(np.ceil(x.shape[0] / max_points))
    return x[::stride]


def _recurrence_matrix(points: np.ndarray, *, epsilon: float, theiler: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n < 2:
        return np.zeros((n, n), dtype=bool)

    d = ssd.squareform(ssd.pdist(pts, metric="euclidean"))
    R = d <= float(epsilon)
    np.fill_diagonal(R, False)

    if theiler > 0:
        tw = int(theiler)
        for k in range(-tw, tw + 1):
            if k == 0:
                continue
            R &= ~np.eye(n, k=k, dtype=bool)
    return R


def _line_lengths_diagonal(R: np.ndarray) -> np.ndarray:
    """Return all diagonal line lengths (>=1) across all diagonals."""
    n = R.shape[0]
    lengths: list[int] = []
    # offsets: negative to positive
    for k in range(-(n - 1), n):
        diag = np.diagonal(R, offset=k)
        if diag.size == 0:
            continue
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _line_lengths_vertical(R: np.ndarray) -> np.ndarray:
    """Return all vertical line lengths (>=1) across columns."""
    n = R.shape[0]
    lengths: list[int] = []
    for j in range(n):
        col = R[:, j]
        run = 0
        for v in col:
            if v:
                run += 1
            else:
                if run > 0:
                    lengths.append(run)
                    run = 0
        if run > 0:
            lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _shannon_entropy(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / float(np.sum(c))
    return float(-np.sum(p * np.log(p)))


def _trend_metric(R: np.ndarray) -> float:
    """TREND proxy: slope of recurrence rate vs diagonal distance."""
    n = R.shape[0]
    if n < 10:
        return 0.0
    offsets = []
    rates = []
    for k in range(1, n):  # ignore k=0
        diag_pos = np.diagonal(R, offset=k)
        diag_neg = np.diagonal(R, offset=-k)
        if diag_pos.size == 0:
            continue
        rate_k = float(np.mean(diag_pos.astype(float)))
        rate_k2 = float(np.mean(diag_neg.astype(float))) if diag_neg.size > 0 else rate_k
        offsets.extend([k, -k])
        rates.extend([rate_k, rate_k2])

    if len(offsets) < 6:
        return 0.0

    x = np.asarray(offsets, dtype=float)
    y = np.asarray(rates, dtype=float)
    slope, _, _, _, _ = scipy.stats.linregress(x, y)
    return float(slope)


def _rpde_proxy(R: np.ndarray) -> float:
    """Recurrence period density entropy (proxy).

    We collect distances between consecutive recurrence points per row,
    then compute Shannon entropy of the pooled histogram.
    """
    n = R.shape[0]
    periods: list[int] = []
    for i in range(n):
        js = np.flatnonzero(R[i])
        if js.size < 2:
            continue
        diffs = np.diff(js)
        diffs = diffs[diffs > 0]
        periods.extend([int(d) for d in diffs])
    if not periods:
        return 0.0
    periods_arr = np.asarray(periods, dtype=int)
    # Histogram up to a reasonable max
    max_p = int(np.clip(np.max(periods_arr), 2, 200))
    bins = np.arange(1, max_p + 2)
    counts, _ = np.histogram(periods_arr, bins=bins)
    return _shannon_entropy(counts)


def _recurrence_network_metrics(R: np.ndarray, *, max_points: int) -> Dict[str, float]:
    n = R.shape[0]
    if n < 5 or n > max_points:
        return {
            "rn_mean_degree": float("nan"),
            "rn_clustering": float("nan"),
            "rn_avg_path_len": float("nan"),
            "rn_diameter": float("nan"),
        }

    G = nx.from_numpy_array(R.astype(int))
    # Use largest connected component for path metrics.
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {
            "rn_mean_degree": 0.0,
            "rn_clustering": 0.0,
            "rn_avg_path_len": float("nan"),
            "rn_diameter": float("nan"),
        }

    degrees = np.asarray([d for _, d in G.degree()], dtype=float)
    rn_mean_degree = float(np.mean(degrees)) if degrees.size else 0.0
    rn_clustering = float(nx.average_clustering(G)) if G.number_of_edges() > 0 else 0.0

    # Path metrics on the giant component
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len) if comps else set()
    if len(giant) < 5:
        return {
            "rn_mean_degree": rn_mean_degree,
            "rn_clustering": rn_clustering,
            "rn_avg_path_len": float("nan"),
            "rn_diameter": float("nan"),
        }

    H = G.subgraph(giant).copy()
    try:
        rn_avg_path_len = float(nx.average_shortest_path_length(H))
        rn_diameter = float(nx.diameter(H))
    except Exception:
        rn_avg_path_len = float("nan")
        rn_diameter = float("nan")

    return {
        "rn_mean_degree": rn_mean_degree,
        "rn_clustering": rn_clustering,
        "rn_avg_path_len": rn_avg_path_len,
        "rn_diameter": rn_diameter,
    }


def compute_rqa_advanced_from_series(
    x: np.ndarray,
    *,
    config: RQAAdvancedConfig,
) -> Dict[str, Any]:
    """Compute advanced RQA metrics from a 1D series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return {"rr": 0.0, "det": 0.0, "lam": 0.0, "l_max": 0, "div": float("inf"), "trend": 0.0}

    emb = takens_embedding_1d(x, m=int(config.emb_dim), tau=int(config.emb_lag))
    emb = _downsample_points(emb, int(config.max_points))

    if config.threshold_by == "fixed":
        eps = float(config.epsilon) if config.epsilon is not None else 0.0
    else:
        eps = fixed_recurrence_rate_epsilon(emb, rr_target=float(config.rr_target), rng_seed=int(config.rng_seed))

    theiler = int(config.theiler_window) if config.theiler_window is not None else int(config.emb_lag) + 1
    R = _recurrence_matrix(emb, epsilon=eps, theiler=theiler)

    n = R.shape[0]
    denom = float(n * n - n) if n > 1 else 1.0
    rr = float(np.sum(R)) / denom

    diag_lengths = _line_lengths_diagonal(R)
    diag_sel = diag_lengths[diag_lengths >= int(config.l_min)]
    det_num = float(np.sum(diag_sel)) if diag_sel.size else 0.0
    det_den = float(np.sum(diag_lengths)) if diag_lengths.size else 0.0
    det = (det_num / det_den) if det_den > 0 else 0.0
    l_max = int(np.max(diag_sel)) if diag_sel.size else 0
    div = float(1.0 / l_max) if l_max > 0 else float("inf")

    vert_lengths = _line_lengths_vertical(R)
    vert_sel = vert_lengths[vert_lengths >= int(config.v_min)]
    lam_num = float(np.sum(vert_sel)) if vert_sel.size else 0.0
    lam_den = float(np.sum(vert_lengths)) if vert_lengths.size else 0.0
    lam = (lam_num / lam_den) if lam_den > 0 else 0.0
    v_max = int(np.max(vert_sel)) if vert_sel.size else 0

    # Vertical entropy: distribution of vertical line lengths >= v_min
    if vert_sel.size:
        uniq, counts = np.unique(vert_sel, return_counts=True)
        v_ent = _shannon_entropy(counts.astype(float))
    else:
        v_ent = 0.0

    trend = _trend_metric(R)
    rpde = _rpde_proxy(R)

    out: Dict[str, Any] = {
        "rr": rr,
        "det": det,
        "lam": lam,
        "l_max": l_max,
        "div": div,
        "trend": trend,
        "v_max": v_max,
        "v_ent": v_ent,
        "rpde_proxy": rpde,
        "epsilon": float(eps),
        "theiler_window": int(theiler),
        "n_points": int(n),
    }

    if config.enable_network:
        out.update(_recurrence_network_metrics(R, max_points=int(config.max_network_points)))

    return out


def compute_rqa_advanced_windowed(
    df: pd.DataFrame,
    *,
    time_s: np.ndarray,
    series_col: str,
    window_n: int,
    step_n: int,
    config: RQAAdvancedConfig,
) -> pd.DataFrame:
    """Compute advanced RQA metrics over sliding windows."""
    x_full = np.asarray(df[series_col], dtype=float)
    t_full = np.asarray(time_s, dtype=float)

    rows: list[Dict[str, Any]] = []
    n = len(df)
    window_n = int(max(20, window_n))
    step_n = int(max(1, step_n))

    for start in range(0, n - window_n + 1, step_n):
        end = start + window_n
        x = x_full[start:end]
        t_mid = float(np.median(t_full[start:end]))
        m = compute_rqa_advanced_from_series(x, config=config)
        m["t_mid_s"] = t_mid
        rows.append(m)

    return pd.DataFrame(rows)


def write_rqa_outputs(
    out_dir: str | Path,
    df_metrics: pd.DataFrame,
    *,
    prefix: str = "rqa_adv",
) -> None:
    """Write CSV + summary JSON."""
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    csv_path = outp / f"{prefix}.csv"
    df_metrics.to_csv(csv_path, index=False, float_format="%.6f")

    summary: Dict[str, Any] = {"rows": int(len(df_metrics))}
    for col in df_metrics.columns:
        if col == "t_mid_s":
            continue
        s = pd.to_numeric(df_metrics[col], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            continue
        summary[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    (outp / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def rqa_early_warning_score(metrics_row: Dict[str, Any]) -> float:
    """A simple interpretable early-warning score from RQA metrics.

    Heuristic:
    - lower DET and lower LAM indicate a transition to less structured dynamics
    - absolute TREND indicates non-stationarity
    """
    det = float(metrics_row.get("det", 0.0))
    lam = float(metrics_row.get("lam", 0.0))
    trend = float(metrics_row.get("trend", 0.0))
    score = 0.55 * (1.0 - det) + 0.35 * (1.0 - lam) + 0.10 * min(1.0, abs(trend) * 100.0)
    return float(np.clip(score, 0.0, 1.0))
