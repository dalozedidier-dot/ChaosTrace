"""CLI: multiscale RQA over sliding windows.

Example
-------
python -m chaostrace.cli.rqa_multiscale \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/rqa_multiscale \
  --scales 3,5,10,20 \
  --series foil_height_m \
  --drop-threshold 0.30

Outputs
-------
- multiscale_rqa_scale_<Xs>.csv (per-scale)
- multiscale_det_lam.html (Plotly if available, else PNG)
- rqa_multiscale_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from chaostrace.data.ingest import load_timeseries
from chaostrace.features.windowing import estimate_sample_hz
from chaostrace.rqa.advanced import RQAAdvancedConfig, compute_rqa_advanced_from_series
from chaostrace.rqa.multivariate import CrossRQAConfig, compute_cross_rqa


def _parse_scales(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty --scales")
    return out


def _iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 1e-6
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return max(q3 - q1, 1e-6)


def _early_score_relative(
    det: float,
    lam: float,
    trend: float,
    *,
    base_det: float,
    base_lam: float,
    base_trend: float,
    iqr_det: float,
    iqr_lam: float,
    iqr_trend: float,
) -> float:
    # Robust z-scores relative to baseline segment
    z_det = (det - base_det) / max(iqr_det, 1e-6)
    z_lam = (lam - base_lam) / max(iqr_lam, 1e-6)
    z_tr = (trend - base_trend) / max(iqr_trend, 1e-6)

    # Early warning heuristic:
    # - DET down relative to baseline
    # - LAM down relative to baseline
    # - TREND deviation magnitude
    comp_det = float(np.clip(max(0.0, -z_det) / 3.0, 0.0, 1.0))
    comp_lam = float(np.clip(max(0.0, -z_lam) / 3.0, 0.0, 1.0))
    comp_tr = float(np.clip(abs(z_tr) / 3.0, 0.0, 1.0))

    score = 0.50 * comp_det + 0.30 * comp_lam + 0.20 * comp_tr
    return float(np.clip(score, 0.0, 1.0))


def _drop_onsets(time_s: np.ndarray, series: np.ndarray, *, thr: float) -> List[float]:
    m = np.asarray(series, dtype=float) < float(thr)
    onsets: List[float] = []
    for i in range(1, len(m)):
        if bool(m[i]) and not bool(m[i - 1]):
            onsets.append(float(time_s[i]))
    return onsets


def _estimate_leads(
    t_mid: np.ndarray,
    early_score: np.ndarray,
    drop_onsets: List[float],
    *,
    early_window_s: float,
    early_threshold: float,
) -> Tuple[int, List[float]]:
    leads: List[float] = []
    t_mid = np.asarray(t_mid, dtype=float)
    early_score = np.asarray(early_score, dtype=float)
    for onset in drop_onsets:
        # windows whose midpoint is before onset and within early window
        m = (t_mid <= float(onset)) & (t_mid >= float(onset) - float(early_window_s)) & (early_score >= float(early_threshold))
        if not np.any(m):
            continue
        t_first = float(np.min(t_mid[m]))
        leads.append(float(onset) - t_first)
    return int(len(leads)), leads


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChaosTrace: multiscale RQA (windowed) with early warning score.")
    p.add_argument("--input", required=True, help="CSV input with time_s and series columns.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--scales", default="3,5,10,20", help="Comma-separated window scales in seconds.")
    p.add_argument("--series", default="foil_height_m", help="Primary series for RQA.")
    p.add_argument("--drop-threshold", type=float, default=0.30, help="Threshold defining drop regime (for summary stats).")
    p.add_argument("--emb-dim", type=int, default=5)
    p.add_argument("--emb-lag", type=int, default=8)
    p.add_argument("--rr-target", type=float, default=0.02, help="Target recurrence rate for fixed-RR epsilon.")
    p.add_argument("--theiler-window", type=int, default=-1, help="Theiler window. -1 uses emb_lag + 1.")
    p.add_argument("--max-points", type=int, default=500, help="Max points per window for distance computation.")
    p.add_argument("--cross", action="store_true", help="Enable Cross-RQA between series and --cross-series.")
    p.add_argument("--cross-series", default="boat_speed", help="Secondary series for cross-RQA.")
    p.add_argument("--early-window-s", type=float, default=2.0, help="Lead time window to count early warnings before drop onsets.")
    p.add_argument("--baseline-windows", type=int, default=0, help="Baseline window count. 0 picks an automatic value per scale.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_timeseries(Path(str(args.input)))
    if "time_s" not in df.columns:
        raise ValueError("Missing column: time_s")

    time_s = df["time_s"].to_numpy(dtype=float)
    hz = estimate_sample_hz(time_s)
    if hz <= 0:
        hz = 10.0

    scales = _parse_scales(str(args.scales))

    adv_cfg = RQAAdvancedConfig(
        emb_dim=int(args.emb_dim),
        emb_lag=int(args.emb_lag),
        rr_target=float(args.rr_target),
        theiler_window=None if int(args.theiler_window) < 0 else int(args.theiler_window),
        max_points=int(args.max_points),
        enable_network=True,
    )

    cross_cfg = CrossRQAConfig(
        emb_dim=int(args.emb_dim),
        emb_lag=int(args.emb_lag),
        rr_target=float(args.rr_target),
        theiler_window=None if int(args.theiler_window) < 0 else int(args.theiler_window),
        max_points=int(args.max_points),
    )

    all_rows: List[Dict[str, Any]] = []
    per_scale_out: Dict[str, str] = {}

    for s in scales:
        window_n = int(max(10, round(float(s) * float(hz))))
        stride_n = max(1, int(round(0.20 * float(hz))))
        rows: List[Dict[str, Any]] = []

        x_full = df[str(args.series)].to_numpy(dtype=float)

        for start in range(0, len(df) - window_n + 1, stride_n):
            end = start + window_n
            x = x_full[start:end]
            m = compute_rqa_advanced_from_series(x, config=adv_cfg)

            m["t_start_s"] = float(time_s[start])
            m["t_end_s"] = float(time_s[end - 1])
            m["t_mid_s"] = float(np.median(time_s[start:end]))
            m["scale_s"] = float(s)

            if bool(args.cross):
                a = df[str(args.series)].to_numpy(dtype=float)[start:end]
                b = df[str(args.cross_series)].to_numpy(dtype=float)[start:end]
                cm = compute_cross_rqa(a, b, cfg=cross_cfg)
                m.update(cm)

            rows.append(m)

        # Baseline segment and baseline-relative early score
        det_arr = np.asarray([float(r.get("det", 0.0)) for r in rows], dtype=float)
        lam_arr = np.asarray([float(r.get("lam", 0.0)) for r in rows], dtype=float)
        tr_arr = np.asarray([float(r.get("trend", 0.0)) for r in rows], dtype=float)

        if int(args.baseline_windows) > 0:
            bN = int(min(int(args.baseline_windows), len(rows)))
        else:
            bN = int(min(max(6, len(rows) // 5), 25))

        base_det = float(np.nanmedian(det_arr[:bN])) if bN > 0 else float(np.nanmedian(det_arr))
        base_lam = float(np.nanmedian(lam_arr[:bN])) if bN > 0 else float(np.nanmedian(lam_arr))
        base_tr = float(np.nanmedian(tr_arr[:bN])) if bN > 0 else float(np.nanmedian(tr_arr))

        iqr_det = _iqr(det_arr[:bN]) if bN > 0 else _iqr(det_arr)
        iqr_lam = _iqr(lam_arr[:bN]) if bN > 0 else _iqr(lam_arr)
        iqr_tr = _iqr(tr_arr[:bN]) if bN > 0 else _iqr(tr_arr)

        early_scores = []
        for r in rows:
            es = _early_score_relative(
                float(r.get("det", 0.0)),
                float(r.get("lam", 0.0)),
                float(r.get("trend", 0.0)),
                base_det=base_det,
                base_lam=base_lam,
                base_trend=base_tr,
                iqr_det=iqr_det,
                iqr_lam=iqr_lam,
                iqr_trend=iqr_tr,
            )
            r["early_score"] = float(es)
            early_scores.append(float(es))

        out_csv = out_dir / f"multiscale_rqa_scale_{float(s):g}s.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        per_scale_out[str(s)] = str(out_csv.name)
        all_rows.extend(rows)

    # Plot DET / LAM and early score (Plotly if available)
    plot_path = None
    try:
        import plotly.graph_objects as go  # type: ignore

        df_all = pd.DataFrame(all_rows).sort_values(["scale_s", "t_mid_s"])
        fig = go.Figure()
        for sc in sorted(df_all["scale_s"].unique()):
            dfi = df_all[df_all["scale_s"] == sc]
            fig.add_trace(go.Scatter(x=dfi["t_mid_s"], y=dfi["det"], mode="lines", name=f"DET {sc:g}s"))
            fig.add_trace(go.Scatter(x=dfi["t_mid_s"], y=dfi["lam"], mode="lines", name=f"LAM {sc:g}s"))
            fig.add_trace(go.Scatter(x=dfi["t_mid_s"], y=dfi["early_score"], mode="lines", name=f"EW {sc:g}s"))
        fig.update_layout(
            template="plotly_dark",
            title="Multiscale RQA: DET, LAM, Early Warning",
            xaxis_title="time_s",
            yaxis_title="value",
            legend=dict(orientation="h"),
        )
        plot_path = out_dir / "multiscale_det_lam.html"
        fig.write_html(str(plot_path))
    except Exception:
        # Fallback: static plot
        import matplotlib.pyplot as plt

        df_all = pd.DataFrame(all_rows).sort_values(["scale_s", "t_mid_s"])
        fig, ax = plt.subplots(figsize=(10, 5))
        for sc in sorted(df_all["scale_s"].unique()):
            dfi = df_all[df_all["scale_s"] == sc]
            ax.plot(dfi["t_mid_s"], dfi["det"], label=f"DET {sc:g}s")
            ax.plot(dfi["t_mid_s"], dfi["lam"], label=f"LAM {sc:g}s")
        ax.set_xlabel("time_s")
        ax.set_ylabel("value")
        ax.legend(ncol=2, fontsize=8)
        plot_path = out_dir / "multiscale_det_lam.png"
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=180)
        plt.close(fig)

    # Summary with a simple lead-time estimate on the smallest scale
    df_all = pd.DataFrame(all_rows)
    summary: Dict[str, Any] = {
        "input": str(args.input),
        "out": str(args.out),
        "scales_s": scales,
        "series": str(args.series),
        "drop_threshold": float(args.drop_threshold),
        "cross_enabled": bool(args.cross),
        "per_scale_csv": per_scale_out,
        "plot": str(plot_path.name) if plot_path is not None else None,
    }

    # Global stats
    if not df_all.empty:
        summary["det_mean"] = float(np.nanmean(df_all["det"].to_numpy(dtype=float)))
        summary["lam_mean"] = float(np.nanmean(df_all["lam"].to_numpy(dtype=float)))
        summary["trend_mean"] = float(np.nanmean(df_all["trend"].to_numpy(dtype=float)))
        summary["early_score_mean"] = float(np.nanmean(df_all["early_score"].to_numpy(dtype=float)))
        summary["early_score_p95"] = float(np.nanpercentile(df_all["early_score"].to_numpy(dtype=float), 95))

    # Lead estimate on the smallest scale, using early_score threshold
    try:
        sc0 = float(min(scales))
        df_sc = df_all[df_all["scale_s"] == sc0].sort_values("t_mid_s")
        if not df_sc.empty:
            drop_on = _drop_onsets(time_s, df[str(args.series)].to_numpy(dtype=float), thr=float(args.drop_threshold))
            baseN = int(min(max(6, len(df_sc) // 5), 25))
            base_es = df_sc["early_score"].to_numpy(dtype=float)[:baseN]
            thr_es = float(max(0.60, np.nanpercentile(base_es, 99.5)))
            matched, leads = _estimate_leads(
                df_sc["t_mid_s"].to_numpy(dtype=float),
                df_sc["early_score"].to_numpy(dtype=float),
                drop_on,
                early_window_s=float(args.early_window_s),
                early_threshold=float(thr_es),
            )
            summary["early_threshold_used"] = float(thr_es)
            summary["drop_onsets_count"] = int(len(drop_on))
            summary["matched_drop_onsets"] = int(matched)
            summary["lead_median_s"] = float(np.median(leads)) if leads else 0.0
            summary["lead_max_s"] = float(np.max(leads)) if leads else 0.0
    except Exception:
        pass

    (out_dir / "rqa_multiscale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
