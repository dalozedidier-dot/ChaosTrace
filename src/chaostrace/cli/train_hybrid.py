from __future__ import annotations

import argparse
from pathlib import Path

from chaostrace.analyzers.markov_drop import markov_drop
from chaostrace.data.ingest import load_timeseries
from chaostrace.features.windowing import estimate_sample_hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output directory for model artifacts")
    ap.add_argument(
        "--cols",
        default="boat_speed,foil_height_m",
        help="Comma-separated feature columns used for training/inference",
    )
    ap.add_argument("--window-s", type=float, default=5.0, help="Window length in seconds")
    ap.add_argument("--stride-s", type=float, default=0.5, help="Stride in seconds")
    ap.add_argument(
        "--horizon-s",
        type=float,
        default=1.0,
        help="Early warning horizon: label is 1 if a drop occurs within this future window",
    )

    ap.add_argument("--drop-threshold", type=float, default=0.35, help="For autolabeling if is_drop missing")
    ap.add_argument("--contrastive-epochs", type=int, default=0, help="Optional contrastive pretrain epochs")
    ap.add_argument("--supervised-epochs", type=int, default=10, help="Supervised training epochs")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="Positive class weight for BCE")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=7, help="Seed for deterministic training")

    args = ap.parse_args()

    df = load_timeseries(Path(args.input))

    if "time_s" in df.columns:
        hz = float(estimate_sample_hz(df["time_s"]))
    else:
        hz = float(estimate_sample_hz(df.index.to_numpy()))

    if "is_drop" not in df.columns:
        res = markov_drop(df, drop_threshold=float(args.drop_threshold))
        df = df.copy()
        df["is_drop"] = res.timeline["is_drop"].to_numpy(dtype=float)

    cols = [c.strip() for c in str(args.cols).split(",") if c.strip()]

    window_n = max(int(round(float(args.window_s) * hz)), 5)
    stride_n = max(int(round(float(args.stride_s) * hz)), 1)
    horizon_n = max(int(round(float(args.horizon_s) * hz)), 0)

    from chaostrace.hybrid.dl.train import train_hybrid_model

    out_dir = Path(args.out)
    model_path = train_hybrid_model(
        df,
        out_dir=out_dir,
        cols=cols,
        window_n=window_n,
        stride_n=stride_n,
        horizon_n=horizon_n,
        contrastive_epochs=int(args.contrastive_epochs),
        supervised_epochs=int(args.supervised_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        pos_weight=float(args.pos_weight),
        device=str(args.device),
        seed=int(args.seed),
    )
    print(str(model_path))


if __name__ == "__main__":
    main()
