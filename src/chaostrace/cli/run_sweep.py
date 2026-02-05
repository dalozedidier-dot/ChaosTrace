
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt
from chaostrace.orchestrator.sweep import sweep, ALERT_THRESHOLD

def _norm(x):
    x = (x - x.min())/(x.max()-x.min()+1e-9)
    return x

def plot(df, tl, out):
    fig, ax = plt.subplots()

    foil = _norm(df.get("foil_height_m", df.iloc[:,1]))
    speed = _norm(df.get("boat_speed", df.iloc[:,2]))

    ax.plot(df["time_s"], foil, label="foil")
    ax.plot(df["time_s"], speed, label="speed")
    ax.plot(tl["time_s"], tl["score_invariant"], linewidth=3, label="invariant")
    ax.plot(tl["time_s"], tl["score_variant"], linestyle="--", label="variant")

    ax.axhline(ALERT_THRESHOLD, linestyle=":")
    ax.legend()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    metrics, tl = sweep(df)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out/"metrics.csv", index=False)
    tl.to_csv(out/"anomalies.csv", index=False)
    plot(df, tl, out/"fig_timeline_inv_var.png")

if __name__ == "__main__":
    main()
