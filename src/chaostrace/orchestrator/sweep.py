
import numpy as np
import pandas as pd

ALERT_THRESHOLD = 0.55

def _z(x):
    x = np.asarray(x, float)
    s = np.std(x) or 1.0
    return (x - np.mean(x)) / s

def _sig(x):
    return 1/(1+np.exp(-x))

def sweep(df, window=10):
    if "time_s" not in df:
        raise ValueError("time_s column required")

    speed = df.get("boat_speed", df.iloc[:,1]).to_numpy(float)
    foil  = df.get("foil_height_m", np.zeros_like(speed))

    roll_std = pd.Series(speed).rolling(window).std().fillna(0).to_numpy()
    null_trace = _sig(-_z(roll_std))

    delta = np.abs(np.gradient(speed)) + np.abs(np.gradient(foil))
    delta_stats = _sig(_z(delta))

    low = (foil < 0.2).astype(int)
    transitions = np.abs(np.diff(low, prepend=low[:1]))
    markov = _sig(_z(transitions))

    rqa = _sig(_z(pd.Series(speed).rolling(window).mean().fillna(0)))

    score = 0.4*null_trace + 0.3*delta_stats + 0.2*markov + 0.1*rqa
    alerts = score > ALERT_THRESHOLD

    invariant = float(np.clip(0.4*np.mean(null_trace)+0.3*np.mean(rqa)+0.2*(1-np.mean(delta_stats)),0,1))
    variant   = float(np.clip(0.4*np.mean(delta_stats)+0.3*(1-np.mean(rqa))+0.2*np.mean(markov),0,1))

    metrics = pd.DataFrame([{
        "score_mean": float(np.mean(score)),
        "score_invariant": invariant,
        "score_variant": variant,
        "alert_frac": float(np.mean(alerts))
    }])

    timeline = pd.DataFrame({
        "time_s": df["time_s"],
        "score_mean": score,
        "score_invariant": invariant,
        "score_variant": variant
    })

    return metrics, timeline
