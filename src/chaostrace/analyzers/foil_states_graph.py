from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from .base import AnalyzerResult

def _state(h: float, t1: float, t2: float) -> str:
    if h < t1:
        return "low"
    if h < t2:
        return "mid"
    return "high"

def foil_state_graph(df: pd.DataFrame, t1: float = 0.30, t2: float = 0.80) -> AnalyzerResult:
    h = df["foil_height_m"].to_numpy(dtype=float)
    states = [_state(float(v), t1, t2) for v in h]

    G = nx.DiGraph()
    for s in ["low", "mid", "high"]:
        G.add_node(s, count=0)
    for s in states:
        G.nodes[s]["count"] += 1

    transitions = 0
    for a, b in zip(states[:-1], states[1:]):
        transitions += 1
        if G.has_edge(a, b):
            G[a][b]["w"] += 1
        else:
            G.add_edge(a, b, w=1)

    low_runs = 0
    cur = 0
    for s in states:
        if s == "low":
            cur += 1
            low_runs = max(low_runs, cur)
        else:
            cur = 0

    timeline = pd.DataFrame({"time_s": df["time_s"], "state": states})
    metrics = {
        "transitions": float(transitions),
        "unique_edges": float(G.number_of_edges()),
        "max_low_run": float(low_runs),
        "p_low": float(np.mean([s == "low" for s in states])),
    }
    return AnalyzerResult(name="foil_state_graph", metrics=metrics, timeline=timeline)
