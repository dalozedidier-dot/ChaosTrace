from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chaostrace.viz.static import save_phase, save_timeline


def test_static_figures_are_written(tmp_path: Path) -> None:
    n = 80
    t = np.linspace(0.0, 8.0, n)
    df = pd.DataFrame(
        {
            "time_s": t,
            "foil_height_m": np.sin(t),
            "boat_speed": 8 + 0.4 * np.cos(t),
            "is_drop": (t > 4.0) & (t < 5.0),
        }
    )
    tl = pd.DataFrame(
        {
            "score_mean": 0.2 + 0.6 * (t > 4.0),
            "score_invariant": 0.3 + 0.2 * np.sin(t),
            "score_variant": 0.1 + 0.7 * (t > 4.0),
        }
    )
    p1, p2 = save_timeline(df, tl, tmp_path, threshold=0.5)
    p3 = save_phase(df, tl, tmp_path, threshold=0.5)
    assert p1.exists() and p1.stat().st_size > 1000
    assert p2.exists() and p2.stat().st_size > 1000
    assert p3.exists() and p3.stat().st_size > 1000
