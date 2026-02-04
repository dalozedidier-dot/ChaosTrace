ChaosTrace update pack (v0.2 -> v0.2.1)

Included fixes to make the tool discriminative:
- Score normalization to a baseline window (robust median/MAD), so stable does not sit near 0.5 by construction.
- Higher alert threshold and a calibrated per-run threshold (alert_threshold_used), capped to preserve discrimination.
- Dynamic weights scaled by component variance.
- Markov drop uses a persistent drop-state (0.5s) + richer multi-state graph (foil_bin x speed_bin) for unique_edges.
- Timeline plot readability: speed/foil normalized, score on secondary axis, threshold line.
- Phase plot: points above threshold are shown in red.
- Updated synth generator and refreshed test CSVs with real drop segments.

Files changed:
- src/chaostrace/orchestrator/sweep.py
- src/chaostrace/cli/run_sweep.py
- scripts/generate_sailgp_synth.py
- test_data/*.csv
- .github/workflows/ci.yml (adds workflow_dispatch)
