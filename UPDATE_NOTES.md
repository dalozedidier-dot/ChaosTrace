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

## v0.2 update (discriminative v2)
- CSV output stabilized with float_format=%.6f to avoid tiny cross-version diffs.
- Added representative plots: fig_phase_repr.png and fig_timeline_repr.png (median-ish run),
  while keeping fig_phase.png / fig_timeline.png as the most-anomalous run for continuity.
- Manifest now records best_run_id / repr_run_id and their thresholds.
