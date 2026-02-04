# ChaosTrace

**Trace the chaos. Hear the silence before the drop.**

A lightweight Python toolkit to detect **regime shifts**, **early foil-drop precursors**, and **subtle anomalies** in high-frequency multivariate sailing telemetry (F50 / SailGP-style).

ChaosTrace focuses on **parametric sweeps**, **phase-space reconstruction**, and **chaos-inspired diagnostics** to surface “silence” inside the storm: abnormal stability, transitions, and weak signals that often precede instability.

## Philosophy

In high-performance foiling, the most dangerous moments are often preceded by silence, not noise. ChaosTrace makes that silence measurable.

## Key features

- **Massive parametric sweeps**: windows, thresholds, embeddings, lags (CI-friendly)
- **Plug-and-play analyzers**:
  - **Null-trace**: suspiciously stable / laminar windows
  - **Delta-stats**: local relative variations
  - **Markov drop**: transition probabilities between regimes
  - **Graph-based**: NetworkX state transitions, cycles, dead-ends
  - **RQA light**: recurrence metrics (lightweight)
  - **Lyapunov-like**: simplified divergence estimation
- **Auditability**: SHA256 hashes + `manifest.json` with parameters/versions, deterministic runs with `--seed`
- **No GPU / no heavy DL**: numpy/scipy/pandas + NetworkX, minimal dependencies

## Quick start

```bash
git clone https://github.com/dalozedidier-dot/ChaosTrace.git
cd ChaosTrace

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -e ".[dev]"

python -m chaostrace.cli.run_sweep \
  --input test_data/sample_timeseries.csv \
  --out _ci_out/demo \
  --runs 25 \
  --seed 42
