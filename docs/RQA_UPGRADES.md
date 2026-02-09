# ChaosTrace RQA Upgrades

This update adds a more defendable RQA layer designed for:
- reducing false positives on stable data via a fixed recurrence rate (FRR) threshold
- producing early warning signals via multiscale windowed DET, LAM, TREND
- capturing coupling between channels via mdRQA and cross-RQA
- making parameter choice reproducible via grid search or Optuna

## Installation

Core only (no optional extras):

```bash
pip install -e .
```

Optional extras:

- RQA research tooling (pyrqa / pyunicorn):

```bash
pip install -e ".[rqa]"
```

- Optimization via Optuna:

```bash
pip install -e ".[opt]"
```

- Plotly HTML visualizations:

```bash
pip install -e ".[viz]"
```

## What’s new

### 1) Advanced RQA metrics

Module:
- `src/chaostrace/rqa/advanced.py`

Key outputs per window:
- RR, DET, LAM
- L_max, DIV = 1 / L_max
- TREND
- V_max, V_ENT
- RPDE_proxy
- Recurrence network metrics (mean degree, clustering, average path length, diameter)

### 2) Defendable parameter choices

Module:
- `src/chaostrace/rqa/params.py`

Provides:
- tau via autocorrelation or AMI (binned mutual information)
- m via a lightweight False Nearest Neighbors estimator
- epsilon selection by fixed recurrence rate (RR target typically 1–5%)

### 3) Multiscale windowed RQA (early warning)

CLI:
- `python -m chaostrace.cli.rqa_multiscale`

Writes:
- `multiscale_rqa_scale_<Xs>.csv` for each requested scale
- `multiscale_det_lam.html` (Plotly if available; fallback PNG otherwise)
- `rqa_multiscale_summary.json`

Example:

```bash
python -m chaostrace.cli.rqa_multiscale \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/rqa_multiscale \
  --scales 3,5,10,20 \
  --series foil_height_m \
  --drop-threshold 0.30 \
  --plotly
```

### 4) mdRQA and cross-RQA (coupling)

Module:
- `src/chaostrace/rqa/multivariate.py`

Use it to quantify coupling between `foil_height_m` and `boat_speed`.

### 5) RQA parameter optimization

CLI:
- `python -m chaostrace.cli.rqa_optimize`

Outputs:
- `rqa_param_optimization.csv`
- `best_rqa_params.json`

Example:

```bash
python -m chaostrace.cli.rqa_optimize \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/rqa_opt \
  --method grid \
  --objective delta_det
```
