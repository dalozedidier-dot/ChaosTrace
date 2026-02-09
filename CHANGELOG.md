## 0.2.5

- Early-warning DL: train on *drop onset* targets (predict the start of a drop, not the full drop segment), reducing label leakage and improving lead time.
- DL inference: causal forward-fill only (no backward fill), so scores at time t never use future anchors.
- Event-level metrics: fix contiguous-event extraction edge case.
- CI: hybrid_dl workflow now enforces early-warning lead_s_max > 0 and F1 >= 0.90; uses tighter stride/horizon for earlier alerts.

## 0.2.4

- Fix sample-rate estimation for DL training (correct window/stride/horizon in samples)
- Make DL windows causal and anchor-aligned for true early-warning labeling
- Improve fusion threshold baseline (early segment) to avoid threshold inflation near events
- Relax fusion threshold clamp (allow earlier alerts) and export alert_frac/threshold in metrics
- Ruff: ignore E401 in notebooks

# Changelog
## 0.2.2
- Fix CI ruff error (unused import) in hybrid Matrix Profile module.
- Add optional DL subpackage (chaostrace.hybrid.dl) with minimal training/inference pipeline.
- Fix train_hybrid/run_hybrid imports and make DL truly optional via extras.
- Remove __pycache__ artifacts from distribution.

## 0.2.3 (2026-02-09)
- Packaging: remove __pycache__ and .pytest_cache from bundle, keep repo clean.
- Remove duplicate experimental `chaostrace.dl` package (hybrid DL lives under `chaostrace.hybrid.dl`).
- Add tests to ensure hybrid CLI/modules import without optional extras (torch, stumpy).



## 0.2.1
- Fix: add missing hybrid modules (matrix_profile, causal drift proxy, metrics) and make chaostrace.hybrid a real package.
- Hybrid CLI run_hybrid now runs end-to-end without optional deps unless enabled.

## 0.2.0
- Ajout du mode hybride (fusion chaos + options ML)
- Ajout des CLI `run_hybrid` et `train_hybrid`
- Seuil dynamique + post-traitement event-level pour réduire les faux positifs
- RQA et Lyapunov proxy bornés (downsampling) pour accélérer les sweeps
- Ajout Matrix Profile (optionnel) et drift causal VAR(1)

## 0.1.0
- Scaffold initial
