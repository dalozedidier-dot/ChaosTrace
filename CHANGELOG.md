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
