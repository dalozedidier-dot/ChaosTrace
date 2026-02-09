# Changelog
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
