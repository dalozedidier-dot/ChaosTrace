# ChaosTrace

Objectif: faire émerger du « silence » (absences, transitions subtiles, régimes stables/instables) au milieu d'un chaos multivarié type F50.
Ce repo est un squelette modulaire, CI-friendly, orienté sweeps paramétriques et artefacts auditables (JSON/CSV + hashes + manifest).

## Ce que tu as tout de suite
- Ingestion CSV/JSON (schema simple, extensible)
- Génération synthétique (bruit réaliste + perturbations minimes pour tester la sensibilité)
- Orchestrateur de sweeps (fenêtres, seuils drop, embedding Takens, lags)
- Analyseurs plug-and-play:
  - Null-trace (laminarité / stabilité anormale)
  - Delta-stats (variations relatives locales)
  - Markov drop (chaîne de Markov sur états foil)
  - Graph-based (NetworkX: transitions d'états, cycles, dead-ends)
  - RQA légère (sans dépendance lourde, métriques de base)
  - Lyapunov-like (approx Rosenstein simplifiée)
- Sorties auditables dans `_ci_out/` avec `manifest.json` (hashes, params, versions)
- CI GitHub Actions: pytest + un mini sweep de smoke

## Démarrage rapide
### 1) Installer
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2) Lancer un sweep
```bash
python -m chaostrace.cli.run_sweep \
  --input test_data/sample_timeseries.csv \
  --out _ci_out/demo \
  --runs 25
```

### 3) Lire les résultats
- `_ci_out/demo/manifest.json` : paramètres + hashes + versions
- `_ci_out/demo/metrics.csv` : métriques par run
- `_ci_out/demo/anomalies.csv` : score timeline agrégé par run
- `_ci_out/demo/fig_phase.png` : portrait de phase 3D (matplotlib)

## Convention de données (minimum)
Colonnes attendues (tu peux en ajouter):
- time_s
- boat_speed
- heading_deg
- wind_speed
- wind_angle_deg
- foil_height_m
- foil_rake_deg
- daggerboard_depth_m
- vmg
- pitch_deg
- roll_deg

Le sample dans `test_data/` est synthétique.

## Philosophie (auditabilité)
- Chaque run est déterministe si tu fixes `--seed`.
- Chaque run écrit un manifest avec hash SHA256 des fichiers produits.
- Pas d'inférence implicite: un analyseur doit écrire ses métriques explicitement.



## Synthetic samples
- `test_data/sample_timeseries_stable.csv`
- `test_data/sample_timeseries_1_2_drops.csv`
- `test_data/sample_timeseries_chaotic.csv`
