# ChaosTrace – Patch complet (v0.3.1)

Objectif: rendre l’outil **discriminatoire** (pas “tout est anomal”), activer Markov/foil_state_graph sur un dataset de test avec drops forcés, et exporter la séparation **invariants / variantes** + visualisation.

## Ce que ce patch applique

1) **Packaging corrigé**
- Ajoute la découverte de packages (src-layout) dans `pyproject.toml` :
  - `[tool.setuptools.packages.find] where=["src"]`
=> `pip install -e ".[dev]"` installe bien `chaostrace`.

2) **Orchestrator sweep rétabli + utile**
- `src/chaostrace/orchestrator/sweep.py`:
  - `build_grid(...)` + `sweep(df, cfgs, seed=...)` (compatible tests/CI).
  - Suite analyzers branchée: `null_trace`, `delta_stats`, `markov_drop`, `rqa_light`, `lyapunov_like`, `foil_state_graph`.
  - Normalisation robuste (percentile) pour éviter le score plat.
  - Deux séries exportées: `score_invariant`, `score_variant` + `score_mean`.
  - `alert_threshold` fixé à **0.55** (au lieu de 0.18) + `alert_count/alert_frac`.
  - `foil_state_graph` reçoit un `t2` dynamique (autour de la médiane foil) pour activer les 3 états et augmenter `unique_edges`.

3) **CLI run_sweep conforme aux workflows**
- `src/chaostrace/cli/run_sweep.py`:
  - Supporte `--input --out --runs --seed` + grilles `--window-s --drop-threshold --emb-dim --emb-lag`.
  - Produit: `metrics.csv`, `anomalies.csv`, `fig_phase.png`, `fig_timeline.png`, `fig_timeline_inv_var.png`, `manifest.json`.
  - Phase-space: points “anomalies” (score_mean>0.55) en rouge.

4) **Drops forcés dans le dataset de test**
- `test_data/sample_timeseries_1_2_drops.csv` régénéré:
  - drop 58–62s (~0.05m)
  - drop 92–93s (~0.10m)
=> Markov (`p01/p10`) et `unique_edges` s’activent.

5) **CI smoke plus pertinente**
- `.github/workflows/ci.yml` utilise `sample_timeseries_1_2_drops.csv` pour le smoke sweep.

6) **Générateur synth amélioré**
- `scripts/generate_sailgp_synth.py`:
  - profils `stable | drops | chaotic`
  - drops forcés identiques au dataset de test.

7) **Interface (optionnelle)**
- `app.py` (Streamlit) pour explorer rapidement invariants/variantes.
  - Dépendances non ajoutées au projet par défaut.
  - Usage local: `pip install streamlit` puis `streamlit run app.py`.

## Comment l’appliquer

À la racine de ton repo ChaosTrace:

1) Dézippe ce patch **à la racine** (overwrite autorisé).
2) Commit.
3) Test local rapide:
   - `pip install -e ".[dev]"`
   - `python -m ruff check .`
   - `python -m pytest`
   - `python -m chaostrace.cli.run_sweep --input test_data/sample_timeseries_1_2_drops.csv --out _ci_out/demo --runs 3 --seed 7`

## Résultat attendu (sanity checks)

- `metrics.csv` doit contenir:
  - `markov_p01 > 0` et `markov_p10 > 0` sur le dataset drops
  - `foil_state_unique_edges > 4`
  - `alert_frac` non saturé à 1.0

- `anomalies.csv` doit contenir:
  - `score_invariant` et `score_variant` (variations visibles autour des drops)
