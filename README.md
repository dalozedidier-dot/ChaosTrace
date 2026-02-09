# ChaosTrace

ChaosTrace est un toolkit léger basé sur des signaux de chaos (Takens, RQA, Lyapunov proxy) pour détecter des transitions de régime et des événements de type foil drop dans des séries temporelles multivariées.

Cette version ajoute un mode hybride, conçu pour augmenter précision et recall tout en gardant l'interprétabilité par les signaux chaos.

## Installation

Core (sans ML) :
```bash
pip install -e ".[dev]"
```

Options :
- Matrix Profile :
```bash
pip install -e ".[mp]"
```

- Deep learning (PyTorch) :
```bash
pip install -e ".[dl]"
```

## Usage

### Sweep chaos (core)
```bash
python -m chaostrace.cli.run_sweep --input test_data/sample_timeseries.csv --out _ci_out/run
```

Fichiers générés :
- metrics.csv
- anomalies.csv
- fig_phase.png
- fig_timeline.png
- fig_timeline_inv_var.png
- manifest.json

### Mode hybride (fusion chaos + MP + causal + DL optionnel)

Exemple hybride sans DL :
```bash
python -m chaostrace.cli.run_hybrid --input test_data/sample_timeseries.csv --out _ci_out/hybrid --enable-causal
```

Avec Matrix Profile :
```bash
python -m chaostrace.cli.run_hybrid --input test_data/sample_timeseries.csv --out _ci_out/hybrid --enable-mp --mp-col boat_speed
```

Avec DL (si un modèle a été entraîné) :
```bash
python -m chaostrace.cli.run_hybrid --input test_data/sample_timeseries.csv --out _ci_out/hybrid --model _ci_out/model_dir
```

Sorties hybrides :
- anomalies_hybrid.csv (scores et alertes)
- metrics_hybrid.json (PRF, event-level, lead times)
- explain_hybrid.jsonl (snapshots explicables sur les points alertés)
- fig_timeline_hybrid.png
- fig_phase_hybrid.png
- manifest.json

## Entraînement DL (optionnel)

Le CLI `train_hybrid` génère des labels `is_drop` via Markov (si absents), puis entraîne un modèle Conv + Transformer sur des fenêtres.

```bash
python -m chaostrace.cli.train_hybrid \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/model_dir \
  --cols boat_speed,foil_height_m \
  --window-s 5 \
  --stride-s 0.5 \
  --horizon-s 0.0 \
  --drop-threshold 0.4 \
  --supervised-epochs 10
```

Pour un prétrain contrastif (SimCLR simplifié) :
```bash
python -m chaostrace.cli.train_hybrid \
  --input test_data/sample_timeseries_1_2_drops.csv \
  --out _ci_out/model_dir \
  --cols boat_speed,foil_height_m \
  --window-s 5 \
  --stride-s 0.5 \
  --horizon-s 1.5 \
  --drop-threshold 0.4 \
  --contrastive-epochs 5 \
  --supervised-epochs 10
```

## Notes de design

- Interprétabilité : les scores chaos restent visibles, et la fusion conserve les composantes et les poids.
- Robustesse : seuil dynamique calculé sur une baseline stable, puis post-traitement event-level (merge gaps, suppression spikes).
- Dépendances : les modules MP et DL sont optionnels, le core reste léger.
