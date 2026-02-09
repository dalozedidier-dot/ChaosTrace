# Viz Cinematic Suite

Cette suite ajoute des visualisations Plotly interactives au style cinématographique, en conservant l'interprétabilité chaos.

Ce que tu obtiens
- Espace de phase 3D interactif (HTML), trajectoire + points colorés par score
- Timeline interactive (HTML) avec zones invariants et variants, drops et alertes
- Recurrence plot (HTML) en heatmap, avec surcouches des drops
- Dashboard (HTML) qui regroupe tout en un seul fichier
- Animation optionnelle de la trajectoire dans l'espace de phase

## Dépendances

Les visualisations Plotly sont optionnelles. Elles ne sont pas requises pour le coeur du toolkit.

```bash
pip install -r requirements_viz.txt
```

Exporter des PNG haute résolution est optionnel.

```bash
pip install kaleido
```

## Usage

Après un run (sweep ou hybrid) qui produit un dossier contenant `metrics.csv`, `anomalies.csv` et `manifest.json`:

```bash
python -m chaostrace.cli.viz_cinematic --run-dir _ci_out/smoke
```

Sortie par défaut
`_ci_out/smoke/viz_cinematic/`
- `01_phase_space_3d.html`
- `02_timeline_interactive.html`
- `03_recurrence_plot.html`
- `04_dashboard.html`
- `metadata.json`

Animation (plus lourd)

```bash
python -m chaostrace.cli.viz_cinematic --run-dir _ci_out/smoke --make-animation
```

PNG (si kaleido est installé)

```bash
python -m chaostrace.cli.viz_cinematic --run-dir _ci_out/smoke --export-png
```

## Résolution de l'input

La timeline et le Takens embedding utilisent idéalement l'input original (ex: `boat_speed`, `foil_height_m`).

Le script tente de le retrouver via `manifest.json` (champ `params.input`).
Si ton dossier de run est déplacé, donne le chemin:

```bash
python -m chaostrace.cli.viz_cinematic --run-dir results/run_001 --input path/to/original.csv
```

Si les colonnes ne sont pas disponibles, la suite dégrade proprement (score seulement).
