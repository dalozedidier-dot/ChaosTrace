# ChaosTrace sur Google Colab

Objectif: fournir une démo exécutable, reproductible, et alignée avec la CI, sans setup local.

## Fichiers fournis

- `notebooks/ChaosTrace_Colab_Demo.ipynb`  
  Notebook Colab qui:
  - vérifie Python >= 3.11
  - clone le repo et installe en mode editable (`pip install -e ".[dev]"`)
  - exécute `run_sweep` (core) puis `run_hybrid` (caual, MP optionnel, DL optionnel)
  - affiche les figures, lit `metrics.csv` et `anomalies.csv`
  - zippe `_ci_out` et propose un téléchargement

## Ouvrir directement dans Colab

```text
https://colab.research.google.com/github/dalozedidier-dot/ChaosTrace/blob/main/notebooks/ChaosTrace_Colab_Demo.ipynb
```

## Recommandations d'intégration

1. Ajouter le notebook au repo sous `notebooks/`.
2. Ajouter un paragraphe dans le `README.md` avec le lien Colab ci-dessus.
3. Garder le notebook strictement aligné avec les CLI officiels (`python -m chaostrace.cli.*`) pour éviter les écarts entre CI, local, et Colab.
