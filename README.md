# htr-medieval-manuscripts-2026

Pipeline HTR (Handwritten Text Recognition) pour manuscrits médiévaux en ancien/moyen français.  
**Projet MD5 — Master Data/IA — Module Vision par ordinateur — HETIC 2026**

## Équipe

| Membre | Rôle |
|---|---|
| À compléter | Responsable technique |
| À compléter | Responsable données |
| À compléter | Responsable expérimentation |
| À compléter | Responsable documentation |

## Résultats

| Métrique | Valeur | Seuil validation |
|---|---|---|
| CER global | À compléter | < 15% |
| WER global | À compléter | < 25% |
| IoU segmentation | À compléter | > 0.75 |

## Installation

```bash
git clone https://github.com/[groupe]/htr-medieval-manuscripts-2026
cd htr-medieval-manuscripts-2026
pip install -e ".[dev]"
```

## Reproduire les résultats

```bash
# 1. Prétraitement (deskew → CLAHE → Sauvola)
htmir-preprocess --input data/raw/ --output data/preprocessed/

# 2. Segmentation + export PAGE XML (à implémenter)
# python -m htmir.segmentation.lines --input data/preprocessed/ --output segmentations/

# 3. Fine-tuning TrOCR + LoRA (config : configs/trocr_lora_r8.yaml)
# python -m htmir.htr.finetune_trocr --config configs/trocr_lora_r8.yaml

# 4. Fine-tuning Kraken
# ketos train -o outputs/kraken/ -f alto segmentations/*.xml

# 5. Évaluation sur le test scellé (UNE SEULE FOIS)
# python -m htmir.evaluation.evaluate_final --model outputs/trocr_lora/ --test data/test/

# 6. Tests
pytest tests/ -v
```

## Hash SHA-256 du test set

```
À compléter après constitution du split — NE PAS REGARDER LE TEST SET AVANT LE RENDU
```

## Structure du dépôt

```
HTmiR_project/
├── src/htmir/              # package Python installable
│   ├── preprocessing/      # deskew, CLAHE, Sauvola, pipeline
│   ├── segmentation/       # layout, lignes, export PAGE XML
│   ├── htr/                # TrOCR+LoRA, Kraken, baseline
│   ├── aggregation/        # Needleman-Wunsch, confiance, data contract
│   ├── evaluation/         # CER, WER, bootstrap, McNemar
│   ├── utils/              # seeds, logger, io
│   └── cli/                # commandes (htmir-preprocess, …)
├── tests/                  # suite pytest
├── configs/                # hyperparamètres YAML
├── docs/                   # conventions, sources, model card, article
├── experiments/            # journal des runs (journal.jsonl)
├── data/
│   ├── raw/                # images sources (gitignored si volumineux)
│   └── preprocessed/       # sortie prétraitement
├── segmentations/          # PAGE XML par page
├── dataset_nlp/            # JSON validé (volet NLP)
├── notebooks/              # exploration et visualisation
├── outputs/                # checkpoints et modèles (gitignored)
└── pyproject.toml
```

Documentation détaillée : [docs/README.md](docs/README.md).

## Dépendances principales

Voir `pyproject.toml` pour les versions exactes.  
Python ≥ 3.11 requis.
