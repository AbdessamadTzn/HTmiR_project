# HTmiR — HTR pour le français médiéval

Fine-tuning d'un modèle HTR (Handwritten Text Recognition) **Kraken** pour la
transcription automatique de manuscrits en **français du XIIIe siècle**
(gothique textualis), à partir du dataset **CATMuS Medieval**.

## Pipeline

```
CATMuS Medieval (HuggingFace)
   │  filtre language=French, century=13   (~75k lignes, ground truth incluse)
   ▼
prepare_catmus.py   →   lignes image (.png) + transcription (.gt.txt)
   ▼
train_kraken.py     →   fine-tuning Kraken depuis le modèle de base CATMuS
   ▼
evaluate.py         →   CER / WER sur le test set
```

L'entraînement s'exécute sur **AWS SageMaker** (GPU), orchestré par
`infrastructure/sagemaker_train.py`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # préparation données + tests
pip install -e ".[train]"        # + Kraken/torch (conteneur d'entraînement)
```

## Utilisation

### Entraînement en local (machine avec GPU) — recommandé

Une seule commande fait tout : récupère le dataset (S3 ou HuggingFace),
fine-tune sur GPU, évalue (CER/WER), et peut uploader le modèle sur S3.

```bash
pip install -e ".[train]"          # Kraken + torch (env propre, pas de conflit)

# Pipeline complet (data → train GPU → eval)
htmir-train-local --config configs/training.yaml

# Variantes
htmir-train-local --config configs/training.yaml --skip-prepare   # data déjà là
htmir-train-local --config configs/training.yaml --upload-model   # push modèle S3
htmir-train-local --config configs/training.yaml --no-s3          # 100 % local
```

Dans `configs/training.yaml`, mettre `training.device: cuda:0` pour le GPU.

### Étapes séparées

```bash
# Préparer les données seules (+ push S3)
htmir-prepare --config configs/training.yaml --push-s3

# Test rapide sur un petit échantillon
htmir-prepare --config configs/training.yaml --max-files 8 --max-samples 25
```

### Entraînement sur SageMaker (si pas de GPU local)

```bash
python infrastructure/sagemaker_train.py launch --config configs/training.yaml
```
> Nécessite un quota GPU `ml.g4dn.xlarge` sur le compte AWS.

## Tests

```bash
pytest
```

## Structure

```
src/htmir/
├── data/prepare_catmus.py   # extraction CATMuS → lignes image+gt.txt
├── data/s3_sync.py          # dataset ⇄ S3 (tar par split)
├── training/train_kraken.py # ketos compile + ketos train
├── eval/evaluate.py         # métriques CER/WER
├── cli/run_local.py         # pipeline local GPU (data → train → eval)
├── viz/dashboard.py         # dashboard Streamlit
└── utils/logger.py
configs/training.yaml        # config dataset + modèle + SageMaker
infrastructure/              # lanceur SageMaker (alternative au local)
tests/                       # tests unitaires (pytest)
```
