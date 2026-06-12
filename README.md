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

```bash
# 1. Préparer les données (filtre French/13e, extraction lignes)
htmir-prepare --config configs/training.yaml

# Test rapide sur un petit échantillon
htmir-prepare --config configs/training.yaml --max-files 8 --max-samples 25

# 2. Lancer l'entraînement sur SageMaker
python infrastructure/sagemaker_train.py launch --config configs/training.yaml
```

## Tests

```bash
pytest
```

## Structure

```
src/htmir/
├── data/prepare_catmus.py   # extraction CATMuS → lignes image+gt.txt
├── training/train_kraken.py # ketos compile + ketos train
├── eval/evaluate.py         # métriques CER/WER
└── utils/logger.py
configs/training.yaml        # config dataset + modèle + SageMaker
infrastructure/              # lanceur SageMaker
tests/                       # tests unitaires (pytest)
```
