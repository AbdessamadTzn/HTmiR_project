# HTmiR — HTR pour les carnets de Léonard de Vinci (écriture miroir)

Pipeline de **Handwritten Text Recognition** pour les manuscrits de Léonard de Vinci en **écriture inversée**, Volet 1 du module Vision par ordinateur — **Master Data/IA, HETIC MD5 2026**.

Le JSON produit (`dataset_nlp/`) alimente le Volet 2 NLP (normalisation, NER, etc.).

## Équipe

| Membre | Rôle |
|---|---|
| À compléter | Responsable technique (pipeline, modèles) |
| À compléter | Responsable données (corpus Vinci, licences) |
| À compléter | Responsable expérimentation (CER/WER, journal) |
| À compléter | Responsable documentation (article, README) |

## Spécificités Vinci

- **Écriture miroir** : normalisation automatique (`htmir.preprocessing.mirror`) avant HTR.
- **Pages mixtes** texte + dessins : segmentation layout + lignes.
- **Transfert** depuis CATMuS / CREMMA puis fine-tuning sur folios Vinci alignés.

## Résultats (à compléter)

| Métrique | Valeur | Seuil validation |
|---|---|---|
| CER global | — | < 15 % |
| WER global | — | < 25 % |
| IoU segmentation | — | > 0,75 |

Hash test scellé : `data/processed/test_set.sha256`

## Installation

```bash
git clone <url-du-depot>
cd HTmiR_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Pipeline end-to-end

```bash
# 1. Placer les folios numérisés dans data/raw/

# 2. Préparer le corpus (annotations .txt + images, ou manifeste)
htmir-prepare-corpus --config configs/corpus_vinci.yaml

# 3. Prétraitement (miroir + deskew + CLAHE + Sauvola)
htmir-preprocess --input data/raw --output data/preprocessed/

# 4. Segmentation seule + PAGE XML
htmir-segment --input data/preprocessed --output segmentations

# 5. Pipeline complet → segmentations/ + dataset_nlp/vinci_output.json
htmir-pipeline --input data/raw

# 6. Fine-tuning TrOCR + LoRA (après manifeste prêt)
# python -m htmir.htr.finetune_trocr ...

# 7. Évaluation (test scellé — une seule fois avant rendu)
htmir-evaluate --manifest data/processed/manifest.json --predictions predictions.jsonl

# 8. Tests
pytest tests/ -v
```

## Structure

```
HTmiR_project/
├── src/htmir/
│   ├── preprocessing/   # miroir, deskew, CLAHE, Sauvola
│   ├── segmentation/    # layout, lignes, PAGE XML
│   ├── htr/             # TrOCR LoRA, Kraken, baseline
│   ├── aggregation/     # NW, confiance, data contract JSON
│   ├── evaluation/      # CER, WER, bootstrap, McNemar
│   ├── corpus/          # manifeste, splits, hash test
│   ├── pipeline/        # orchestration end-to-end
│   └── cli/             # commandes htmir-*
├── configs/             # corpus_vinci.yaml, trocr_lora_r8.yaml, …
├── docs/                # conventions, sources, model card, article
├── data/
├── segmentations/
├── dataset_nlp/
├── experiments/journal.jsonl
└── tests/
```

Documentation : [docs/README.md](docs/README.md)

## Reproductibilité

- Seed : `42` (`htmir.utils.seeds`)
- Dépendances : `pyproject.toml`
- Journal d’expériences : `experiments/journal.jsonl`
- Test scellé : ne pas ouvrir avant évaluation finale

## Licence & éthique

Corpus sous licences documentées dans [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).  
Section biais / représentativité : [docs/MODEL_CARD.md](docs/MODEL_CARD.md) et article scientifique.
