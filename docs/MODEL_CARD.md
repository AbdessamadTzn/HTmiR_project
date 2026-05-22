# Model Card — HTR Carnets Léonard de Vinci (écriture miroir)

## Description

Pipeline de reconnaissance de texte manuscrit (HTR) pour les **carnets de Léonard de Vinci en écriture inversée**, dans le cadre du Volet 1 Vision (HETIC MD5 2026).  
Les images sont normalisées en sens de lecture avant inférence ; la sortie JSON alimente le Volet 2 NLP.

## Modèles produits

| Modèle | Architecture | Adaptation | CER val | WER val |
|---|---|---|---|---|
| trocr-baseline | TrOCR-base-handwritten | Aucun | À compléter | À compléter |
| trocr-lora-r8-vinci | TrOCR + LoRA r=8 | Fine-tuning Vinci + transfert | À compléter | À compléter |
| kraken-vinci | Kraken fine-tuné | Lignes + HTR | À compléter | À compléter |
| ensemble-nw | TrOCR + Kraken + Needleman–Wunsch | Vote pondéré | À compléter | À compléter |

## Données

- **Cible :** folios Vinci (écriture miroir, italien autographe).
- **Transfert :** CATMuS Medieval, CREMMA — voir [DATA_SOURCES.md](DATA_SOURCES.md).
- **Test scellé :** hash SHA-256 dans `data/processed/test_set.sha256` — ne pas consulter avant rendu final.

## Performances (objectifs brief)

| Métrique | Validation | Excellence |
|---|---|---|
| CER global | < 15 % | < 8 % |
| WER global | < 25 % | < 15 % |
| IoU segmentation | > 0,75 | > 0,85 |

## Limitations et biais

- Modèles pré-entraînés sur graphies **non miroir** : dépendance forte à `normalize_mirror_writing`.
- Faible volume de transcriptions Vinci alignées vs corpus de transfert médiéval.
- Confusion fréquente entre **lettres et symboles techniques** (géométrie, hydraulique).
- Dessins sans texte : risque de fausses détections de lignes.
- Pas évalué sur d’autres auteurs en écriture miroir (contemporains rares).
- **Représentativité :** homme, Italie, XVIe s., carnets techniques — généralisation limitée.

## Usage

```bash
# Pipeline complet (après placement des images dans data/raw/)
htmir-pipeline --input data/raw --output-root .

# Évaluation sur prédictions
htmir-evaluate --manifest data/processed/manifest.json --predictions predictions.jsonl
```

```python
from htmir.htr.baseline import TrOCRBaseline
text, conf = TrOCRBaseline().transcribe_image("data/lines/folio_001_line_001.png")
```

## Citation

```bibtex
@misc{htr_vinci_2026,
  title={Pipeline HTR pour carnets Léonard de Vinci (écriture miroir)},
  author={Groupe HTR HETIC 2026},
  year={2026},
  institution={HETIC},
}
```
