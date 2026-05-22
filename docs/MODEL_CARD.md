# Model Card — HTR Medieval Manuscripts 2026

## Description

Pipeline de reconnaissance de texte manuscrit (HTR) fine-tuné sur des manuscrits médiévaux 
en ancien et moyen français (VIIIe–XVIIe siècle).

## Modèles produits

| Modèle | Architecture | LoRA r | CER val | WER val |
|---|---|---|---|---|
| trocr-lora-r8 | TrOCR-base + LoRA | 8 | À compléter | À compléter |
| trocr-lora-r16 | TrOCR-base + LoRA | 16 | À compléter | À compléter |
| kraken-medieval | Kraken fine-tuné | — | À compléter | À compléter |

## Données d'entraînement

- CATMuS Medieval, CREMMA Médiéval, e-NDP, HIMANIS-Guérin
- Voir [DATA_SOURCES.md](DATA_SOURCES.md) pour les licences et attributions

## Performances (test scellé)

| Métrique | Valeur | Seuil validation | Seuil excellence |
|---|---|---|---|
| CER global | À compléter | < 15% | < 8% |
| WER global | À compléter | < 25% | < 15% |
| IoU segmentation | À compléter | > 0.75 | > 0.85 |

## Limitations

- Performances dégradées sur les écritures gothiques cursives tardives (XVIe–XVIIe s.)
- Sensible aux images très dégradées (taches, déchirures > 20% de la surface)
- Pas évalué sur des manuscrits non latins ou non romans
- Corpus surreprésenté en français de l'Île-de-France (voir section biais de l'article)

## Usage

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("outputs/trocr_lora")
model = VisionEncoderDecoderModel.from_pretrained("outputs/trocr_lora")

image = Image.open("ligne_manuscrit.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Citation

```bibtex
@misc{htr_medieval_2026,
  title={Pipeline HTR pour manuscrits médiévaux},
  author={Groupe HTR HETIC 2026},
  year={2026},
  institution={HETIC},
}
```
