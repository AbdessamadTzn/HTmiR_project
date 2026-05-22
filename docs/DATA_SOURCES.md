# Sources de données — Manuscrits Vinci (écriture inversée)

## Corpus principal (évaluation & fine-tuning cible)

| Ressource | Source | Licence | Contenu |
|---|---|---|---|
| Codex Atlanticus (sélection) | [Biblioteca Ambrosiana](https://www.codex-atlanticus.it/) / numérisations publiques | Vérifier par folio | Carnets techniques, écriture miroir |
| Manuscrits Vinci — Gallica BnF | [gallica.bnf.fr](https://gallica.bnf.fr) | Gallica (recherche non commerciale) | Folios numérisés, manuscrits de Léonard et école |
| Leonardo da Vinci — notebooks (sélection Zenodo / IIIF) | Zenodo, institutions partenaires | CC-BY selon dépôt | Pages haute résolution |

**Important :** documenter chaque folio utilisé (ID, URL, date de téléchargement) dans `data/raw/manifest_sources.csv`.

## Corpus de transfert (pré-entraînement HTR)

Pour compenser le faible volume de transcriptions Vinci alignées, fine-tuning initial sur manuscrits proches :

| Dataset | Source | Licence | Usage |
|---|---|---|---|
| CATMuS Medieval | huggingface.co/datasets/CATMuS/medieval | CC-BY 4.0 | Pré-adaptation graphies anciennes |
| CREMMA Médiéval | github.com/HTR-United/cremma-medieval | CC-BY 4.0 | Italien / latin médiéval manuscrit |
| GalliCorpora (sous-ensemble italien) | HTR-United | CC-BY 4.0 | Renforcement si disponible |

## Modèles pré-entraînés

| Modèle | Source | Licence | Usage Vinci |
|---|---|---|---|
| microsoft/trocr-base-handwritten | HuggingFace | MIT | Baseline + LoRA |
| Kraken (BLLA + modèle latin) | HTR-United / kraken | Apache 2.0 | Segmentation lignes, 2e hypothèse HTR |
| facebook/sam-vit-base | HuggingFace | Apache 2.0 | Segmentation layout (zones texte vs dessin) |

## Spécificités du corpus Vinci

- **Écriture miroir** : normalisation obligatoire avant entraînement (voir `htmir.preprocessing.mirror`).
- **Mélange texte / schémas** : segmentation layout pour isoler les bandes de texte.
- **Encre brune / papier ivoire** : CLAHE + Sauvola paramétrés dans `configs/preprocessing_vinci.yaml`.
- **Biais** : surreprésentation de mains droites en écriture miroir, italien technique, XVe–XVIe s. — voir section éthique de l’article.

## Attribution

Citer au minimum :

- HTR-United / CREMMA / CATMuS pour les données de transfert
- Institution de conservation pour les images Vinci (Ambrosiana, BnF, etc.)
- Pinche et al. (2022) CATMuS ; Clérice et al. (2021) CREMMA
