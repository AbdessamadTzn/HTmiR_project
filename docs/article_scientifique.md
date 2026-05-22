# Traitement automatique de manuscrits médiévaux : pipeline HTR end-to-end avec TrOCR et Kraken

**Groupe HTR HETIC 2026** — Master Data/IA, Module Vision par ordinateur  
Projet MD5 — 2026

---

## Résumé (Abstract)

Ce travail présente un pipeline complet de reconnaissance de texte manuscrit (HTR) appliqué à des manuscrits médiévaux en ancien et moyen français (VIIIe–XVIIe siècle). À partir de corpus librement accessibles (CATMuS Medieval, CREMMA Médiéval, e-NDP, HIMANIS-Guérin), nous implémentons une chaîne de traitement reproductible couvrant le prétraitement d'images (deskew, CLAHE, binarisation Sauvola), la segmentation de mise en page (SAM) et de lignes (Kraken BLLA), puis le fine-tuning comparatif de TrOCR avec adaptateurs LoRA (r=8, r=16) et de Kraken. Les transcriptions sont agrégées par alignement Needleman-Wunsch et livrées dans un format JSON validé par un data contract explicite, incluant les polygones de segmentation en PAGE XML. Les performances sur le jeu de test scellé atteignent CER=[À compléter]% et WER=[À compléter]%, avec des intervalles de confiance bootstrap à 95% reportés. Nous discutons des biais de représentation du corpus et des perspectives d'amélioration pour le Volet 2 (analyse linguistique NLP).

**Mots-clés** : HTR, manuscrits médiévaux, TrOCR, LoRA, Kraken, CER, humanités numériques

---

## 1. Introduction

La numérisation massive des collections patrimoniales constitue une avancée considérable pour les chercheurs en humanités. La Bibliothèque nationale de France recense environ 380 000 manuscrits dont environ 11 millions de documents accessibles via Gallica. Cependant, une image numérisée n'est pas un texte : la transcription manuelle par des paléographes reste trop lente et coûteuse pour passer à l'échelle (environ 50 €/h, pour un marché mondial estimé à 31 Md$).

La reconnaissance automatique de l'écriture manuscrite (HTR) répond à ce verrou. Des projets comme CREMMA, CATMuS ou CoMMA ont démontré la faisabilité d'une transcription automatique de qualité sur des corpus normalisés. Malgré ces avancées, l'obtention de données d'entraînement annotées reste le principal goulot d'étranglement, en particulier pour des écritures dialectales ou très individualisées.

Ce projet s'inscrit dans ce contexte. Ses **contributions principales** sont :
1. Un pipeline HTR end-to-end reproductible, du scan brut au JSON livrable pour le NLP.
2. Une comparaison contrôlée de TrOCR+LoRA et Kraken sur corpus médiéval normalisé.
3. Un export systématique des polygones de segmentation en PAGE XML, favorisant la réutilisation communautaire.
4. Une analyse critique des biais de représentation du corpus constitué.

---

## 2. État de l'art

### 2.1 Architectures HTR

**Kraken** est un moteur HTR open-source basé sur des réseaux récurrents (LSTM) combinés à une analyse de lignes de base (BLLA). Il constitue le backend d'eScriptorium et dispose de nombreux modèles pré-entraînés sur HTR-United pour les écritures historiques.

**TrOCR** (Microsoft, 2021) reformule l'HTR comme un problème de vision-langage : un encodeur Vision Transformer (ViT) extrait les représentations visuelles, un décodeur Transformer génère la transcription token par token. Ce paradigme surpasse les approches CNN-LSTM sur de nombreux benchmarks.

**LoRA** (Hu et al., 2022) — Low-Rank Adaptation — permet de fine-tuner efficacement de grands modèles en n'entraînant qu'un petit nombre de paramètres additionnels (matrices de rang r), réduisant drastiquement les coûts computationnels tout en préservant les performances.

**SAM** (Kirillov et al., 2023) — Segment Anything Model — fournit une segmentation sémantique généraliste applicable à la détection des zones texte/illustration/marge dans les pages de manuscrits.

### 2.2 Jeux de données médiévaux

**CREMMA Médiéval** rassemble quinze manuscrits en ancien français (XIIIe–XVe s.), produits avec eScriptorium/Kraken, sous licence CC-BY.

**CATMuS Medieval** (Consistent Approach to Transcribing ManuScript) fédère les corpus CREMMA, GalliCorpora, HTRomance et DEEDS pour constituer plus de 160 000 lignes sur 200 manuscrits (VIIIe–XVIIe s., dix langues), disponible sur HuggingFace.

**e-NDP** et **HIMANIS-Guérin** complètent avec des registres administratifs parisiens et royaux du XIVe siècle, représentant des écritures cursives chancelleresques.

### 2.3 Métriques d'évaluation

Le **CER** (Character Error Rate) mesure la distance de Levenshtein normalisée par la longueur de la référence. Le **WER** (Word Error Rate) opère au niveau des mots. L'**IAA** (Inter-Annotator Agreement) mesure la variabilité humaine comme plafond théorique du modèle. Les **intervalles de confiance bootstrap** (N=1000) quantifient l'incertitude de ces estimations.

---

## 3. Données

### 3.1 Corpus constitué

| Dataset | Période | Lignes | Type d'écriture | Licence |
|---|---|---|---|---|
| CATMuS Medieval | VIIIe–XVIIe s. | ~120 000 | Gothique, caroline, cursive | CC-BY 4.0 |
| CREMMA Médiéval | XIIIe–XVe s. | ~18 000 | Gothique textura/cursive | CC-BY 4.0 |
| e-NDP | XIVe–XVe s. | ~12 000 | Cursive notariale | CC-BY 4.0 |
| HIMANIS-Guérin | XIVe s. | ~8 000 | Gothique chancelleresque | CC-BY 4.0 |
| **Total** | | **~158 000** | | |

### 3.2 Conventions de transcription

Transcription semi-diplomatique : fidèle à la lettre, abréviations développées entre crochets `[ue]`, lacunes signalées `[†]`. Détail complet dans `CONVENTIONS_TRANSCRIPTION.md`.

### 3.3 Constitution des splits

Le split train/val/test est effectué **avant tout développement**, à l'échelle des manuscrits entiers (pas des lignes) pour éviter la contamination. Ratio : 80/10/10. Le test set est scellé par son hash SHA-256 dès le jour 1.

| Split | Manuscrits | Lignes | Hash SHA-256 |
|---|---|---|---|
| Train | ~160 | ~126 000 | À compléter |
| Val | ~20 | ~16 000 | À compléter |
| Test | ~20 | ~16 000 | À compléter |

---

## 4. Méthodes

### 4.1 Prétraitement des images

La chaîne de prétraitement, implémentée dans `src/preprocessing/`, s'applique dans cet ordre :

1. **Correction d'inclinaison (deskew)** : détection de l'angle via transformée de Hough sur les contours Canny, rotation par matrice affine.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) : clip_limit=2.0, grille 8×8. Améliore la lisibilité sans surexposer les zones enluminées.
3. **Binarisation Sauvola** : fenêtre=25, k=0.2, r=128. Robuste aux variations locales d'illumination caractéristiques des parchemins vieillis.

La pipeline est paramétrable via `PreprocessingConfig` et entièrement reproductible (seed fixé, versions figées dans `pyproject.toml`).

### 4.2 Segmentation

**Layout** : SAM (facebook/sam-vit-base) détecte les zones texte, illustration et marge. Les pages multi-colonnes et les registres tabulaires font l'objet d'un post-traitement spécifique pour garantir l'ordre de lecture.

**Lignes** : Kraken BLLA extrait les lignes de base et les polygones englobants. Les polygones sont exportés immédiatement en PAGE XML (conforme au schéma 2019-07-15) dans `segmentations/`.

**IoU de segmentation** : évalué sur 50 lignes de référence fournies par l'équipe pédagogique. Objectif : IoU > 0.75.

### 4.3 Modèles HTR

#### Baseline zéro fine-tuning

TrOCR (`microsoft/trocr-base-handwritten`) et Kraken (modèle médiéval HTR-United) sont évalués sans fine-tuning pour établir la ligne de base.

#### TrOCR + LoRA

Fine-tuning de `microsoft/trocr-base-handwritten` avec adaptateurs LoRA (cibles : `q_proj`, `v_proj` du décodeur). Deux configurations testées : r=8 (léger) et r=16 (plus expressif). Arrêt prématuré sur la CER de validation.

| Hyperparamètre | Valeur |
|---|---|
| LoRA r | 8 / 16 |
| lora_alpha | 16 |
| lora_dropout | 0.1 |
| learning_rate | 5e-5 |
| batch_size | 8 |
| Optimiseur | AdamW |
| Arrêt prématuré | patience=3 |

#### Kraken (ketos train)

Fine-tuning d'un modèle Kraken médiéval existant (HTR-United) via `ketos train`. Format d'entrée : PAGE XML produit à l'étape de segmentation.

### 4.4 Agrégation et data contract

Quand les deux pipelines sont opérationnels, un vote pondéré par alignement Needleman-Wunsch combine les hypothèses TrOCR et Kraken, pondérées par leurs scores de confiance respectifs.

Une ligne est marquée `needs_review: true` si : confiance < 0.6, longueur < 5 caractères, ou discordance inter-modèles > 30% CER.

La sortie est validée par le schéma JSON du data contract (voir `src/aggregation/data_contract.py`), qui inclut obligatoirement les polygones de segmentation pour chaque ligne.

---

## 5. Résultats

*(Section à compléter après les expériences)*

### 5.1 Performances sur le test scellé

| Modèle | CER | WER | IC bootstrap 95% |
|---|---|---|---|
| TrOCR baseline | À compléter | À compléter | À compléter |
| TrOCR + LoRA r=8 | À compléter | À compléter | À compléter |
| TrOCR + LoRA r=16 | À compléter | À compléter | À compléter |
| Kraken baseline | À compléter | À compléter | À compléter |
| Kraken fine-tuné | À compléter | À compléter | À compléter |
| Agrégation NW | À compléter | À compléter | À compléter |

### 5.2 Test de McNemar (TrOCR vs Kraken)

χ²=[À compléter], p=[À compléter], significatif=[À compléter]

### 5.3 Segmentation

IoU moyen sur les 50 lignes de référence : [À compléter]  
Taux needs_review : [À compléter]%

### 5.4 Ablation

*(Courbes d'apprentissage et ablation des composants de prétraitement à ajouter)*

---

## 6. Discussion

### 6.1 Biais de représentation du corpus

Le corpus présente plusieurs déséquilibres identifiés :

- **Géographique** : surreprésentation de l'Île-de-France (scripta francienne) au détriment des dialectes périphériques (picard, normand, occitan).
- **Temporel** : les XIIIe–XIVe siècles représentent la majorité des lignes ; le VIIIe–XIIe et le XVIe–XVIIe sont sous-représentés.
- **Type documentaire** : les textes littéraires sont surreprésentés par rapport aux documents administratifs et notariaux.
- **Profil de copiste** : biais vers des écritures soignées (scriptorium), au détriment des écritures cursives individualisées.

Ces biais impliquent des performances dégradées sur les types d'écritures sous-représentés, à signaler explicitement lors du déploiement du pipeline.

### 6.2 Limitations techniques

- La segmentation SAM peut échouer sur les pages à forte densité d'enluminures.
- TrOCR est sensible aux images de faible résolution (< 100 dpi).
- La correction Sauvola peut sur-binariser les zones d'encre très pâle.
- Le vote Needleman-Wunsch suppose deux modèles indépendants, ce qui n'est pas strictement vérifié (même corpus d'entraînement).

### 6.3 Perspectives d'amélioration

- Augmentation de données : déformations élastiques, variations de contraste, bruitage simulé.
- Intégration d'un modèle de langage pour la correction post-HTR.
- Extension à d'autres langues romanes médiévales (occitan, catalan, galicien).

---

## 7. Conclusion et travaux futurs

Nous avons présenté un pipeline HTR complet, reproductible et documenté pour les manuscrits médiévaux en ancien/moyen français. Le pipeline couvre l'intégralité de la chaîne depuis l'image brute jusqu'au JSON livrable pour le Volet 2 (analyse NLP), en passant par le prétraitement, la segmentation, le fine-tuning et l'évaluation rigoureuse.

Le **Volet 2** prendra en entrée les transcriptions produites ici pour réaliser : normalisation orthographique, annotation morpho-syntaxique, extraction d'entités nommées (personnes, lieux, dates) et modélisation thématique. La qualité du présent pipeline conditionne directement la pertinence de ces analyses.

---

## Références

Clérice, T. et al. (2021). CREMMA Medieval. HTR-United. https://github.com/HTR-United/cremma-medieval

Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*.

Li, M. et al. (2021). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. *arXiv:2109.10282*.

Pinche, A. et al. (2022). CATMuS Medieval. Zenodo. https://doi.org/10.5281/zenodo.catmus

Kiessling, B. (2019). Kraken — an Universal Text Recognizer for the Humanities. *DH 2019*.

Leifert, G. et al. (2016). Cells in Multidimensional Recurrent Neural Networks. *arXiv:1312.4314*.

---

## Annexes

### A. Exemples de transcriptions

*(Captures d'écran image + résultat HTR à ajouter)*

### B. Schéma du data contract

Voir `src/aggregation/data_contract.py` — `DATA_CONTRACT_SCHEMA`.

### C. Courbes de calibration des scores de confiance

*(À ajouter après les expériences)*
