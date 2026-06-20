# HTmiR — Reconnaissance d'écriture manuscrite pour le français médiéval (XIIIe s.)

*[English version](README.md)*

Fine-tuning d'un modèle HTR Kraken pour la transcription automatique de
manuscrits en français du XIIIe siècle (gothique *textualis*), entraîné sur le
corpus CATMuS Medieval. Ce dépôt couvre le volet Vision par ordinateur
(segmentation et reconnaissance) ; le volet NLP de post-traitement est documenté
séparément.

**Modèle (HuggingFace) :** https://huggingface.co/abdessamadtouzani/htmir-french-13c
**Démo en ligne (Streamlit) :** https://htmirproject.streamlit.app
**Article de recherche :** `paper/article.tex` (format IEEE, français)

---

## Résultats

| Métrique | Valeur | IC 95 % (bootstrap, N=1000) |
|---|---|---|
| CER (taux d'erreur caractère, validation) | 4,47 % | [3,7 %, 5,3 %] |
| WER (taux d'erreur mot, validation) | 23,6 % | [21,0 %, 26,2 %] |
| Précision caractère | 95,5 % | — |
| IoU segmentation de lignes (HTRomance, jeu indépendant) | 78,5 % | — |

Cible : CER < 8 %. Méthodologie et discussion complètes dans `paper/`.

---

## 1. Motivation

Le projet visait initialement la transcription des carnets en écriture
spéculaire de Léonard de Vinci. La contrainte bloquante est qu'**aucune vérité
terrain alignée** (paires image/transcription) n'est publiquement disponible
pour ces manuscrits, alors que le fine-tuning HTR supervisé exige précisément de
telles paires. S'y ajoute la publication récente de nouveau matériel de Vinci
(folios de carnets et dessins annotés parus quelques jours avant le début de ce
travail), non encore accompagné de transcriptions exploitables par la machine.

Deux conséquences. D'abord, la vérité terrain de Vinci pourrait réalistiquement
être **produite avec [eScriptorium](https://gitlab.com/scripta/escriptorium)** —
la plateforme open-source d'annotation/transcription bâtie sur Kraken — via une
campagne d'annotation assistée ; c'est un projet en soi, hors du présent
périmètre. Ensuite, plutôt que d'annoter des pages de Vinci à la main dans le
temps imparti, nous construisons et validons d'abord un pipeline HTR complet et
reproductible sur un corpus médiéval richement annoté — le **français du XIIIe
siècle** — dont l'écriture gothique partage avec la main de Vinci la densité, les
abréviations et les ligatures qui font la difficulté du HTR ancien. Le pipeline
est ensuite directement réutilisable pour une future campagne de vérité terrain
et de fine-tuning sur Vinci.

## 2. Données

- **Source :** CATMuS Medieval (HuggingFace `CATMuS/medieval`), filtré sur
  `language = French`, `century = 13`.
- **Volumétrie :** ~21 600 lignes — 19 238 train / 969 validation / 1 369 test.
  Alphabet de 252 symboles (abréviations médiévales préservées).
- **Extraction :** interrogation via DuckDB (predicate pushdown) pour éviter de
  télécharger les ~25 Go de fichiers parquet ; seules les lignes pertinentes sont
  matérialisées.

### Stockage : cloud (S3), aucune copie locale

Les données d'entraînement ne sont **ni versionnées dans ce dépôt, ni conservées
en local**. La source de vérité est **AWS S3**
(`s3://htmir-data/datasets/catmus-french-13c`). Les données sont **téléchargées
au moment de l'entraînement** :

```
lancement de l'entraînement
   -> download_dataset(bucket="htmir-data", prefix="datasets/catmus-french-13c")   [src/htmir/data/s3_sync.py]
   -> si absent de S3 : préparation depuis HuggingFace (CATMuS) puis push S3 (idempotent)
   -> ketos compile + ketos train
```

Cette logique s'exécute dans les deux points d'entrée :
`src/htmir/cli/run_local.py` (GPU local) et
`infrastructure/train_entrypoint.py` (SageMaker).

## 3. Pipeline

```
CATMuS Medieval (HuggingFace)  ->  S3 (source de vérité)
        |
        v
prepare_catmus.py   ->  images de lignes (.png) + transcriptions (.gt.txt)
        |
        v
train_kraken.py     ->  fine-tuning depuis le modèle de base CATMuS Medieval (ketos)
        |
        v
evaluate.py         ->  CER / WER (+ IC bootstrap), IoU de segmentation
```

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # préparation des données + tests
pip install -e ".[train]"    # + Kraken / torch (environnement d'entraînement)
pip install -e ".[viz]"      # + dashboard Streamlit
```

## 5. Utilisation

### Entraînement (GPU local, recommandé)

Une seule commande récupère le dataset (S3, fallback HuggingFace), fine-tune sur
GPU et évalue :

```bash
htmir-train-local --config configs/training.yaml
htmir-train-local --config configs/training.yaml --skip-prepare   # données déjà présentes
htmir-train-local --config configs/training.yaml --upload-model   # push du modèle sur S3
htmir-train-local --config configs/training.yaml --no-s3          # 100 % local
```

Mettre `training.device: cuda:0` dans `configs/training.yaml` pour le GPU.

### Entraînement (SageMaker, sans GPU local)

```bash
python infrastructure/sagemaker_train.py launch --config configs/training.yaml
```
Nécessite un quota GPU `ml.g4dn.xlarge` sur le compte AWS.

### Préparation des données seule

```bash
htmir-prepare --config configs/training.yaml --push-s3
htmir-prepare --config configs/training.yaml --max-files 8 --max-samples 25   # échantillon rapide
```

### Évaluation

```bash
htmir-evaluate --model htmir-french-13c_best.mlmodel \
               --test-arrow data/catmus-french-13c/test.arrow
```

### Dashboard

```bash
streamlit run src/htmir/viz/dashboard.py
```
Également déployé sur https://htmirproject.streamlit.app

## 6. Reproductibilité

- Toutes les exécutions sont pilotées par `configs/training.yaml` (dataset,
  modèle, entraînement, SageMaker).
- Les graines aléatoires sont fixées (ex. rééchantillonnage bootstrap).
- Tests unitaires : `pytest` (préparation des données, construction des commandes
  d'entraînement, métriques d'évaluation, synchronisation S3, dashboard).
- Données et modèle versionnés sur S3.

## 7. Structure du projet

```
src/htmir/
  data/prepare_catmus.py    extraction CATMuS -> images de lignes + .gt.txt
  data/s3_sync.py           dataset <-> S3 (tar par split)
  training/train_kraken.py  ketos compile + ketos train
  eval/evaluate.py          CER / WER, IC bootstrap
  eval/seg_eval.py          IoU de segmentation
  cli/run_local.py          pipeline GPU local (data -> train -> eval)
  viz/dashboard.py          dashboard Streamlit
  utils/logger.py
configs/training.yaml       configuration dataset + modèle + SageMaker
infrastructure/             lanceur SageMaker et point d'entrée du conteneur
paper/                      article de recherche (LaTeX) + figures
tests/                      tests unitaires (pytest)
```

## 8. Article de recherche

Le livrable suit la structure normalisée d'un article de recherche (résumé,
introduction, travaux liés, méthodes, résultats, discussion, conclusion,
bibliographie, annexes). Sources et figures générées automatiquement dans
`paper/` :

```bash
python paper/make_figures.py        # régénère les figures depuis les métriques
# compiler paper/article.tex avec une chaîne LaTeX (ex. Overleaf)
```
