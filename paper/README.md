# Article scientifique HTmiR

Article en français (LaTeX) — construit **progressivement** au fil du projet.

## Fichiers

- `article.tex` — l'article (Abstract → Conclusion)
- `make_figures.py` — génère les figures depuis les vraies métriques d'entraînement + le rapport d'éval
- `figures/` — graphiques générés (CER, accuracy, bootstrap CI, IoU segmentation)

## Régénérer les figures

```bash
python paper/make_figures.py
```
Lit `data/catmus-french-13c/training_metrics.csv` (entraînement) et
`data/catmus-french-13c/eval_report.json` (bootstrap CI + IoU), puis écrit les
`.pdf`/`.png` dans `figures/`.

## Format

L'article utilise la classe **IEEEtran** (format 2 colonnes de journal
scientifique). Cette classe est incluse par défaut sur Overleaf — rien à
installer.

## Compiler le PDF

**Option A — Overleaf** (recommandé) : New Project → Upload Project → glisser
`article.tex` + le dossier `figures/` → compiler. IEEEtran est déjà disponible.

**Option B — Local** (distribution TeX installée) :
```bash
cd paper
pdflatex article.tex
```

## État (progressif)

| Section | Statut |
|---|---|
| Introduction (da Vinci → pivot) | ✅ rédigée |
| Données (CATMuS français XIIIe + HTRomance) | ✅ rédigée |
| Méthode (fine-tuning + bootstrap + IoU) | ✅ rédigée |
| Résultats (CER ~4,5 %, figures) | ✅ rédigée |
| Robustesse statistique (bootstrap CI) | ✅ rédigée |
| Segmentation + IoU (HTRomance XIIIe) | ✅ rédigée |
| Évaluation test set | 🔲 à finaliser (bug ketos test Kraken 7) |
| Application Léonard de Vinci | 🔲 perspective |
| Post-traitement NLP | 🔲 prochaine partie |
