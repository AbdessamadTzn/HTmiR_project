# HTmiR — Soutenance (10 slides)

> HTR pour le français médiéval (XIIIᵉ s.) — Computer Vision + NLP
> Graphes prêts dans `paper/figures/` (`.pdf` et `.png`). Les **[TES IMAGES]**
> = captures que tu ajoutes toi-même (manuscrit + sortie modèle).

---

## Slide 1 — Titre & contexte
**HTmiR : reconnaissance de l'écriture manuscrite, français médiéval XIIIᵉ**
- Master Recherche HETIC — Abdessamad Touzani & Emmanuel
- **Objectif long terme** : transcrire les carnets en écriture spéculaire de
  Léonard de Vinci.
- **Verrou** : aucune vérité terrain publique pour Vinci → on ne peut pas
  fine-tuner directement.
- **Démarche** : valider d'abord un pipeline HTR complet sur un corpus médiéval
  richement annoté — le **français du XIIIᵉ (CATMuS)** — réutilisable ensuite.

*(pas de graphe)*

---

## Slide 2 — Pipeline global (vue d'ensemble)
Schéma en une ligne :
```
CATMuS → préparation → fine-tuning Kraken → évaluation (CER/WER + IoU)
                                                   ↓
                                        data contract JSON
                                                   ↓
                              NLP : normalisation → correction → CER avant/après
```
- **CV** : segmentation + reconnaissance
- **NLP** : post-traitement de la sortie HTR
- Tableau de bord Streamlit + stockage Supabase

*(pas de graphe, ou un schéma maison)*

---

## Slide 3 — Données (CATMuS Medieval)
- Source : HuggingFace `CATMuS/medieval`, filtré **français + siècle 13**
- Extraction efficace via **DuckDB** (predicate pushdown — évite 25 Go de parquet)
- **21 576 lignes** annotées :
  - train **19 238** · validation **969** · test **1 369**
- Alphabet de **252 symboles** (abréviations médiévales préservées)

**[GRAPHIQUE]** : onglet *Dataset* du dashboard (répartition des splits +
distribution des longueurs de lignes) — capture d'écran.

---

## Slide 4 — Méthode (fine-tuning Kraken)
- Modèle de base : **CATMuS Medieval** (Kraken), fine-tuné via `ketos`
- `--resize union` : fusionne l'alphabet de base + caractères du corpus
- Hyperparamètres :
  | Paramètre | Valeur |
  |---|---|
  | Learning rate | 1×10⁻⁴ |
  | Batch | 8 |
  | Epochs (max) | 50 |
  | Early stopping (patience) | 10 |
  | Matériel | GPU RTX 4060 |
- Transfert d'apprentissage → bonnes perfs avec peu de données

*(pas de graphe, la table suffit)*

---

## Slide 5 — Résultats : reconnaissance ⭐ (cœur CV)
- **CER validation ≈ 4,47 %** (accuracy caractère 95,5 %) — **sous la cible 8 %**
- WER ≈ 27,6 % (accuracy mot 72,4 %)
- Convergence rapide, early stopping (meilleur epoch = 4)

**[GRAPHIQUE]** : `figures/cer_par_epoch.pdf` (CER par epoch + cible 8 %)
**[GRAPHIQUE]** : `figures/accuracy_par_epoch.pdf` (accuracy caractère vs mot)

---

## Slide 6 — Robustesse statistique (bootstrap)
- Intervalles de confiance à 95 % par **bootstrap (N = 1000)** :
  - **CER : [3,7 % ; 5,3 %]**
  - WER : [25,1 % ; 30,2 %]
- La borne haute du CER reste **sous la cible 8 %** → résultat fiable

**[GRAPHIQUE]** : `figures/bootstrap_ci.pdf` (forest plot CER/WER)

---

## Slide 7 — Segmentation de pages (IoU) ⭐
- Évalué sur un corpus **indépendant** : HTRomance (manuscrits français XIIIᵉ)
- **IoU moyen 78,5 %**
  | Manuscrit | IoU moyen | Lignes ≥ 0,75 |
  |---|---|---|
  | Vie de saint Alexis (prose, 1 col.) | **85,4 %** | 92,7 % |
  | Roman de Troie (2 colonnes) | 71,7 % | 38,1 % |
- Seuil 0,75 atteint sur les mises en page à **colonne unique**

**[GRAPHIQUE]** : `figures/iou_segmentation.pdf` (barres IoU par manuscrit)

---

## Slide 8 — Démo : le modèle en action ⭐
- Pipeline : image de page → `kraken segment + ocr` → **ALTO XML** (texte +
  confiance par caractère + polygones)

**[TES IMAGES]** :
1. l'**image du manuscrit** que tu testes
2. la **sortie texte** du modèle (transcription)
3. côte à côte : **vérité terrain vs sortie modèle** (montre que ça lit bien)

> Astuce : prends une page propre à colonne unique (meilleur rendu).

---

## Slide 9 — Partie NLP (post-traitement)
- **Data contract JSON** (entrée obligatoire) : texte + `char_confidences` +
  polygone + `needs_review`, validé par `jsonschema`
- **CER avant/après** sur HTRomance (5 pages) :
  - brut **6,40 %** → après filtrage `needs_review` **6,28 %**
  - normalisation (u/v, i/j, NFC) appliquée des 2 côtés
- **Correction** : lexique ancien français **LGeRM (80k formes)** propose des
  candidats ; **D'AlemBERT** (modèle médiéval) choisit selon le contexte
  - ex. `eusemble → ensemble` ; D'AlemBERT 5× plus confiant que CamemBERT

**[TES IMAGES]** : exemples d'**abréviations** (`⁊`→et, tilde nasal) et de
correction sur une ligne.

---

## Slide 10 — Conclusion & perspectives
**Acquis**
- Pipeline HTR complet, reproductible (**110 tests**, dashboard, Supabase)
- Reconnaissance CER ≈ 4,5 % · Segmentation IoU ≈ 78,5 %
- Volet NLP : data contract + normalisation + correction (sans entraînement)

**Limites (honnêtes)**
- Mises en page multi-colonnes plus dures (IoU)
- Sur un HTR déjà bon, la correction apporte peu

**Perspectives**
- Vérité terrain pour Léonard de Vinci (retournement spéculaire + annotation)
- Lexique enrichi, modèle de langue médiéval dédié

*(pas de graphe)*

---

### Récap des chiffres clés (anti-sèche)
| Métrique | Valeur |
|---|---|
| Lignes dataset | 21 576 (19 238 / 969 / 1 369) |
| CER validation | **4,47 %** (IC95 [3,7 ; 5,3]) |
| WER validation | 27,6 % (IC95 [25,1 ; 30,2]) |
| Accuracy caractère | 95,5 % |
| IoU segmentation (moy.) | **78,5 %** |
| IoU prose / 2-colonnes | 85,4 % / 71,7 % |
| CER NLP (brut→reviewed) | 6,40 % → 6,28 % |
| Lexique ancien français | 80 012 formes (LGeRM) |
| Tests | 110 ✅ |

### Graphes disponibles (`paper/figures/`)
- `cer_par_epoch.pdf` — slide 5
- `accuracy_par_epoch.pdf` — slide 5
- `bootstrap_ci.pdf` — slide 6
- `iou_segmentation.pdf` — slide 7
