# Conventions NLP — HTmiR

Post-traitement des transcriptions HTR (français médiéval, XIIIᵉ s.). Ce document
justifie les choix de normalisation et de mesure. Code : [src/htmir/nlp/](src/htmir/nlp/).

## 1. Data contract (entrée obligatoire)

Le HTR (Kraken) produit un **ALTO XML** par page. On le convertit en
**data contract JSON** ([data_contract.py](src/htmir/nlp/data_contract.py)),
validé par `jsonschema`. Par ligne :

| Champ | Source ALTO | Usage |
|---|---|---|
| `text` | `String/@CONTENT` (mots joints par espace) | normalisation, CER |
| `polygon` | `TextLine/Shape/Polygon` | localisation, IoU |
| `baseline` | `TextLine/@BASELINE` | segmentation |
| `char_confidences` | `Glyph/@GC` (par glyphe) | correction, `needs_review` |
| `mean_confidence` | moyenne des `GC` | tri qualité |
| `needs_review` | `mean_confidence < 0.70` | filtrage |

Seuil `needs_review` = **0,70** (choisi via l'EDA : les lignes sous ce seuil sont
majoritairement des fausses détections — cf. §4).

## 2. Normalisation pour le CER (règles, des DEUX côtés)

Appliquée à l'hypothèse **et** à la vérité terrain, car elle fusionne des
distinctions graphiques **non significatives** en ancien français. Ordre :

| Étape | Règle | Pourquoi |
|---|---|---|
| `NFC` | composition Unicode canonique | aligne combinant vs précomposé |
| `u/v` | `v→u`, `V→U` | u/v interchangeables au XIIIᵉ |
| `i/j` | `j→i`, `J→I` | i/j interchangeables |
| `long_s` | `ſ→s` | variante positionnelle du s |
| `espaces` | collapse + trim | bruit de segmentation |

> **Choix clé** : notre vérité terrain HTRomance est **diplomatique** (garde
> abréviations, `u` pour `v`, pas d'apostrophes). Développer les abréviations
> *éloignerait* l'hypothèse de cette GT → on **ne le fait pas pour le CER**.
> La normalisation ci-dessus, appliquée des deux côtés, ne peut que *baisser*
> le CER (test garanti dans `test_nlp_normalize.py`).

## 3. Développement des abréviations (lisibilité, séparé)

[normalize.py](src/htmir/nlp/normalize.py) `expand_abbreviations` : `⁊→et`,
tilde nasal vocalique (`ã→an`), table (`ꝑ→per`…). **But : lisibilité humaine**,
pas le CER (produit un texte qui s'écarte de la GT diplomatique). Table à
enrichir selon le corpus.

## 4. Mesure du CER (sans dépendre d'une convention externe)

On rapporte trois variantes, du strict au pertinent :

| Métrique | Définition |
|---|---|
| `cer_raw` | toutes les lignes Kraken vs GT |
| `cer_reviewed` | après filtrage `needs_review` (confiance) |
| `cer_normalized` | règles §2 appliquées des deux côtés |

**Référence** : GT HTRomance (`.chocomufin.xml`). On a donc un **CER absolu**.
Pour un corpus *sans* GT (ex. Léonard de Vinci), on basculerait sur une
**évaluation relative** (distance entre versions successives, cf. outil
Evaluation-HTR).

## 5. Reproductibilité

- Seeds fixés (bootstrap CER : `seed=42`).
- Dépendances versionnées (`pyproject.toml`, extra `nlp`).
- Tests `pytest` : validation du schéma JSON + non-dégradation du CER par la
  normalisation.
- Résultats persistés sur **Supabase** (tables `runs` / `lines`), affichés sur
  le dashboard.
