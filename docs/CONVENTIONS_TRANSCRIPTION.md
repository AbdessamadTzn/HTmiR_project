# Conventions de transcription — Carnets de Léonard de Vinci

## Contexte

Les manuscrits de Léonard de Vinci sont rédigés en **écriture miroir** (de droite à gauche, pour un gaucher).  
Le pipeline normalise d’abord les images en sens de lecture (`normalize_mirror_writing`) avant HTR ; les transcriptions sont en **sens de lecture**, pas en sens miroir du folio brut.

## Niveau de transcription

Transcription **semi-diplomatique** en italien autographe (XVe–XVIe s.) :

- Fidélité à la graphie de Léonard (abréviations, ligatures).
- Développement des abréviations entre crochets : `p[er]`, `ch[osa]`.
- Pas de traduction française/anglaise dans le JSON Volet 1 (réservé au Volet NLP).

## Écriture miroir

| Étape | Convention |
|---|---|
| Image source | Numérisation telle que fournie (Gallica, Ambrosiana, etc.) |
| Image HTR | Retournement horizontal si détection miroir ou `--force-mirror` |
| Transcription | Sens de lecture (gauche → droite) après normalisation |
| Métadonnée | `mirror_normalized: true` dans le manifeste si flip appliqué |

## Abréviations et symboles

| Type | Convention |
|---|---|
| Abréviation | `q[ue]`, `p[er]`, `ch[osa]` |
| Lacune partielle | `[...]` |
| Lacune illisible | `[†]` |
| Mot incertain | `{mot?}` |
| Chiffres / mesures | Transcrits tels quels ; unités entre crochets si ajoutées : `3[braccia]` |

## Mélange texte / dessin

- **Texte seul** : transcription complète ligne par ligne.
- **Zone illustrée sans texte** : pas de ligne HTR ; région exclue ou `type: illustration` en PAGE XML.
- **Annotations dans les dessins** : lignes séparées si baseline détectée.

## Langues

- Principal : italien de la Renaissance (graphies florentines).
- Latin occasionnel : balises `<lat>...</lat>` si présent dans la référence.

## Flag `needs_review`

Une ligne est marquée `needs_review: true` si :

- Confiance du modèle < 0,6
- Longueur < 5 caractères
- Discordance TrOCR / Kraken > 30 % CER (vote Needleman–Wunsch)
- Zone dégradée ou forte inclinaison (> 5°) au prétraitement
- Présence de symboles non alphabétiques dominants (schémas techniques)

## Références éditoriales

- Editions critiques des carnets (ex. corpus Ambrosiana / Institut de France).
- Alignement avec les conventions HTR-United pour l’export PAGE XML et le JSON Volet 2.
