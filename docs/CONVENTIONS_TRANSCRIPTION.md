# Conventions de transcription

## Niveau de transcription

Ce projet adopte une transcription **semi-diplomatique** : fidèle à la lettre du manuscrit, mais avec développement des abréviations signalé par des crochets.

## Abréviations

| Convention | Exemple manuscrit | Transcription |
|---|---|---|
| Développement | q̃ | q[ue] |
| Tilde générique | ē | e[n] ou e[m] selon contexte |
| Suspension | ds | d[eu]s |

## Lacunes et dégradations

- Lacune lisible partielle : `[...]`
- Lacune illisible : `[†]`
- Mot incertain : `{mot?}`

## Casse et ponctuation

- Respect de la casse du manuscrit
- Pas d'ajout de ponctuation absente du manuscrit
- Majuscules initiales conservées telles quelles

## Chiffres et nombres

- Chiffres romains transcrits tels quels : `xii`, `XX`
- Pas de conversion en chiffres arabes

## Langues mélangées

- Latin intercalé signalé par `<lat>texte latin</lat>`
- Rubriques en latin transcrites normalement

## Caractères spéciaux

- ß (eszett médiéval) → `ß` (Unicode conservé)
- Lettre yogh : `ȝ`
- Lettre thorn : `þ`
- Lettre wynn : `ƿ`

## Gestion du needs_review

Une ligne est marquée `needs_review: true` si :
- Confiance du modèle < 0.6
- Longueur < 5 caractères
- Discordance TrOCR/Kraken > 30% CER
- Présence de zones dégradées détectées lors du prétraitement
