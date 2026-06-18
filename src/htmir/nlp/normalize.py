"""Normalisation déterministe des transcriptions (règles d'abord, IA ensuite).

Deux familles **distinctes** de règles :

1. **Normalisation pour le CER** (:func:`normalize_for_cer`) — appliquée des
   DEUX côtés (hypothèse *et* vérité terrain). Elle fusionne des distinctions
   graphiques non significatives (Unicode NFC, ``u/v``, ``i/j``, ``ſ→s``,
   espaces) → fait baisser le CER de façon légitime.

2. **Développement pour la lisibilité** (:func:`expand_abbreviations`) — résout
   les abréviations médiévales (``⁊``→``et``, tilde nasal, table). Produit un
   texte plus lisible pour l'humain, mais qui s'écarte d'une GT *diplomatique*
   (donc à ne PAS utiliser pour le CER contre une telle GT).

Toutes les règles sont documentées dans ``CONVENTIONS_NLP.md``.
"""

import re
import unicodedata

# ── Briques élémentaires ─────────────────────────────────────────────────────

def to_nfc(text: str) -> str:
    """Normalisation Unicode NFC (composition canonique)."""
    return unicodedata.normalize("NFC", text)


def fold_uv(text: str) -> str:
    """Fusionne u/v (interchangeables en graphie médiévale) → ``u``."""
    return text.replace("v", "u").replace("V", "U")


def fold_ij(text: str) -> str:
    """Fusionne i/j → ``i``."""
    return text.replace("j", "i").replace("J", "I")


def fold_long_s(text: str) -> str:
    """s long ``ſ`` → ``s``."""
    return text.replace("ſ", "s")


def collapse_whitespace(text: str) -> str:
    """Réduit les espaces multiples et trim."""
    return re.sub(r"\s+", " ", text).strip()


# Ordre des étapes de normalisation CER (pour mesurer le CER après CHAQUE étape)
NORMALIZATION_STEPS: list[tuple[str, callable]] = [
    ("NFC", to_nfc),
    ("u/v", fold_uv),
    ("i/j", fold_ij),
    ("long_s", fold_long_s),
    ("espaces", collapse_whitespace),
]


def normalize_for_cer(text: str) -> str:
    """Applique toutes les règles CER, dans l'ordre."""
    for _, fn in NORMALIZATION_STEPS:
        text = fn(text)
    return text


def normalize_steps_cumulative(text: str) -> list[tuple[str, str]]:
    """Retourne le texte après chaque étape cumulée.

    Returns:
        Liste ``[("brut", txt), ("NFC", txt), ("+u/v", txt), ...]`` permettant
        de tracer la courbe du CER étape par étape.
    """
    out = [("brut", text)]
    cur = text
    for name, fn in NORMALIZATION_STEPS:
        cur = fn(cur)
        out.append((f"+{name}", cur))
    return out


# ── Développement des abréviations (lisibilité) ──────────────────────────────

# Table minimale, à enrichir selon le corpus (cf. CONVENTIONS_NLP.md).
ABBREVIATION_TABLE: dict[str, str] = {
    "⁊": "et",      # et tironien
    "ꝑ": "per",     # p barré
    "ꝓ": "pro",
    "ꝙ": "qui",
}


def expand_abbreviations(text: str, table: dict[str, str] | None = None) -> str:
    """Développe les abréviations pour la lisibilité (PAS pour le CER).

    - remplace les signes de la table d'abréviations ;
    - résout le tilde nasal combinant (``◌̃`` sur une lettre → +``n``),
      heuristique simple : ``q̃``→``que`` non géré ici (ambigu), on se limite
      au tilde nasal vocalique.

    Args:
        text: Texte source.
        table: Table d'abréviations (défaut :data:`ABBREVIATION_TABLE`).
    """
    table = table or ABBREVIATION_TABLE
    for abbr, full in table.items():
        text = text.replace(abbr, full)

    # tilde nasal combinant (U+0303) : voyelle + ̃ → voyelle + n
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"([aeiou])̃", r"\1n", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFC", text)
    return text


def count_residual_abbreviations(text: str) -> int:
    """Compte les marques d'abréviation résiduelles (diagnostic EDA).

    Compte les signes de la table + les marques combinantes nasales/tilde.
    """
    n = sum(text.count(a) for a in ABBREVIATION_TABLE)
    nfd = unicodedata.normalize("NFD", text)
    n += len(re.findall(r"[̃ͣ-ͯ]", nfd))  # tilde + abrév. en exposant
    return n
