"""Correction guidée par confiance/lexique (sans entraînement).

Principe (cf. ``CONVENTIONS_NLP.md``) :

1. un mot **non attesté** dans le lexique est suspect ;
2. on génère des **candidats** proches (distance d'édition, cf.
   :mod:`htmir.nlp.lexicon`) ;
3. on **choisit** :
   - soit le candidat le plus proche (``scorer=None``, rapide, lexique seul) ;
   - soit le candidat le mieux noté dans le **contexte** par un modèle de
     langue masqué (``scorer`` issu de :func:`make_mlm_scorer`, ex. D'AlemBERT).

Aucun entraînement : le MLM est utilisé figé. ``transformers`` est une
dépendance **optionnelle** (seulement pour la sélection contextuelle).
"""

import re
from collections.abc import Callable

from htmir.nlp.lexicon import candidates as lexicon_candidates
from htmir.nlp.lexicon import is_known

# scorer(words, idx, candidates) -> meilleur candidat (ou None)
Scorer = Callable[[list[str], int, list[str]], "str | None"]

_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _split_affixes(token: str) -> tuple[str, str, str]:
    """Sépare ponctuation de début/fin du cœur du mot."""
    m = _AFFIX.match(token)
    return m.group(1), m.group(2), m.group(3)


def correct_line(
    text: str,
    lexicon: set[str],
    scorer: Scorer | None = None,
    max_distance: int = 2,
) -> tuple[str, list[dict]]:
    """Corrige une ligne : mots non attestés → meilleur candidat du lexique.

    Args:
        text: Texte de la ligne.
        lexicon: Lexique (ensemble de formes attestées).
        scorer: Sélecteur contextuel optionnel (sinon : candidat le plus proche).
        max_distance: Distance d'édition maximale pour les candidats.

    Returns:
        ``(texte_corrigé, corrections)`` où chaque correction est un dict
        ``{index, original, corrected, candidates}``.
    """
    tokens = text.split()
    out = list(tokens)
    corrections: list[dict] = []

    for i, tok in enumerate(tokens):
        prefix, core, suffix = _split_affixes(tok)
        if not core or is_known(core, lexicon):
            continue
        cands = lexicon_candidates(core, lexicon, max_distance=max_distance)
        if not cands:
            continue
        best = (scorer(tokens, i, cands) if scorer else None) or cands[0]
        if best.lower() != core.lower():
            out[i] = f"{prefix}{best}{suffix}"
            corrections.append({
                "index": i,
                "original": core,
                "corrected": best,
                "candidates": cands,
            })

    return " ".join(out), corrections


def make_mlm_scorer(model_id: str = "pjox/dalembert") -> Scorer:
    """Construit un sélecteur contextuel via un modèle de langue masqué.

    Note le candidat à la position masquée : on garde celui de plus forte
    probabilité **parmi les candidats** (scoring contraint). Ne traite que les
    candidats tokenisés en un seul token.

    Args:
        model_id: Modèle HuggingFace MLM (défaut : D'AlemBERT, français pré-moderne).

    Returns:
        Une fonction ``scorer(words, idx, candidates) -> str | None``.
    """
    import torch  # dépendance optionnelle
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForMaskedLM.from_pretrained(model_id)
    mdl.eval()

    def scorer(words: list[str], idx: int, cands: list[str]) -> str | None:
        masked = [tok.mask_token if i == idx else w for i, w in enumerate(words)]
        enc = tok(" ".join(masked), return_tensors="pt")
        pos = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]
        with torch.no_grad():
            probs = mdl(**enc).logits[0, pos].softmax(-1)
        best, best_p = None, -1.0
        for c in cands:
            ids = tok.encode(" " + c, add_special_tokens=False)
            if len(ids) == 1:
                p = probs[ids[0]].item()
                if p > best_p:
                    best, best_p = c, p
        return best

    return scorer
