"""Lexique d'ancien/moyen français pour la correction guidée.

Source : LGeRM Morphologique (Moyen Français), ~88 000 formes, distribué par
`ILR-Stuttgart/old-french-lemmatization-tools`. Le fichier n'est pas versionné
(licence ATILF, 2 Mo) — :func:`ensure_lexicon` le télécharge si absent.

Le lexique sert à (1) savoir si un mot est *attesté* et (2) proposer des
*candidats* proches (distance d'édition) pour un mot suspect. Le **choix** du
bon candidat relève ensuite d'un modèle de langue (cf. :mod:`htmir.nlp.correction`).
"""

from pathlib import Path

from htmir.eval.evaluate import levenshtein

LGERM_URL = (
    "https://raw.githubusercontent.com/ILR-Stuttgart/"
    "old-french-lemmatization-tools/main/lexicons/old-french/lgerm/lgerm-medieval.tsv"
)
DEFAULT_PATH = Path("data/lexicons/lgerm-medieval.tsv")


def load_lexicon(path: Path = DEFAULT_PATH, min_len: int = 2) -> set[str]:
    """Charge les formes (colonne 1 du TSV LGeRM), en minuscules.

    Args:
        path: Chemin du TSV ``forme\\tcatégorie\\tlemme``.
        min_len: Longueur minimale d'une forme retenue.

    Returns:
        Ensemble de formes attestées (minuscules, alphabétiques).
    """
    path = Path(path)
    lex: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        form = line.split("\t", 1)[0].strip().lower()
        if len(form) >= min_len and form.isalpha():
            lex.add(form)
    return lex


def ensure_lexicon(path: Path = DEFAULT_PATH, url: str = LGERM_URL) -> Path:
    """Télécharge le lexique LGeRM s'il est absent. Retourne le chemin."""
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def is_known(word: str, lexicon: set[str]) -> bool:
    """Le mot est-il attesté dans le lexique (insensible à la casse) ?"""
    return word.lower() in lexicon


def candidates(
    word: str,
    lexicon: set[str],
    max_distance: int = 2,
    max_len_diff: int = 2,
    k: int = 10,
) -> list[str]:
    """Propose les formes du lexique les plus proches d'un mot.

    Args:
        word: Mot suspect.
        lexicon: Lexique (ensemble de formes).
        max_distance: Distance d'édition maximale.
        max_len_diff: Écart de longueur maximal (filtre rapide).
        k: Nombre de candidats retournés.

    Returns:
        Candidats triés par distance croissante puis ordre alphabétique.
    """
    wl = word.lower()
    scored: list[tuple[int, str]] = []
    for cand in lexicon:
        if abs(len(cand) - len(wl)) > max_len_diff:
            continue
        d = levenshtein(wl, cand)
        if d <= max_distance:
            scored.append((d, cand))
    scored.sort()
    return [c for _, c in scored[:k]]
