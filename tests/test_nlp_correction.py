"""Tests de la correction lexique/contextuelle (htmir.nlp.correction).

Sans téléchargement de modèle : on teste le lexique, la sélection par défaut
(candidat le plus proche) et la sélection par un *scorer* injecté (stub).
"""

from htmir.nlp.correction import correct_line
from htmir.nlp.lexicon import candidates, is_known

LEX = {"ensemble", "comme", "celle", "qui", "moult", "les", "poins", "tordoit"}


# ── lexique ──────────────────────────────────────────────────────────────────

def test_is_known_case_insensitive():
    assert is_known("Comme", LEX)
    assert not is_known("xyzqw", LEX)


def test_candidates_ranked_by_distance():
    cands = candidates("eusemble", LEX, max_distance=2)
    assert "ensemble" in cands
    assert cands[0] == "ensemble"  # distance 1, le plus proche


def test_candidates_none_when_far():
    assert candidates("zzzzzz", LEX, max_distance=2) == []


# ── correction : sélection par défaut (candidat le plus proche) ──────────────

def test_correct_line_default_picks_nearest():
    text = "les poins eusemble tordoit"
    corrected, changes = correct_line(text, LEX)
    assert corrected == "les poins ensemble tordoit"
    assert len(changes) == 1
    assert changes[0]["original"] == "eusemble"
    assert changes[0]["corrected"] == "ensemble"


def test_correct_line_leaves_known_words():
    text = "comme celle qui moult"  # tous attestés
    corrected, changes = correct_line(text, LEX)
    assert corrected == text
    assert changes == []


def test_correct_line_preserves_punctuation():
    text = "poins eusemble,"
    corrected, _ = correct_line(text, LEX)
    assert corrected == "poins ensemble,"


# ── correction : sélection via scorer injecté (sans transformers) ────────────

def test_correct_line_uses_scorer():
    # un scorer qui force un autre candidat que le plus proche
    def scorer(words, idx, cands):
        return "comme" if "comme" in cands else None

    # 'comne' est absent du lexique → suspect ; le scorer choisit 'comme'
    corrected, changes = correct_line("celle comne qui", LEX, scorer=scorer)
    assert "comme" in corrected
    assert changes[0]["corrected"] == "comme"
