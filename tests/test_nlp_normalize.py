"""Tests de la normalisation (htmir.nlp.normalize).

Inclut le test exigé par le brief : la normalisation par règles, appliquée des
deux côtés, ne doit PAS dégrader le CER sur un échantillon de référence.
"""

import unicodedata

from htmir.eval.evaluate import corpus_metrics
from htmir.nlp.normalize import (
    count_residual_abbreviations,
    expand_abbreviations,
    fold_ij,
    fold_uv,
    normalize_for_cer,
    normalize_steps_cumulative,
    to_nfc,
)


# ── briques ──────────────────────────────────────────────────────────────────

def test_nfc_idempotent():
    s = unicodedata.normalize("NFD", "é")  # e + accent combinant
    assert to_nfc(s) == "é"
    assert len(to_nfc(s)) == 1


def test_fold_uv_ij():
    assert fold_uv("vie") == "uie"
    assert fold_ij("jour") == "iour"


def test_normalize_for_cer_combines():
    assert normalize_for_cer("Vie  Jour ſont") == "Uie Iour sont"


def test_cumulative_steps_count():
    steps = normalize_steps_cumulative("abc")
    # brut + 5 étapes (NFC, u/v, i/j, long_s, espaces)
    assert len(steps) == 6
    assert steps[0][0] == "brut"


# ── expansion (lisibilité) ───────────────────────────────────────────────────

def test_expand_tironian_et():
    assert expand_abbreviations("a ⁊ b") == "a et b"


def test_expand_nasal_tilde():
    # 'ã' (a + tilde combinant) → 'an'
    s = unicodedata.normalize("NFC", "ã")
    assert expand_abbreviations(s) == "an"


def test_count_residual_abbreviations():
    assert count_residual_abbreviations("a ⁊ b") >= 1


# ── test clé du brief : la normalisation ne dégrade pas le CER ────────────────

def test_normalization_does_not_degrade_cer():
    """Sur un échantillon de référence, CER(norm(gt), norm(hyp)) <= CER(gt, hyp).

    La normalisation appliquée des deux côtés ne peut que fusionner des
    distinctions → le CER ne doit jamais augmenter.
    """
    pairs = [
        ("li rois de France", "li rovs de Irance"),   # v/i erreurs graphiques
        ("vie eternele", "uie eternele"),             # u/v
        ("ſeignor", "seignor"),                        # s long
        ("bonne dame", "bonne  dame"),                 # espaces
    ]
    cer_raw = corpus_metrics(pairs)["cer"]
    norm_pairs = [(normalize_for_cer(r), normalize_for_cer(h)) for r, h in pairs]
    cer_norm = corpus_metrics(norm_pairs)["cer"]
    assert cer_norm <= cer_raw
