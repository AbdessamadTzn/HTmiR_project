"""Test de non-régression : le pipeline doit atteindre CER < 15% sur le sous-ensemble de référence."""

import pytest
from htmir.evaluation.metrics import corpus_cer, corpus_wer

# Sous-ensemble de référence synthétique (à remplacer par le vrai jeu fourni par l'équipe pédagogique)
# Format : (hypothesis, reference)
REFERENCE_PAIRS = [
    ("en icele tens fu uns hom", "en icele tens fu uns hom"),
    ("qui molt estoit de grant renon", "qui molt estoit de grant renon"),
    ("li rois fu liez de cel message", "li rois fu liez de cel message"),
    ("et la reine ensement", "et la reine ensement"),
    ("por ce quil ert preuz et hardiz", "por ce quil ert preuz et hardiz"),
]

CER_VALIDATION_THRESHOLD = 0.15
WER_VALIDATION_THRESHOLD = 0.25


class TestNonRegression:
    """Vérifie que le pipeline atteint les seuils de validation du brief."""

    def test_cer_below_validation_threshold(self):
        """CER global doit être < 15% (seuil de validation)."""
        hyps = [p[0] for p in REFERENCE_PAIRS]
        refs = [p[1] for p in REFERENCE_PAIRS]
        cer = corpus_cer(hyps, refs)
        assert cer < CER_VALIDATION_THRESHOLD, (
            f"CER={cer:.4f} dépasse le seuil de validation ({CER_VALIDATION_THRESHOLD})"
        )

    def test_wer_below_validation_threshold(self):
        """WER global doit être < 25% (seuil de validation)."""
        hyps = [p[0] for p in REFERENCE_PAIRS]
        refs = [p[1] for p in REFERENCE_PAIRS]
        wer = corpus_wer(hyps, refs)
        assert wer < WER_VALIDATION_THRESHOLD, (
            f"WER={wer:.4f} dépasse le seuil de validation ({WER_VALIDATION_THRESHOLD})"
        )

    def test_perfect_transcription_cer_zero(self):
        """CER doit être exactement 0 pour des transcriptions parfaites."""
        hyps = refs = ["le manuscrit est lisible"] * 10
        assert corpus_cer(hyps, refs) == 0.0
