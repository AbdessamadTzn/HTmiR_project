"""Tests du flag needs_review et calibration."""

from htmir.aggregation.confidence import needs_review, calibrate_confidence


def test_low_confidence_needs_review():
    assert needs_review(0.5, "une ligne assez longue")


def test_short_text_needs_review():
    assert needs_review(0.9, "oui")


def test_model_disagreement():
    assert needs_review(0.9, "le chat noir", hyp_alt="xyz abc def")


def test_calibrate_penalty():
    assert calibrate_confidence(0.8, cer_on_val=0.2) < 0.8
