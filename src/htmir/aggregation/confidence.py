"""Scores de confiance et flag needs_review (brief Volet 1)."""

from htmir.evaluation.metrics import compute_cer

CONFIDENCE_THRESHOLD = 0.6
MIN_LINE_LENGTH = 5
MODEL_DISAGREEMENT_CER = 0.30


def line_confidence_from_logits(mean_prob: float | None, fallback: float = 0.75) -> float:
    """Convertit une probabilité moyenne du décodeur en score [0, 1]."""
    if mean_prob is None:
        return fallback
    return float(max(0.0, min(1.0, mean_prob)))


def needs_review(
    confidence: float,
    text: str,
    hyp_alt: str | None = None,
    degraded_region: bool = False,
) -> bool:
    """Décide si une ligne doit être relue manuellement.

    Critères (brief) :
    - confiance < 0.6
    - longueur < 5 caractères
    - discordance TrOCR/Kraken > 30 % CER
    - zone dégradée détectée au prétraitement

    Args:
        confidence: Score calibré [0, 1].
        text: Transcription retenue.
        hyp_alt: Transcription du second modèle (optionnel).
        degraded_region: True si page/ligne marquée dégradée.

    Returns:
        True si relecture humaine recommandée.
    """
    if degraded_region:
        return True
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(text.strip()) < MIN_LINE_LENGTH:
        return True
    if hyp_alt is not None and text and hyp_alt:
        if compute_cer(text, hyp_alt) > MODEL_DISAGREEMENT_CER:
            return True
    return False


def calibrate_confidence(raw: float, cer_on_val: float | None = None) -> float:
    """Ajustement simple : pénalise si le CER validation global est élevé."""
    if cer_on_val is None:
        return raw
    penalty = min(0.25, cer_on_val)
    return float(max(0.0, min(1.0, raw * (1.0 - penalty))))
