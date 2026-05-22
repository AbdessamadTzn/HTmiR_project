"""Métriques d'évaluation HTR : CER, WER, IAA."""

import editdistance
import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def compute_cer(hypothesis: str, reference: str) -> float:
    """Calcule le Character Error Rate (CER).

    Args:
        hypothesis: Transcription produite par le modèle.
        reference: Transcription de référence (ground truth).

    Returns:
        CER entre 0.0 et 1.0+ (distance Levenshtein / longueur référence).

    Raises:
        ValueError: Si la référence est vide.

    Example:
        >>> compute_cer("le chat", "le chien") 
        0.5
    """
    if not reference:
        raise ValueError("La référence ne peut pas être vide.")
    dist = editdistance.eval(hypothesis, reference)
    return dist / len(reference)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Calcule le Word Error Rate (WER).

    Args:
        hypothesis: Transcription produite par le modèle.
        reference: Transcription de référence.

    Returns:
        WER entre 0.0 et 1.0+.

    Raises:
        ValueError: Si la référence est vide.

    Example:
        >>> compute_wer("le chat noir", "le chien noir")
        0.333
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        raise ValueError("La référence ne peut pas être vide.")
    dist = editdistance.eval(hyp_words, ref_words)
    return dist / len(ref_words)


def corpus_cer(hypotheses: list[str], references: list[str]) -> float:
    """CER global sur un corpus (distance totale / caractères totaux).

    Args:
        hypotheses: Liste de transcriptions produites.
        references: Liste de transcriptions de référence.

    Returns:
        CER global du corpus.

    Raises:
        ValueError: Si les listes ont des longueurs différentes.

    Example:
        >>> corpus_cer(["abc", "def"], ["abc", "xyz"])
        0.5
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses et references doivent avoir la même longueur.")
    total_dist = sum(editdistance.eval(h, r) for h, r in zip(hypotheses, references))
    total_ref = sum(len(r) for r in references)
    if total_ref == 0:
        raise ValueError("Corpus de référence vide.")
    return total_dist / total_ref


def corpus_wer(hypotheses: list[str], references: list[str]) -> float:
    """WER global sur un corpus.

    Args:
        hypotheses: Liste de transcriptions produites.
        references: Liste de transcriptions de référence.

    Returns:
        WER global du corpus.

    Example:
        >>> corpus_wer(["le chat", "bon jour"], ["le chien", "bonjour"])
        0.5
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses et references doivent avoir la même longueur.")
    total_dist = sum(
        editdistance.eval(h.split(), r.split())
        for h, r in zip(hypotheses, references)
    )
    total_ref = sum(len(r.split()) for r in references)
    if total_ref == 0:
        raise ValueError("Corpus de référence vide.")
    return total_dist / total_ref


def iaa_cer(annotations_a: list[str], annotations_b: list[str]) -> float:
    """CER inter-annotateurs (Inter-Annotator Agreement).

    Args:
        annotations_a: Transcriptions de l'annotateur A.
        annotations_b: Transcriptions de l'annotateur B.

    Returns:
        CER entre les deux annotateurs (plafond humain).

    Example:
        >>> iaa_cer(["le chat"], ["le chât"])
        0.14
    """
    return corpus_cer(annotations_a, annotations_b)
