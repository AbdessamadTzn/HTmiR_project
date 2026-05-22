"""Agrégation de transcriptions par alignement Needleman-Wunsch."""

import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def needleman_wunsch(seq_a: str, seq_b: str,
                     match: int = 1, mismatch: int = -1, gap: int = -1) -> tuple[str, str]:
    """Alignement global Needleman-Wunsch entre deux séquences de caractères.

    Args:
        seq_a: Première séquence (transcription modèle A).
        seq_b: Deuxième séquence (transcription modèle B).
        match: Score de correspondance.
        mismatch: Score de non-correspondance.
        gap: Pénalité de gap.

    Returns:
        Tuple (aligned_a, aligned_b) avec '-' pour les gaps.

    Example:
        >>> a, b = needleman_wunsch("ACGT", "ACCT")
        >>> assert len(a) == len(b)
    """
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i-1][j-1] + (match if seq_a[i-1] == seq_b[j-1] else mismatch)
            dp[i][j] = max(diag, dp[i-1][j] + gap, dp[i][j-1] + gap)

    aligned_a, aligned_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match if seq_a[i-1] == seq_b[j-1] else mismatch
            if dp[i][j] == dp[i-1][j-1] + score:
                aligned_a.append(seq_a[i-1]); aligned_b.append(seq_b[j-1]); i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + gap:
            aligned_a.append(seq_a[i-1]); aligned_b.append('-'); i -= 1
        else:
            aligned_a.append('-'); aligned_b.append(seq_b[j-1]); j -= 1

    return "".join(reversed(aligned_a)), "".join(reversed(aligned_b))


def aggregate_transcriptions(
    hyp_a: str, conf_a: float,
    hyp_b: str, conf_b: float,
) -> tuple[str, float]:
    """Agrège deux transcriptions par vote pondéré après alignement NW.

    Args:
        hyp_a: Transcription du modèle A.
        conf_a: Confiance du modèle A (0-1).
        hyp_b: Transcription du modèle B.
        conf_b: Confiance du modèle B (0-1).

    Returns:
        Tuple (transcription_agrégée, confiance_moyenne).

    Example:
        >>> text, conf = aggregate_transcriptions("le chat", 0.9, "le chet", 0.7)
    """
    aligned_a, aligned_b = needleman_wunsch(hyp_a, hyp_b)
    result = []
    for ca, cb in zip(aligned_a, aligned_b):
        if ca == cb:
            result.append(ca)
        elif ca == '-':
            result.append(cb if conf_b >= conf_a else '')
        elif cb == '-':
            result.append(ca if conf_a >= conf_b else '')
        else:
            result.append(ca if conf_a >= conf_b else cb)
    aggregated = "".join(c for c in result if c != '-')
    avg_conf = (conf_a + conf_b) / 2
    return aggregated, avg_conf
