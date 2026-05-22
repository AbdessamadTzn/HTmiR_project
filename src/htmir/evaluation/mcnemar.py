"""Test de McNemar pour comparer deux modèles HTR sur le même corpus."""

import numpy as np
from scipy import stats
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def mcnemar_test(
    errors_a: list[bool],
    errors_b: list[bool],
) -> dict:
    """Test de McNemar pour comparer deux systèmes HTR.

    Args:
        errors_a: Booléens indiquant les erreurs du modèle A (True = erreur).
        errors_b: Booléens indiquant les erreurs du modèle B (True = erreur).

    Returns:
        Dict avec 'statistic', 'p_value', 'significant' (p < 0.05), 'n01', 'n10'.

    Raises:
        ValueError: Si les listes ont des longueurs différentes.

    Example:
        >>> result = mcnemar_test(errors_trocr, errors_kraken)
        >>> print(f"p={result['p_value']:.4f}, significatif={result['significant']}")
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("Les deux listes doivent avoir la même longueur.")
    errors_a = np.array(errors_a, dtype=bool)
    errors_b = np.array(errors_b, dtype=bool)
    # n01 : A correct, B erreur ; n10 : A erreur, B correct
    n01 = int(np.sum(~errors_a & errors_b))
    n10 = int(np.sum(errors_a & ~errors_b))
    # Statistique avec correction de continuité (Yates)
    if (n01 + n10) == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))
    result = {
        "statistic": float(statistic),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n01": n01,
        "n10": n10,
    }
    logger.info(f"McNemar : χ²={statistic:.4f}, p={p_value:.4f}, sig={p_value < 0.05}")
    return result
