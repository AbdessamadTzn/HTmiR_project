"""Intervalles de confiance bootstrap sur le CER (N=1000, 95%)."""

import numpy as np
from htmir.evaluation.metrics import compute_cer
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def bootstrap_cer(
    hypotheses: list[str],
    references: list[str],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Calcule l'IC bootstrap sur le CER ligne par ligne.

    Args:
        hypotheses: Transcriptions du modèle.
        references: Transcriptions de référence.
        n_iterations: Nombre de ré-échantillonnages (1000 requis par le brief).
        confidence: Niveau de confiance (0.95 par défaut).
        seed: Seed pour reproductibilité.

    Returns:
        Dict avec clés 'mean', 'lower', 'upper', 'std'.

    Example:
        >>> ic = bootstrap_cer(hyps, refs)
        >>> print(f"CER {ic['mean']:.3f} [{ic['lower']:.3f}, {ic['upper']:.3f}]")
    """
    rng = np.random.default_rng(seed)
    cer_per_line = np.array([
        compute_cer(h, r) for h, r in zip(hypotheses, references)
    ])
    n = len(cer_per_line)
    boot_means = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(cer_per_line, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    logger.info(f"Bootstrap CER : {boot_means.mean():.4f} [{lower:.4f}, {upper:.4f}]")
    return {
        "mean": float(boot_means.mean()),
        "lower": lower,
        "upper": upper,
        "std": float(boot_means.std()),
    }
