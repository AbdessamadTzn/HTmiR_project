"""Binarisation adaptative (Sauvola) pour manuscrits anciens."""

import cv2
import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def sauvola_binarize(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Binarisation de Sauvola, robuste aux variations locales d'illumination.

    Args:
        image: Image en niveaux de gris (uint8).
        window_size: Taille de la fenêtre locale (pixels, impair).
        k: Paramètre de sensibilité (0.2 recommandé pour manuscrits).
        r: Dynamique de la déviation standard (128.0 par défaut).

    Returns:
        Image binaire (uint8, 0=fond, 255=encre).

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris.

    Example:
        >>> binary = sauvola_binarize(gray_img)
    """
    if image.ndim != 2:
        raise ValueError("Sauvola requiert une image en niveaux de gris (2D).")
    if window_size % 2 == 0:
        window_size += 1

    # Calcul de la moyenne et écart-type locaux via intégrale
    image_f = image.astype(np.float64)
    mean = cv2.boxFilter(image_f, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(image_f ** 2, -1, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    threshold = mean * (1.0 + k * (std / r - 1.0))
    binary = np.where(image_f < threshold, 0, 255).astype(np.uint8)
    logger.debug(f"Sauvola : window={window_size}, k={k}, r={r}")
    return binary


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """Binarisation adaptative OpenCV comme alternative rapide à Sauvola.

    Args:
        image: Image en niveaux de gris (uint8).

    Returns:
        Image binaire (uint8).

    Example:
        >>> binary = adaptive_binarize(gray_img)
    """
    if image.ndim != 2:
        raise ValueError("Requiert une image en niveaux de gris.")
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
