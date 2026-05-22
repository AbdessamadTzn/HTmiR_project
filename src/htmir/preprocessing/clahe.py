"""Amélioration du contraste par CLAHE pour manuscrits anciens."""

import cv2
import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applique CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Image BGR ou niveaux de gris.
        clip_limit: Limite de contraste adaptatif (2.0 recommandé pour manuscrits).
        tile_grid_size: Taille de la grille de tuiles.

    Returns:
        Image avec contraste amélioré (même format que l'entrée).

    Example:
        >>> enhanced = apply_clahe(image, clip_limit=2.0)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        result = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    else:
        result = clahe.apply(image)
    logger.debug(f"CLAHE appliqué : clip_limit={clip_limit}, grid={tile_grid_size}")
    return result
