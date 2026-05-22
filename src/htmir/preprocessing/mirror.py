"""Normalisation de l'écriture miroir (manuscrits de Léonard de Vinci)."""

import cv2
import numpy as np


def is_likely_mirror_writing(image: np.ndarray, sample_cols: int = 80) -> bool:
    """Heuristique : compare la densité d'encre sur les bords gauche/droit.

    Sur une page Vinci numérisée en sens « lecture miroir », le texte commence
    souvent du côté droit de l'image brute. Cette détection reste indicative.

    Args:
        image: Image BGR ou niveaux de gris.
        sample_cols: Largeur (en px) des bandes latérales analysées.

    Returns:
        True si le bord droit semble plus dense que le gauche (écriture miroir probable).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    band = min(sample_cols, w // 4)
    if band < 1:
        return False
    left = gray[:, :band]
    right = gray[:, w - band :]
    left_ink = float(np.mean(left < 200))
    right_ink = float(np.mean(right < 200))
    return right_ink > left_ink * 1.15


def normalize_mirror_writing(image: np.ndarray, force: bool = False) -> tuple[np.ndarray, bool]:
    """Retourne l'image en sens de lecture (flip horizontal si miroir détecté).

    Args:
        image: Image BGR.
        force: Applique le flip même si l'heuristique est négative.

    Returns:
        Tuple (image_normalisée, flipped).
    """
    flipped = force or is_likely_mirror_writing(image)
    if flipped:
        return cv2.flip(image, 1), True
    return image, False
