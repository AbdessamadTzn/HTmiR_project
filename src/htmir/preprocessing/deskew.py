"""Correction d'inclinaison des images de manuscrits."""

import cv2
import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def detect_skew_angle(image: np.ndarray) -> float:
    """Détecte l'angle d'inclinaison d'une image via la transformée de Hough.

    Args:
        image: Image en niveaux de gris (uint8).

    Returns:
        Angle d'inclinaison en degrés (négatif = penche à gauche).

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris.

    Example:
        >>> angle = detect_skew_angle(gray_img)
    """
    if image.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2D).")
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return 0.0
    angle = np.median(angles)
    # On ne corrige que les petits angles (max ±45°)
    if abs(angle) > 45:
        angle = 0.0
    return float(angle)


def deskew(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Corrige l'inclinaison d'une image.

    Args:
        image: Image BGR ou niveaux de gris.
        angle: Angle de correction en degrés. Si None, détecté automatiquement.

    Returns:
        Image redressée de même taille.

    Example:
        >>> straight = deskew(image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if angle is None:
        angle = detect_skew_angle(gray)
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    logger.debug(f"Deskew appliqué : angle={angle:.2f}°")
    return rotated
