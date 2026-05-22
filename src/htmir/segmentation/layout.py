"""Segmentation grossière de la zone de texte (carnets Vinci : texte + dessins)."""

from dataclasses import dataclass

import cv2
import numpy as np

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextRegion:
    """Région de texte principale sur une page.

    Args:
        region_id: Identifiant PAGE XML.
        polygon: Contour [[x,y], ...].
        type: Type PAGE (paragraph, heading, etc.).
    """

    region_id: str
    polygon: list[list[int]]
    type: str = "paragraph"


def detect_main_text_region(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
) -> TextRegion:
    """Détecte la plus grande composante connexe « encre » comme zone de texte.

    Les dessins techniques de Vinci peuvent créer de fausses régions ;
    on privilégie la composante la plus large en bande horizontale.

    Args:
        image: Image BGR ou niveaux de gris.
        min_area_ratio: Surface minimale relative à ignorer.

    Returns:
        TextRegion couvrant la zone principale (ou page entière en repli).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_idx = -1
    best_score = 0.0
    page_area = h * w
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < page_area * min_area_ratio:
            continue
        score = area * (bw / max(bh, 1))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        logger.warning("Aucune région détectée — page entière utilisée")
        poly = [[0, 0], [w, 0], [w, h], [0, h]]
        return TextRegion(region_id="region_001", polygon=poly)

    mask = (labels == best_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    poly = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return TextRegion(region_id="region_001", polygon=poly)
