"""Validation de qualité des images de manuscrits avant upload S3.

Quatre critères séquentiels (arrêt au premier échec) :
  1. Lisibilité du fichier image.
  2. Résolution minimale.
  3. Ratio de fond clair (page vide).
  4. Densité d'encre minimale.

Un cinquième indicateur non-bloquant, le score de bleed-through
(transpercement d'encre), est calculé pour la traçabilité.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageQuality:
    """Résultat de l'analyse qualité d'un folio numérisé.

    Args:
        path: Chemin analysé.
        width: Largeur en pixels (0 si non lisible).
        height: Hauteur en pixels (0 si non lisible).
        ink_density: Part de pixels « encre » après seuillage Otsu (0–1).
        blank_ratio: Part de pixels très clairs (fond ≥ 230) (0–1).
        bleedthrough_score: Heuristique de transpercement (0–1).
        is_readable: Le fichier a pu être ouvert et décodé.
        rejection_reason: Motif de rejet (vide si l'image est acceptable).
    """

    path: Path
    width: int = 0
    height: int = 0
    ink_density: float = 0.0
    blank_ratio: float = 0.0
    bleedthrough_score: float = 0.0
    is_readable: bool = False
    rejection_reason: str = ""

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def passed(self) -> bool:
        return self.is_readable and not self.rejection_reason


def validate_folio_image(
    path: Path,
    min_width: int = 1200,
    min_height: int = 800,
    max_blank_ratio: float = 0.95,
    min_ink_density: float = 0.005,
) -> ImageQuality:
    """Analyse la qualité d'un folio numérisé.

    Le pipeline de validation s'arrête dès le premier critère non satisfait
    pour éviter des calculs inutiles sur des images clairement hors normes.

    Args:
        path: Chemin vers l'image (JPEG, PNG, TIFF).
        min_width: Largeur minimale acceptable (pixels).
        min_height: Hauteur minimale acceptable (pixels).
        max_blank_ratio: Ratio maximal de pixels très clairs avant rejet « page vide ».
        min_ink_density: Densité minimale de pixels encre requise.

    Returns:
        :class:`ImageQuality` décrivant les résultats de chaque contrôle.
    """
    quality = ImageQuality(path=Path(path))

    # 1 — Lisibilité
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        quality.rejection_reason = "unreadable"
        logger.warning(f"Image illisible : {path}")
        return quality

    quality.is_readable = True
    quality.height, quality.width = img.shape

    # 2 — Résolution minimale
    if quality.width < min_width or quality.height < min_height:
        quality.rejection_reason = (
            f"resolution_too_low ({quality.width}×{quality.height}px "
            f"— minimum {min_width}×{min_height}px)"
        )
        return quality

    # 3 — Page vide (fond très clair dominant)
    quality.blank_ratio = float(np.mean(img >= 230))
    if quality.blank_ratio > max_blank_ratio:
        quality.rejection_reason = (
            f"blank_page (blank_ratio={quality.blank_ratio:.3f} > {max_blank_ratio})"
        )
        return quality

    # 4 — Densité d'encre (seuil Otsu inversé)
    _, binary_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    quality.ink_density = float(np.count_nonzero(binary_inv)) / img.size
    if quality.ink_density < min_ink_density:
        quality.rejection_reason = (
            f"insufficient_ink (ink_density={quality.ink_density:.5f} < {min_ink_density})"
        )
        return quality

    # 5 — Score de bleed-through (indicatif, non-bloquant)
    # Calcul de la variance locale dans les zones de fond clair :
    # une forte variance indique des taches de transpercement.
    kernel = np.ones((15, 15), np.float32) / 225
    local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
    light_mask = img > 200
    if light_mask.any():
        residual = np.abs(img.astype(np.float32) - local_mean)
        quality.bleedthrough_score = float(
            min(np.mean(residual[light_mask]) / 30.0, 1.0)
        )

    logger.debug(
        f"Qualité {path.name} : {quality.width}×{quality.height}px | "
        f"encre={quality.ink_density:.4f} | "
        f"fond={quality.blank_ratio:.3f} | "
        f"bleed={quality.bleedthrough_score:.3f}"
    )
    return quality


def is_acceptable(quality: ImageQuality, max_bleedthrough: float = 0.4) -> bool:
    """Retourne ``True`` si l'image passe tous les critères de qualité.

    Args:
        quality: Résultat renvoyé par :func:`validate_folio_image`.
        max_bleedthrough: Seuil maximal pour le score de bleed-through.

    Returns:
        ``True`` si l'image peut être uploadée sur S3.
    """
    return quality.passed and quality.bleedthrough_score <= max_bleedthrough
