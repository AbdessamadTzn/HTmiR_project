"""Segmentation de lignes par projection horizontale (pages binarisées ou grises)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LineSegment:
    """Une ligne détectée sur une page.

    Args:
        line_id: Identifiant (ex. line_001).
        y0: Borne verticale haute (px).
        y1: Borne verticale basse (px).
        x0: Marge gauche utile (px).
        x1: Marge droite utile (px).
        polygon: Contour [[x,y], ...] au format data contract.
        baseline: Points de ligne de base [[x,y], ...].
    """

    line_id: str
    y0: int
    y1: int
    x0: int
    x1: int
    polygon: list[list[int]]
    baseline: list[list[int]]


def _ink_projection(binary: np.ndarray) -> np.ndarray:
    """Projection horizontale : somme des pixels sombres par ligne."""
    inv = 255 - binary if binary.max() > 1 else (255 - (binary * 255).astype(np.uint8))
    return np.sum(inv > 127, axis=1).astype(np.float32)


def segment_lines(
    image: np.ndarray,
    min_line_height: int = 12,
    gap_merge: int = 6,
    margin_frac: float = 0.02,
) -> list[LineSegment]:
    """Détecte les bandes de texte par analyse de projection.

    Args:
        image: Image BGR ou binaire (une canal).
        min_line_height: Hauteur minimale d'une ligne en pixels.
        gap_merge: Fusionne les bandes séparées par moins de N px vides.
        margin_frac: Marge horizontale ignorée (fraction de largeur).

    Returns:
        Liste de LineSegment ordonnée de haut en bas (ordre de lecture).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = _ink_projection(binary)
    h, w = binary.shape
    threshold = max(3.0, 0.02 * w)

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y, val in enumerate(proj):
        if val >= threshold and not in_band:
            in_band = True
            start = y
        elif val < threshold and in_band:
            in_band = False
            if y - start >= min_line_height:
                bands.append((start, y))
    if in_band and h - start >= min_line_height:
        bands.append((start, h))

    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    x0 = int(w * margin_frac)
    x1 = int(w * (1 - margin_frac))
    segments: list[LineSegment] = []
    for i, (y0, y1) in enumerate(merged):
        lid = f"line_{i + 1:03d}"
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        baseline = [[x0, y1 - 2], [x1, y1 - 2]]
        segments.append(
            LineSegment(
                line_id=lid,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                polygon=poly,
                baseline=baseline,
            )
        )
    logger.info(f"{len(segments)} ligne(s) détectée(s)")
    return segments


def crop_line_image(page: np.ndarray, segment: LineSegment, pad: int = 4) -> np.ndarray:
    """Extrait l'image d'une ligne avec padding."""
    h, w = page.shape[:2]
    y0 = max(0, segment.y0 - pad)
    y1 = min(h, segment.y1 + pad)
    x0 = max(0, segment.x0 - pad)
    x1 = min(w, segment.x1 + pad)
    return page[y0:y1, x0:x1].copy()


def segment_page_file(
    image_path: Path,
    output_dir: Path | None = None,
) -> list[LineSegment]:
    """Segmente une page depuis un fichier image."""
    image_path = Path(image_path)
    page = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if page is None:
        raise ValueError(f"Image illisible : {image_path}")
    segments = segment_lines(page)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            crop = crop_line_image(page, seg)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_{seg.line_id}.png"), crop)
    return segments
