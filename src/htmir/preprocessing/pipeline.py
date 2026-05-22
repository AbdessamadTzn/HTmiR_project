"""Pipeline de prétraitement complet pour images de manuscrits anciens."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from htmir.preprocessing.deskew import deskew
from htmir.preprocessing.clahe import apply_clahe
from htmir.preprocessing.binarize import sauvola_binarize
from htmir.utils.logger import get_logger
from htmir.utils.seeds import fixer_seeds

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration du pipeline de prétraitement.

    Args:
        clahe_clip_limit: Limite de contraste CLAHE.
        clahe_tile_grid: Taille de grille pour CLAHE.
        sauvola_window: Fenêtre de binarisation Sauvola.
        sauvola_k: Paramètre k de Sauvola.
        sauvola_r: Paramètre r de Sauvola.
        auto_deskew: Active la correction d'inclinaison automatique.
    """
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    sauvola_r: float = 128.0
    auto_deskew: bool = True


@dataclass
class PreprocessingResult:
    """Résultat du pipeline de prétraitement.

    Args:
        original: Image originale.
        deskewed: Image après correction d'inclinaison.
        enhanced: Image après CLAHE.
        binary: Image binarisée finale.
        skew_angle: Angle d'inclinaison détecté.
    """
    original: np.ndarray
    deskewed: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    skew_angle: float


def preprocess_image(
    image_path: Path | str,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Applique le pipeline complet de prétraitement à une image de manuscrit.

    Chaîne : lecture → deskew → CLAHE → Sauvola.

    Args:
        image_path: Chemin vers l'image (TIFF, JPEG, PNG).
        config: Configuration du pipeline. Défaut si None.

    Returns:
        PreprocessingResult avec toutes les étapes intermédiaires.

    Raises:
        FileNotFoundError: Si l'image n'existe pas.
        ValueError: Si l'image ne peut pas être lue.

    Example:
        >>> result = preprocess_image("page_001.tif")
        >>> cv2.imwrite("binary.png", result.binary)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    config = config or PreprocessingConfig()
    logger.info(f"Prétraitement : {image_path.name}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    # Étape 1 : deskew
    skew_angle = 0.0
    if config.auto_deskew:
        gray_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        from htmir.preprocessing.deskew import detect_skew_angle
        skew_angle = detect_skew_angle(gray_tmp)
    deskewed = deskew(image, angle=skew_angle if config.auto_deskew else 0.0)

    # Étape 2 : CLAHE
    enhanced = apply_clahe(
        deskewed,
        clip_limit=config.clahe_clip_limit,
        tile_grid_size=config.clahe_tile_grid,
    )

    # Étape 3 : Sauvola
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = sauvola_binarize(gray, window_size=config.sauvola_window,
                              k=config.sauvola_k, r=config.sauvola_r)

    logger.info(f"Prétraitement terminé : skew={skew_angle:.2f}°")
    return PreprocessingResult(
        original=image,
        deskewed=deskewed,
        enhanced=enhanced,
        binary=binary,
        skew_angle=skew_angle,
    )


def batch_preprocess(
    input_dir: Path,
    output_dir: Path,
    config: PreprocessingConfig | None = None,
    extensions: tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png"),
) -> list[Path]:
    """Prétraite un dossier complet d'images de manuscrits.

    Args:
        input_dir: Dossier contenant les images sources.
        output_dir: Dossier de sortie pour les images binarisées.
        config: Configuration du pipeline.
        extensions: Extensions de fichiers acceptées.

    Returns:
        Liste des chemins vers les images binarisées produites.

    Example:
        >>> paths = batch_preprocess(Path("raw/"), Path("preprocessed/"))
    """
    fixer_seeds(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in extensions]
    logger.info(f"{len(images)} images à traiter dans {input_dir}")

    output_paths = []
    for img_path in images:
        try:
            result = preprocess_image(img_path, config)
            out_path = output_dir / (img_path.stem + "_binary.png")
            cv2.imwrite(str(out_path), result.binary)
            output_paths.append(out_path)
        except Exception as e:
            logger.error(f"Erreur sur {img_path.name} : {e}")

    logger.info(f"{len(output_paths)}/{len(images)} images prétraitées")
    return output_paths
