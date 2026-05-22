"""Pipeline end-to-end : prétraitement → segmentation → HTR → JSON NLP."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from htmir.preprocessing.pipeline import PreprocessingConfig, preprocess_image
from htmir.segmentation.layout import detect_main_text_region
from htmir.segmentation.lines import segment_lines, crop_line_image
from htmir.segmentation.export_xml import build_page_xml, save_page_xml
from htmir.htr.baseline import TrOCRBaseline
from htmir.aggregation.export_json import (
    build_dataset_output,
    build_page_output,
    line_dict_from_prediction,
    export_dataset_nlp,
)
from htmir.evaluation.metrics import corpus_cer, corpus_wer
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def process_page(
    image_path: Path,
    preprocessed_dir: Path,
    segmentations_dir: Path,
    model: TrOCRBaseline | None,
    preprocess_config: PreprocessingConfig,
) -> dict:
    """Traite une page complète et retourne l'entrée pages[] du data contract."""
    result = preprocess_image(image_path, preprocess_config)
    page_id = image_path.stem
    enhanced_path = preprocessed_dir / f"{page_id}_enhanced.png"
    cv2.imwrite(str(enhanced_path), result.enhanced)

    page_img = result.enhanced
    region = detect_main_text_region(page_img)
    lines = segment_lines(page_img)
    if not lines:
        logger.warning(f"Aucune ligne sur {page_id}")

    h, w = page_img.shape[:2]
    text_regions = [
        {
            "id": region.region_id,
            "polygon": np.array(region.polygon),
            "lines": [],
        }
    ]

    line_outputs = []
    for seg in lines:
        crop = crop_line_image(page_img, seg)
        line_img = preprocessed_dir / f"{page_id}_{seg.line_id}.png"
        cv2.imwrite(str(line_img), crop)

        text, conf = ("", 0.0)
        if model is not None:
            text, conf = model.transcribe_image(line_img)

        line_outputs.append(
            line_dict_from_prediction(seg, text, conf, degraded=result.skew_angle > 5.0)
        )
        text_regions[0]["lines"].append(
            {
                "id": seg.line_id,
                "polygon": np.array(seg.polygon),
                "baseline": np.array(seg.baseline),
                "text": text,
            }
        )

    xml_path = segmentations_dir / f"{page_id}.xml"
    tree = build_page_xml(image_path.name, w, h, text_regions)
    save_page_xml(tree, xml_path)

    return build_page_output(
        page_id=page_id,
        image_filename=image_path.name,
        page_xml_path=str(xml_path.as_posix()),
        lines=line_outputs,
    )


def run_pipeline(
    input_dir: Path,
    output_root: Path = Path("."),
    use_baseline: bool = True,
    force_mirror: bool = True,
) -> Path:
    """Exécute le pipeline sur toutes les images d'un dossier."""
    preprocessed_dir = output_root / "data" / "preprocessed"
    segmentations_dir = output_root / "segmentations"
    nlp_path = output_root / "dataset_nlp" / "vinci_output.json"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    segmentations_dir.mkdir(parents=True, exist_ok=True)

    config = PreprocessingConfig(normalize_mirror=True, force_mirror_flip=force_mirror)
    model = TrOCRBaseline() if use_baseline else None

    extensions = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
    pages = []

    for img_path in images:
        try:
            page = process_page(
                img_path, preprocessed_dir, segmentations_dir, model, config
            )
            pages.append(page)
        except Exception as e:
            logger.error(f"Échec {img_path.name} : {e}")

    cer, wer = 0.0, 0.0
    output = build_dataset_output(
        corpus="vinci-notebooks-inverted",
        model="trocr-base-handwritten-baseline" if use_baseline else "ensemble",
        pages=pages,
        cer_global=cer,
        wer_global=wer,
    )
    export_dataset_nlp(output, nlp_path)
    logger.info(f"Pipeline terminé — {len(pages)} page(s), JSON : {nlp_path}")
    return nlp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline HTR Vinci end-to-end.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--no-baseline", action="store_true", help="Segmentation seule, sans TrOCR")
    parser.add_argument("--no-mirror", action="store_true", help="Désactiver normalisation miroir")
    args = parser.parse_args()
    run_pipeline(
        args.input,
        args.output_root,
        use_baseline=not args.no_baseline,
        force_mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
