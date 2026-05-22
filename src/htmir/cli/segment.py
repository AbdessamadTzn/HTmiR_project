"""CLI : segmentation de lignes + export PAGE XML."""

import argparse
from pathlib import Path

import cv2

from htmir.segmentation.lines import segment_page_file
from htmir.segmentation.layout import detect_main_text_region
from htmir.segmentation.export_xml import build_page_xml, save_page_xml
from htmir.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmente des pages et exporte PAGE XML.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("segmentations"))
    parser.add_argument("--lines-dir", type=Path, default=Path("data/lines"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.lines_dir.mkdir(parents=True, exist_ok=True)

    for img in sorted(args.input.glob("*")):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        page = cv2.imread(str(img))
        if page is None:
            continue
        region = detect_main_text_region(page)
        segments = segment_page_file(img, args.lines_dir)
        h, w = page.shape[:2]
        lines_xml = [
            {
                "id": s.line_id,
                "polygon": np.array(s.polygon),
                "baseline": np.array(s.baseline),
                "text": "",
            }
            for s in segments
        ]
        tree = build_page_xml(
            img.name,
            w,
            h,
            [{"id": region.region_id, "polygon": np.array(region.polygon), "lines": lines_xml}],
        )
        save_page_xml(tree, args.output / f"{img.stem}.xml")
    logger.info(f"Segmentation terminée → {args.output}")
