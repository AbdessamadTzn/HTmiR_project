"""Tests de segmentation de lignes et layout."""

import numpy as np
import cv2

from htmir.segmentation.lines import segment_lines
from htmir.segmentation.layout import detect_main_text_region


def _synthetic_page(n_lines: int = 3) -> np.ndarray:
    h, w = 400, 600
    page = np.full((h, w), 255, dtype=np.uint8)
    for i in range(n_lines):
        y0 = 80 + i * 90
        cv2.rectangle(page, (50, y0), (550, y0 + 25), 0, -1)
    return page


def test_segment_lines_detects_bands():
    page = _synthetic_page(3)
    lines = segment_lines(page, min_line_height=10)
    assert len(lines) >= 2
    assert lines[0].y0 < lines[-1].y0


def test_layout_region_covers_page():
    page = _synthetic_page(1)
    region = detect_main_text_region(page)
    assert len(region.polygon) >= 4
