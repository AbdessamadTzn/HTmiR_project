"""Tests de normalisation écriture miroir."""

import numpy as np
import cv2

from htmir.preprocessing.mirror import normalize_mirror_writing, is_likely_mirror_writing


def test_force_mirror_flip():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(img, "abc", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    out, flipped = normalize_mirror_writing(img, force=True)
    assert flipped
    assert out.shape == img.shape


def test_mirror_detection_heuristic():
    img = np.full((80, 200), 255, dtype=np.uint8)
    img[:, 150:] = 0
    assert is_likely_mirror_writing(img)
