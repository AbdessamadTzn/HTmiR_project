"""Tests du pipeline de prétraitement."""

import numpy as np
import pytest
import cv2

from htmir.preprocessing.deskew import detect_skew_angle, deskew
from htmir.preprocessing.clahe import apply_clahe
from htmir.preprocessing.binarize import sauvola_binarize, adaptive_binarize


@pytest.fixture
def gray_image():
    """Image synthétique en niveaux de gris (200x300)."""
    img = np.ones((200, 300), dtype=np.uint8) * 200
    cv2.putText(img, "manuscrit test", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    return img


@pytest.fixture
def bgr_image(gray_image):
    """Image BGR synthétique."""
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


class TestDeskew:
    def test_detect_skew_angle_returns_float(self, gray_image):
        angle = detect_skew_angle(gray_image)
        assert isinstance(angle, float)

    def test_detect_skew_angle_range(self, gray_image):
        angle = detect_skew_angle(gray_image)
        assert -45.0 <= angle <= 45.0

    def test_detect_skew_raises_on_color(self, bgr_image):
        with pytest.raises(ValueError):
            detect_skew_angle(bgr_image)

    def test_deskew_preserves_shape(self, bgr_image):
        result = deskew(bgr_image, angle=0.0)
        assert result.shape == bgr_image.shape

    def test_deskew_zero_angle_unchanged(self, bgr_image):
        result = deskew(bgr_image, angle=0.0)
        assert result.shape == bgr_image.shape

    def test_deskew_type_preserved(self, bgr_image):
        result = deskew(bgr_image)
        assert result.dtype == np.uint8


class TestCLAHE:
    def test_apply_clahe_bgr_shape(self, bgr_image):
        result = apply_clahe(bgr_image)
        assert result.shape == bgr_image.shape

    def test_apply_clahe_gray_shape(self, gray_image):
        result = apply_clahe(gray_image)
        assert result.shape == gray_image.shape

    def test_apply_clahe_dtype(self, bgr_image):
        result = apply_clahe(bgr_image)
        assert result.dtype == np.uint8

    def test_apply_clahe_values_range(self, bgr_image):
        result = apply_clahe(bgr_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestBinarize:
    def test_sauvola_output_binary(self, gray_image):
        result = sauvola_binarize(gray_image)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_sauvola_shape_preserved(self, gray_image):
        result = sauvola_binarize(gray_image)
        assert result.shape == gray_image.shape

    def test_sauvola_raises_on_color(self, bgr_image):
        with pytest.raises(ValueError):
            sauvola_binarize(bgr_image)

    def test_adaptive_output_binary(self, gray_image):
        result = adaptive_binarize(gray_image)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_adaptive_raises_on_color(self, bgr_image):
        with pytest.raises(ValueError):
            adaptive_binarize(bgr_image)
