"""Tests de validation qualité des images (image_validator)."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from htmir.collection.image_validator import (
    validate_folio_image,
    is_acceptable,
    ImageQuality,
)


def _write_image(img: np.ndarray, suffix: str = ".jpg") -> Path:
    """Sauvegarde un tableau numpy dans un fichier temporaire et retourne son chemin."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = Path(tmp.name)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def valid_folio() -> Path:
    """Folio synthétique valide : texte simulé sur fond clair, 1800×2400px."""
    img = np.ones((2400, 1800), dtype=np.uint8) * 240
    for y in range(200, 2200, 60):
        cv2.line(img, (80, y), (1700, y), 30, 2)
    for x in range(100, 1700, 15):
        img[220:2180, x] = np.clip(img[220:2180, x] - np.random.randint(0, 80), 0, 255)
    path = _write_image(img)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def blank_folio() -> Path:
    """Page entièrement blanche — doit être rejetée."""
    img = np.ones((2400, 1800), dtype=np.uint8) * 255
    path = _write_image(img)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def tiny_folio() -> Path:
    """Image trop petite — doit être rejetée pour résolution insuffisante."""
    img = np.ones((300, 400), dtype=np.uint8) * 200
    cv2.putText(img, "abc", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    path = _write_image(img)
    yield path
    path.unlink(missing_ok=True)


class TestValidateFolioImage:
    def test_valid_folio_passes(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert q.is_readable
        assert q.rejection_reason == ""
        assert q.passed

    def test_valid_folio_dimensions(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert q.width == 1800
        assert q.height == 2400

    def test_blank_folio_rejected(self, blank_folio):
        q = validate_folio_image(blank_folio)
        assert not q.passed
        assert "blank_page" in q.rejection_reason or "insufficient_ink" in q.rejection_reason

    def test_tiny_folio_rejected(self, tiny_folio):
        q = validate_folio_image(tiny_folio)
        assert not q.passed
        assert "resolution_too_low" in q.rejection_reason

    def test_nonexistent_file(self):
        q = validate_folio_image(Path("/nonexistent/image.jpg"))
        assert not q.is_readable
        assert q.rejection_reason == "unreadable"

    def test_megapixels(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert abs(q.megapixels - 4.32) < 0.1

    def test_bleedthrough_score_range(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert 0.0 <= q.bleedthrough_score <= 1.0

    def test_ink_density_positive(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert q.ink_density > 0


class TestIsAcceptable:
    def test_acceptable_valid(self, valid_folio):
        q = validate_folio_image(valid_folio)
        assert is_acceptable(q)

    def test_not_acceptable_blank(self, blank_folio):
        q = validate_folio_image(blank_folio)
        assert not is_acceptable(q)

    def test_not_acceptable_tiny(self, tiny_folio):
        q = validate_folio_image(tiny_folio)
        assert not is_acceptable(q)

    def test_high_bleedthrough_rejected(self, valid_folio):
        q = validate_folio_image(valid_folio)
        # Forcer un score de bleed-through élevé
        q.bleedthrough_score = 0.9
        assert not is_acceptable(q, max_bleedthrough=0.4)

    def test_custom_bleedthrough_threshold(self, valid_folio):
        q = validate_folio_image(valid_folio)
        q.bleedthrough_score = 0.5
        assert not is_acceptable(q, max_bleedthrough=0.4)
        assert is_acceptable(q, max_bleedthrough=0.6)
