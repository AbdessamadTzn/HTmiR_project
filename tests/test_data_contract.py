"""Tests du data contract JSON."""

import pytest
import jsonschema
from htmir.aggregation.data_contract import validate_output, sha256_file
from pathlib import Path
import tempfile, json


def make_valid_output():
    return {
        "metadata": {
            "corpus": "catmus-medieval",
            "model": "trocr-base-handwritten-lora-r8",
            "cer_global": 0.07,
            "wer_global": 0.12,
            "created_at": "2026-05-01T10:00:00",
            "train_hash_sha256": "abc123",
        },
        "pages": [
            {
                "page_id": "ms_001_p001",
                "image_filename": "ms_001_p001.tif",
                "page_xml_path": "segmentations/ms_001_p001.xml",
                "lines": [
                    {
                        "line_id": "line_001",
                        "text": "En icele tens fu uns hom",
                        "confidence": 0.92,
                        "needs_review": False,
                        "polygon": [[10, 20], [200, 20], [200, 40], [10, 40]],
                        "baseline": [[10, 38], [200, 38]],
                    }
                ],
            }
        ],
    }


class TestDataContract:
    def test_valid_output_passes(self):
        validate_output(make_valid_output())

    def test_missing_metadata_fails(self):
        data = make_valid_output()
        del data["metadata"]
        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)

    def test_missing_cer_fails(self):
        data = make_valid_output()
        del data["metadata"]["cer_global"]
        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)

    def test_negative_confidence_fails(self):
        data = make_valid_output()
        data["pages"][0]["lines"][0]["confidence"] = -0.1
        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)

    def test_confidence_above_1_fails(self):
        data = make_valid_output()
        data["pages"][0]["lines"][0]["confidence"] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)

    def test_missing_polygon_fails(self):
        data = make_valid_output()
        del data["pages"][0]["lines"][0]["polygon"]
        with pytest.raises(jsonschema.ValidationError):
            validate_output(data)

    def test_sha256_deterministic(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"test": 42}, f)
            p = Path(f.name)
        h1 = sha256_file(p)
        h2 = sha256_file(p)
        assert h1 == h2
        assert len(h1) == 64
