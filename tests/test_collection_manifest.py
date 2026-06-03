"""Tests du manifeste de collecte (CollectionManifest + FolioRecord)."""

import json
import tempfile
from pathlib import Path

import pytest

from htmir.collection.manifest_builder import CollectionManifest, FolioRecord


@pytest.fixture
def sample_record() -> FolioRecord:
    return FolioRecord(
        folio_id="btv1b10022860x_f0001",
        source="gallica",
        s3_uri="s3://htmir-data/raw/gallica/btv1b10022860x/folio_0001.jpg",
        ark_id="btv1b10022860x",
        licence="gallica-non-commercial",
        width=2000,
        height=2800,
        sha256="abc123",
        status="validated",
    )


@pytest.fixture
def sample_manifest(sample_record) -> CollectionManifest:
    m = CollectionManifest()
    m.add(sample_record)
    return m


class TestFolioRecord:
    def test_megapixels(self, sample_record):
        assert abs(sample_record.megapixels - 5.6) < 0.01

    def test_default_status(self):
        r = FolioRecord(folio_id="x_f0001", source="gallica", s3_uri="s3://b/k")
        assert r.status == "downloaded"

    def test_default_date_is_iso8601(self):
        r = FolioRecord(folio_id="x_f0001", source="gallica", s3_uri="s3://b/k")
        assert "T" in r.date_downloaded and "+" in r.date_downloaded


class TestCollectionManifest:
    def test_add_and_len(self, sample_record):
        m = CollectionManifest()
        assert len(m) == 0
        m.add(sample_record)
        assert len(m) == 1

    def test_add_updates_existing(self, sample_manifest, sample_record):
        updated = FolioRecord(
            folio_id=sample_record.folio_id,
            source="gallica",
            s3_uri="s3://htmir-data/raw/gallica/btv1b10022860x/folio_0001.jpg",
            status="preprocessed",
        )
        sample_manifest.add(updated)
        assert len(sample_manifest) == 1
        assert sample_manifest.get(sample_record.folio_id).status == "preprocessed"

    def test_contains(self, sample_manifest, sample_record):
        assert sample_record.folio_id in sample_manifest
        assert "nonexistent_f0001" not in sample_manifest

    def test_filter_status(self, sample_manifest):
        m = sample_manifest
        m.add(FolioRecord(folio_id="other_f0001", source="zenodo", s3_uri="", status="rejected"))
        validated = m.filter_status("validated")
        rejected = m.filter_status("rejected")
        assert len(validated) == 1
        assert len(rejected) == 1

    def test_stats(self, sample_manifest):
        m = sample_manifest
        m.add(FolioRecord(folio_id="z_f0001", source="zenodo", s3_uri="", status="rejected"))
        m.add(FolioRecord(folio_id="z_f0002", source="zenodo", s3_uri="", status="rejected"))
        stats = m.stats()
        assert stats["validated"] == 1
        assert stats["rejected"] == 2

    def test_iteration(self, sample_manifest, sample_record):
        records = list(sample_manifest)
        assert len(records) == 1
        assert records[0].folio_id == sample_record.folio_id


class TestManifestPersistence:
    def test_save_and_load_roundtrip(self, sample_manifest):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            sample_manifest.save(path)
            loaded = CollectionManifest.load(path)
            assert len(loaded) == len(sample_manifest)
            r = loaded.get("btv1b10022860x_f0001")
            assert r is not None
            assert r.width == 2000
            assert r.licence == "gallica-non-commercial"
        finally:
            path.unlink(missing_ok=True)

    def test_save_json_structure(self, sample_manifest):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            sample_manifest.save(path)
            data = json.loads(path.read_text())
            assert "schema_version" in data
            assert "total" in data
            assert "records" in data
            assert data["total"] == 1
            assert len(data["records"]) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_load_ignores_unknown_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(
                {
                    "schema_version": "0.9",
                    "total": 1,
                    "records": [
                        {
                            "folio_id": "x_f0001",
                            "source": "gallica",
                            "s3_uri": "s3://b/k",
                            "unknown_future_field": "ignored",
                        }
                    ],
                },
                tmp,
            )
            path = Path(tmp.name)
        try:
            loaded = CollectionManifest.load(path)
            assert len(loaded) == 1
        finally:
            path.unlink(missing_ok=True)
