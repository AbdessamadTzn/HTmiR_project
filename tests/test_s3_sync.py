"""Tests unitaires pour htmir.data.s3_sync."""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock

from htmir.data import s3_sync


# ── construction des clés (pur) ─────────────────────────────────────────────


def test_split_tar_key():
    assert s3_sync.split_tar_key("datasets/catmus", "train") == "datasets/catmus/train.tar.gz"


def test_split_tar_key_strips_trailing_slash():
    assert s3_sync.split_tar_key("datasets/catmus/", "test") == "datasets/catmus/test.tar.gz"


def test_manifest_key():
    assert s3_sync.manifest_key("datasets/x") == "datasets/x/dataset_manifest.json"


# ── make_split_archive ──────────────────────────────────────────────────────


def test_make_split_archive(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "line_000000.png").write_bytes(b"img")
    (split / "line_000000.gt.txt").write_text("li rois", encoding="utf-8")

    archive = tmp_path / "train.tar.gz"
    s3_sync.make_split_archive(split, archive)

    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "train/line_000000.png" in names
    assert "train/line_000000.gt.txt" in names


# ── upload_dataset (boto3 mocké) ────────────────────────────────────────────


def test_upload_dataset_archives_each_split(tmp_path):
    root = tmp_path / "ds"
    for split in ("train", "test"):
        d = root / split
        d.mkdir(parents=True)
        (d / "line_000000.png").write_bytes(b"img")
        (d / "line_000000.gt.txt").write_text("x", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("{}", encoding="utf-8")

    client = MagicMock()
    keys = s3_sync.upload_dataset(root, "htmir-data", "datasets/catmus", client)

    # 2 tarballs + 1 manifeste
    assert "datasets/catmus/train.tar.gz" in keys
    assert "datasets/catmus/test.tar.gz" in keys
    assert "datasets/catmus/dataset_manifest.json" in keys
    assert client.upload_file.call_count == 3
    # les archives temporaires sont nettoyées
    assert not list(root.glob("*.tar.gz"))


def test_upload_dataset_skips_empty_split(tmp_path):
    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)        # vide → ignoré
    full = root / "test"
    full.mkdir()
    (full / "line_000000.png").write_bytes(b"img")

    client = MagicMock()
    keys = s3_sync.upload_dataset(root, "b", "p", client)
    assert keys == ["p/test.tar.gz"]


# ── download_dataset (boto3 mocké) ──────────────────────────────────────────


def test_download_dataset_extracts(tmp_path):
    # Prépare une archive train.tar.gz à "télécharger"
    src = tmp_path / "src" / "train"
    src.mkdir(parents=True)
    (src / "line_000000.gt.txt").write_text("li rois", encoding="utf-8")
    archive = tmp_path / "train.tar.gz"
    s3_sync.make_split_archive(src, archive)

    def fake_download(bucket, key, dest):
        Path(dest).write_bytes(archive.read_bytes())

    client = MagicMock()
    client.download_file.side_effect = fake_download

    out = tmp_path / "out"
    s3_sync.download_dataset("b", "p", out, ["train"], client)

    assert (out / "train" / "line_000000.gt.txt").read_text(encoding="utf-8") == "li rois"


def test_download_dataset_missing_split_is_skipped(tmp_path):
    client = MagicMock()
    client.download_file.side_effect = Exception("404 NoSuchKey")
    out = tmp_path / "out"
    # ne doit pas lever
    s3_sync.download_dataset("b", "p", out, ["train"], client)
    assert out.exists()
