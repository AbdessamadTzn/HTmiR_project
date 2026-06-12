"""Tests unitaires pour htmir.data.prepare_catmus."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from htmir.data import prepare_catmus


# ── _build_query ────────────────────────────────────────────────────────────


def test_build_query_contains_filter():
    """La requête doit filtrer sur langue, siècle et texte non vide."""
    q = prepare_catmus._build_query(["http://a.parquet"], "French", 13)
    assert "language = 'French'" in q
    assert "century = 13" in q
    assert "im.bytes" in q
    assert "length(trim(text)) > 0" in q


def test_build_query_multiple_urls():
    """Toutes les URLs doivent apparaître dans read_parquet."""
    urls = ["http://a.parquet", "http://b.parquet"]
    q = prepare_catmus._build_query(urls, "French", 13)
    assert "'http://a.parquet'" in q
    assert "'http://b.parquet'" in q


# ── get_parquet_urls ────────────────────────────────────────────────────────


def test_get_parquet_urls_filters_by_split():
    """Seules les URLs du split demandé sont retournées."""
    fake = {
        "parquet_files": [
            {"split": "train", "url": "http://train0.parquet"},
            {"split": "train", "url": "http://train1.parquet"},
            {"split": "test", "url": "http://test0.parquet"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake
    mock_resp.raise_for_status = MagicMock()

    with patch.object(prepare_catmus.requests, "get", return_value=mock_resp):
        urls = prepare_catmus.get_parquet_urls("CATMuS/medieval", "train")

    assert urls == ["http://train0.parquet", "http://train1.parquet"]


def test_get_parquet_urls_empty_split():
    """Un split inexistant retourne une liste vide."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"parquet_files": []}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(prepare_catmus.requests, "get", return_value=mock_resp):
        assert prepare_catmus.get_parquet_urls("x/y", "validation") == []


# ── extract_split (intégration DuckDB + PIL sur fixture locale) ──────────────


def test_extract_split_keeps_only_matching_rows(catmus_parquet, tmp_path):
    """Seules les lignes French/13e non vides sont extraites (3 sur 6)."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()

    out_dir = tmp_path / "train"
    n = prepare_catmus.extract_split(
        con, [catmus_parquet], "French", 13, out_dir,
        start_idx=0, max_samples=None,
    )

    assert n == 3
    pngs = sorted(out_dir.glob("*.png"))
    txts = sorted(out_dir.glob("*.gt.txt"))
    assert len(pngs) == 3
    assert len(txts) == 3
    # Les transcriptions exclues ne doivent pas apparaître
    all_text = " ".join(p.read_text(encoding="utf-8") for p in txts)
    assert "bono regimine" not in all_text   # latin exclu
    assert "rois fu en France" not in all_text  # 14e exclu


def test_extract_split_respects_max_samples(catmus_parquet, tmp_path):
    """max_samples plafonne le nombre de lignes écrites."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    out_dir = tmp_path / "train"
    n = prepare_catmus.extract_split(
        con, [catmus_parquet], "French", 13, out_dir,
        start_idx=0, max_samples=2,
    )
    assert n == 2
    assert len(list(out_dir.glob("*.png"))) == 2


def test_extract_split_naming_offset(catmus_parquet, tmp_path):
    """start_idx décale la numérotation des fichiers."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    out_dir = tmp_path / "test"
    prepare_catmus.extract_split(
        con, [catmus_parquet], "French", 13, out_dir,
        start_idx=100, max_samples=1,
    )
    assert (out_dir / "line_000100.png").exists()
    assert (out_dir / "line_000100.gt.txt").exists()


# ── prepare (orchestration, splits mockés) ──────────────────────────────────


def test_prepare_writes_manifest(catmus_parquet, tmp_path):
    """prepare() doit écrire un manifeste JSON avec les comptes par split."""
    cfg = {
        "dataset": {
            "hf_repo": "CATMuS/medieval",
            "filter": {"language": "French", "century": 13},
            "splits": ["train", "test"],
        },
        "output": {"local_dir": str(tmp_path / "out")},
    }

    # Mock get_parquet_urls : train -> fixture, test -> vide
    def fake_urls(repo, split):
        return [catmus_parquet] if split == "train" else []

    with patch.object(prepare_catmus, "get_parquet_urls", side_effect=fake_urls):
        summary = prepare_catmus.prepare(cfg, overrides={})

    assert summary["total"] == 3
    assert summary["splits"]["train"] == 3
    manifest = Path(cfg["output"]["local_dir"]) / "dataset_manifest.json"
    assert manifest.exists()
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["filter"] == {"language": "French", "century": 13}
