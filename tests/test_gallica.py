"""Tests du collecteur Gallica (requêtes SRU + téléchargement IIIF mockés)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

from htmir.collection.gallica import (
    search_vinci_manuscripts,
    fetch_iiif_manifest,
    download_folio,
    count_folios,
    GallicaFolio,
    VINCI_SRU_QUERY,
)

# ── Fixtures de réponses simulées ────────────────────────────────────────────

_SRU_XML_TWO_ARKS = b"""<?xml version="1.0" encoding="UTF-8"?>
<srw:searchRetrieveResponse
    xmlns:srw="http://www.loc.gov/zing/srw/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/">
  <srw:records>
    <srw:record>
      <srw:recordData>
        <oai_dc:dc>
          <dc:identifier>https://gallica.bnf.fr/ark:/12148/btv1b10022860x</dc:identifier>
          <dc:type>manuscrit</dc:type>
        </oai_dc:dc>
      </srw:recordData>
    </srw:record>
    <srw:record>
      <srw:recordData>
        <oai_dc:dc>
          <dc:identifier>https://gallica.bnf.fr/ark:/12148/btv1b100301371</dc:identifier>
          <dc:type>manuscrit</dc:type>
        </oai_dc:dc>
      </srw:recordData>
    </srw:record>
  </srw:records>
</srw:searchRetrieveResponse>"""

_SRU_XML_EMPTY = b"""<?xml version="1.0" encoding="UTF-8"?>
<srw:searchRetrieveResponse
    xmlns:srw="http://www.loc.gov/zing/srw/"
    xmlns:dc="http://purl.org/dc/elements/1.1/">
  <srw:records/>
</srw:searchRetrieveResponse>"""

_IIIF_MANIFEST_3_CANVASES = {
    "sequences": [
        {
            "canvases": [
                {"@id": "canvas/1", "images": [{"resource": {"@id": "img/1"}}]},
                {"@id": "canvas/2", "images": [{"resource": {"@id": "img/2"}}]},
                {"@id": "canvas/3", "images": [{"resource": {"@id": "img/3"}}]},
            ]
        }
    ]
}


# ── Tests de search_vinci_manuscripts ────────────────────────────────────────


class TestSearchVinciManuscripts:
    def _mock_response(self, content: bytes) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.content = content
        return resp

    @patch("htmir.collection.gallica.requests.get")
    def test_returns_ark_ids(self, mock_get):
        mock_get.return_value = self._mock_response(_SRU_XML_TWO_ARKS)
        arks = search_vinci_manuscripts(max_results=10, rate_limit=0)
        assert "btv1b10022860x" in arks
        assert "btv1b100301371" in arks

    @patch("htmir.collection.gallica.requests.get")
    def test_deduplicates_arks(self, mock_get):
        # Deux pages SRU retournant les mêmes ARK
        mock_get.return_value = self._mock_response(_SRU_XML_TWO_ARKS)
        arks = search_vinci_manuscripts(max_results=100, rate_limit=0)
        assert len(arks) == len(set(arks))

    @patch("htmir.collection.gallica.requests.get")
    def test_empty_results(self, mock_get):
        mock_get.return_value = self._mock_response(_SRU_XML_EMPTY)
        arks = search_vinci_manuscripts(max_results=10, rate_limit=0)
        assert arks == []

    @patch("htmir.collection.gallica.requests.get")
    def test_respects_max_results(self, mock_get):
        mock_get.return_value = self._mock_response(_SRU_XML_TWO_ARKS)
        arks = search_vinci_manuscripts(max_results=1, rate_limit=0)
        assert len(arks) <= 1

    @patch("htmir.collection.gallica.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("timeout")
        arks = search_vinci_manuscripts(max_results=10, rate_limit=0)
        assert arks == []


# ── Tests de fetch_iiif_manifest ─────────────────────────────────────────────


class TestFetchIiifManifest:
    @patch("htmir.collection.gallica.requests.get")
    def test_returns_dict(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _IIIF_MANIFEST_3_CANVASES
        mock_get.return_value = resp
        manifest = fetch_iiif_manifest("btv1b10022860x")
        assert isinstance(manifest, dict)
        assert "sequences" in manifest

    @patch("htmir.collection.gallica.requests.get")
    def test_calls_correct_url(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {}
        mock_get.return_value = resp
        fetch_iiif_manifest("btv1b10022860x")
        called_url = mock_get.call_args[0][0]
        assert "ark:/12148/btv1b10022860x/manifest.json" in called_url


# ── Tests de count_folios ────────────────────────────────────────────────────


class TestCountFolios:
    @patch("htmir.collection.gallica.fetch_iiif_manifest")
    def test_counts_canvases(self, mock_fetch):
        mock_fetch.return_value = _IIIF_MANIFEST_3_CANVASES
        assert count_folios("btv1b10022860x") == 3

    @patch("htmir.collection.gallica.fetch_iiif_manifest")
    def test_returns_zero_on_error(self, mock_fetch):
        mock_fetch.side_effect = Exception("not found")
        assert count_folios("invalid_ark") == 0


# ── Tests de GallicaFolio ────────────────────────────────────────────────────


class TestGallicaFolio:
    def test_folio_id(self):
        f = GallicaFolio(ark_id="btv1b10022860x", folio_idx=3)
        assert f.folio_id == "btv1b10022860x_f0003"

    def test_folio_id_padding(self):
        f = GallicaFolio(ark_id="abc123", folio_idx=42)
        assert f.folio_id == "abc123_f0042"
