"""Tests unitaires pour htmir.eval.seg_eval."""

from pathlib import Path

import pytest

from htmir.eval.seg_eval import (
    baseline_to_bbox,
    match_and_iou,
    parse_alto_baselines,
    parse_kraken_baselines,
)


def test_baseline_to_bbox_horizontal():
    bl = [[100, 50], [300, 50]]
    bbox = baseline_to_bbox(bl)
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    assert min(xs) == 100 and max(xs) == 300
    assert min(ys) < 50 and max(ys) > 50  # marge verticale ajoutée


def test_match_and_iou_perfect():
    """GT = pred identiques → IoU=1.0."""
    bl = [[[0, 100], [200, 100]]]
    stats = match_and_iou(bl, bl)
    assert stats["mean_iou"] == pytest.approx(1.0, abs=1e-4)
    assert stats["pct_above_threshold"] == pytest.approx(1.0, abs=1e-4)


def test_match_and_iou_no_overlap():
    """Lignes complètement séparées verticalement → IoU≈0."""
    gt = [[[0, 0], [100, 0]]]
    pred = [[[0, 500], [100, 500]]]
    stats = match_and_iou(gt, pred)
    assert stats["mean_iou"] < 0.05


def test_match_and_iou_empty():
    stats = match_and_iou([], [[0, 0], [100, 0]])
    assert stats["mean_iou"] == 0.0
    assert stats["n_lines"] == 0


def test_parse_alto_baselines(tmp_path):
    """Parse un ALTO XML minimal avec une ligne et une baseline."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    xml_content = f"""<?xml version="1.0"?>
<alto xmlns="{ns}">
  <Description>
    <sourceImageInformation>
      <fileName>test_image.jpg</fileName>
    </sourceImageInformation>
  </Description>
  <Layout>
    <Page>
      <PrintSpace>
        <TextBlock>
          <TextLine ID="line_1" BASELINE="100 200 300 200">
            <String CONTENT="li rois de France" HPOS="100" VPOS="180" WIDTH="200" HEIGHT="40"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""
    xml_path = tmp_path / "test.chocomufin.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    result = parse_alto_baselines(xml_path)
    assert result["image"] == "test_image.jpg"
    assert len(result["baselines"]) == 1
    assert result["baselines"][0] == [[100, 200], [300, 200]]
    assert result["texts"][0] == "li rois de France"


def test_parse_kraken_baselines_missing(tmp_path):
    """Fichier absent → liste vide."""
    assert parse_kraken_baselines(tmp_path / "missing.json") == []


def test_parse_kraken_baselines(tmp_path):
    """Parse un JSON de sortie Kraken avec baselines."""
    import json

    data = {
        "lines": [
            {"baseline": [[10, 20], [30, 20]], "boundary": []},
            {"baseline": [[10, 50], [30, 50]], "boundary": []},
        ]
    }
    p = tmp_path / "seg.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    bls = parse_kraken_baselines(p)
    assert len(bls) == 2
    assert bls[0] == [[10, 20], [30, 20]]
