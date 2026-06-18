"""Tests du data contract (htmir.nlp.data_contract)."""

import pytest

from htmir.nlp.data_contract import (
    alto_to_contract,
    validate_contract,
)

_ALTO = """<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description><sourceImageInformation><fileName>p.png</fileName></sourceImageInformation></Description>
  <Layout><Page WIDTH="1000" HEIGHT="2000"><PrintSpace>
    <TextBlock>
      <TextLine ID="l0" BASELINE="10 50 200 50">
        <Shape><Polygon POINTS="10 20 200 20 200 60 10 60"/></Shape>
        <String CONTENT="li" WC="0.99"><Glyph CONTENT="l" GC="0.99"/><Glyph CONTENT="i" GC="0.99"/></String>
        <SP/>
        <String CONTENT="rois" WC="0.95"><Glyph CONTENT="r" GC="0.9"/><Glyph CONTENT="o" GC="0.95"/><Glyph CONTENT="i" GC="0.95"/><Glyph CONTENT="s" GC="0.95"/></String>
      </TextLine>
      <TextLine ID="l1" BASELINE="10 100 80 100">
        <Shape><Polygon POINTS="10 80 80 80 80 120 10 120"/></Shape>
        <String CONTENT="xx" WC="0.40"><Glyph CONTENT="x" GC="0.4"/><Glyph CONTENT="x" GC="0.4"/></String>
      </TextLine>
    </TextBlock>
  </PrintSpace></Page></Layout>
</alto>"""


def test_contract_basic_structure():
    c = alto_to_contract(_ALTO, model="m")
    assert c["source_image"] == "p.png"
    assert c["model"] == "m"
    assert c["page"] == {"width": 1000, "height": 2000}
    assert len(c["lines"]) == 2


def test_contract_text_and_confidences():
    c = alto_to_contract(_ALTO)
    line0 = c["lines"][0]
    assert line0["text"] == "li rois"
    assert len(line0["char_confidences"]) == 6  # l,i,r,o,i,s
    assert line0["polygon"][0] == [10, 20]
    assert line0["baseline"] == [[10, 50], [200, 50]]


def test_needs_review_threshold():
    c = alto_to_contract(_ALTO, review_threshold=0.7)
    # ligne 0 : conf élevée → pas de review ; ligne 1 : conf 0.4 → review
    assert c["lines"][0]["needs_review"] is False
    assert c["lines"][1]["needs_review"] is True


def test_validate_contract_ok():
    c = alto_to_contract(_ALTO)
    validate_contract(c)  # ne doit pas lever


def test_validate_contract_rejects_bad():
    with pytest.raises(ValueError):
        validate_contract({"lines": [{"id": "x"}]})  # source_image manquant + ligne incomplète
