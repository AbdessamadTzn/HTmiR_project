"""Tests du split corpus reproductible."""

from htmir.corpus.manifest import LineRecord
from htmir.corpus.split import assign_splits


def _records(n_pages: int = 10, lines_per_page: int = 2) -> list[LineRecord]:
    out = []
    for p in range(n_pages):
        for l in range(lines_per_page):
            lid = f"folio_{p:03d}_line_{l:02d}"
            out.append(
                LineRecord(
                    line_id=lid,
                    page_id=f"folio_{p:03d}",
                    image_path=f"data/lines/{lid}.png",
                    text="p[er]che la natura",
                    split="train",
                )
            )
    return out


def test_split_no_page_leakage():
    records = assign_splits(_records(), seed=42)
    test_pages = {r.page_id for r in records if r.split == "test"}
    train_pages = {r.page_id for r in records if r.split == "train"}
    assert test_pages.isdisjoint(train_pages)


def test_split_reproducible():
    a = assign_splits(_records(), seed=7)
    b = assign_splits(_records(), seed=7)
    assert [r.split for r in a] == [r.split for r in b]
