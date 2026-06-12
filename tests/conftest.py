"""Fixtures partagées pour les tests HTmiR."""

import io

import pytest


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """Octets d'une petite image PNG valide (10x10 blanc)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def catmus_parquet(tmp_path, tiny_png_bytes):
    """Crée un parquet local imitant la structure CATMuS/medieval.

    Colonnes : ``im`` (struct bytes/path), ``text``, ``language``, ``century``.
    Contient un mélange de lignes French/13e (à garder) et d'autres (à filtrer).

    Returns:
        Chemin (str) du fichier parquet généré.
    """
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [
        # (text, language, century) — image identique pour simplifier
        ("ki tant estoit preuz et vaillanz", "French", 13),   # garder
        ("car molt par ert de grant noblece", "French", 13),  # garder
        ("   ", "French", 13),                                # texte vide → exclu
        ("de bono regimine", "Latin", 13),                    # mauvaise langue → exclu
        ("au tens que li rois fu en France", "French", 14),   # mauvais siècle → exclu
        ("et puis sen ala en sa contree", "French", 13),      # garder
    ]

    im_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    im_values = [{"bytes": tiny_png_bytes, "path": f"line{i}.png"} for i in range(len(rows))]

    table = pa.table(
        {
            "im": pa.array(im_values, type=im_struct),
            "text": pa.array([r[0] for r in rows]),
            "language": pa.array([r[1] for r in rows]),
            "century": pa.array([r[2] for r in rows], type=pa.int8()),
        }
    )
    path = tmp_path / "catmus_fixture.parquet"
    pq.write_table(table, path)
    return str(path)
