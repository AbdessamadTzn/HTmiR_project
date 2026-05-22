"""Construction du JSON de sortie conforme au data contract (Volet NLP)."""

from datetime import datetime, timezone
from pathlib import Path

from htmir.aggregation.confidence import needs_review, calibrate_confidence
from htmir.aggregation.data_contract import save_output
from htmir.segmentation.lines import LineSegment


def build_page_output(
    page_id: str,
    image_filename: str,
    page_xml_path: str | None,
    lines: list[dict],
) -> dict:
    """Construit l'entrée pages[] du data contract."""
    return {
        "page_id": page_id,
        "image_filename": image_filename,
        "page_xml_path": page_xml_path or "",
        "lines": lines,
    }


def line_dict_from_prediction(
    segment: LineSegment,
    text: str,
    confidence: float,
    hyp_alt: str | None = None,
    degraded: bool = False,
    cer_val: float | None = None,
) -> dict:
    """Formate une ligne pour le JSON final."""
    conf = calibrate_confidence(confidence, cer_val)
    return {
        "line_id": segment.line_id,
        "text": text,
        "confidence": round(conf, 4),
        "needs_review": needs_review(conf, text, hyp_alt, degraded),
        "polygon": segment.polygon,
        "baseline": segment.baseline,
    }


def build_dataset_output(
    corpus: str,
    model: str,
    pages: list[dict],
    cer_global: float,
    wer_global: float,
    train_hash_sha256: str = "",
) -> dict:
    """Assemble le document racine du data contract."""
    return {
        "metadata": {
            "corpus": corpus,
            "model": model,
            "cer_global": cer_global,
            "wer_global": wer_global,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_hash_sha256": train_hash_sha256,
            "writing_system": "latin_mirror_normalized",
            "language": "it-renaissance-autograph",
        },
        "pages": pages,
    }


def export_dataset_nlp(data: dict, path: Path) -> None:
    """Valide et écrit dataset_nlp/output.json."""
    save_output(data, path)
