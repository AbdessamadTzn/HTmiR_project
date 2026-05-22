"""Validation du data contract JSON pour la sortie HTR vers le module NLP."""

import json
import hashlib
from pathlib import Path
from typing import Any

import jsonschema
from htmir.utils.logger import get_logger

logger = get_logger(__name__)

DATA_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "HTR Output Data Contract",
    "type": "object",
    "required": ["metadata", "pages"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["corpus", "model", "cer_global", "wer_global", "created_at"],
            "properties": {
                "corpus": {"type": "string"},
                "model": {"type": "string"},
                "cer_global": {"type": "number", "minimum": 0},
                "wer_global": {"type": "number", "minimum": 0},
                "created_at": {"type": "string"},
                "train_hash_sha256": {"type": "string"},
            },
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["page_id", "image_filename", "lines"],
                "properties": {
                    "page_id": {"type": "string"},
                    "image_filename": {"type": "string"},
                    "page_xml_path": {"type": "string"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["line_id", "text", "confidence", "needs_review", "polygon"],
                            "properties": {
                                "line_id": {"type": "string"},
                                "text": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "needs_review": {"type": "boolean"},
                                "polygon": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                                "baseline": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def validate_output(data: dict) -> None:
    """Valide le JSON de sortie HTR contre le data contract.

    Args:
        data: Dictionnaire Python représentant la sortie HTR.

    Raises:
        jsonschema.ValidationError: Si le JSON ne respecte pas le schéma.

    Example:
        >>> validate_output(output_dict)
    """
    jsonschema.validate(instance=data, schema=DATA_CONTRACT_SCHEMA)
    logger.info("Data contract validé avec succès.")


def save_output(data: dict, output_path: Path) -> None:
    """Valide et sauvegarde le JSON de sortie HTR.

    Args:
        data: Dictionnaire Python de sortie HTR validé.
        output_path: Chemin du fichier JSON de sortie.

    Example:
        >>> save_output(output, Path("dataset_nlp/output.json"))
    """
    validate_output(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Sortie HTR sauvegardée : {output_path}")


def sha256_file(path: Path) -> str:
    """Calcule le SHA-256 d'un fichier pour garantir la non-contamination.

    Args:
        path: Chemin du fichier.

    Returns:
        Hash SHA-256 en hexadécimal.

    Example:
        >>> h = sha256_file(Path("data/test.json"))
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
