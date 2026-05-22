"""Manifeste de lignes pour entraînement et évaluation HTR."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Literal

Split = Literal["train", "val", "test"]


@dataclass
class LineRecord:
    """Une ligne de manuscrit dans le manifeste.

    Args:
        line_id: Identifiant unique.
        page_id: Identifiant de page (folio).
        image_path: Chemin relatif vers l'image de ligne ou de page.
        text: Transcription de référence.
        split: Partition train/val/test.
        source: Provenance (ex. vinci-codex-atlanticus).
        mirror_normalized: True si l'image a été retournée pour lecture.
    """

    line_id: str
    page_id: str
    image_path: str
    text: str
    split: Split
    source: str = "vinci-notebooks"
    mirror_normalized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: Path) -> list[LineRecord]:
    """Charge un manifeste JSONL ou JSON (liste)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(LineRecord(**json.loads(line)))
        return records
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["lines"] if isinstance(data, dict) else data
    return [LineRecord(**item) for item in items]


def save_manifest(records: list[LineRecord], path: Path) -> None:
    """Sauvegarde le manifeste en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lines": [r.to_dict() for r in records]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
