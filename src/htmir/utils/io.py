"""Utilitaires I/O (manifeste d'expériences, journal)."""

import json
from datetime import datetime, timezone
from pathlib import Path


def append_experiment_journal(
    entry: dict,
    path: Path = Path("experiments/journal.jsonl"),
) -> None:
    """Ajoute une ligne au journal d'expériences (reproductibilité)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
