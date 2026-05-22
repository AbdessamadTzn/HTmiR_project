"""Découpage reproductible train / val / test."""

from pathlib import Path
import random

from htmir.corpus.manifest import LineRecord, Split
from htmir.utils.seeds import fixer_seeds


def assign_splits(
    records: list[LineRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    by_page: bool = True,
) -> list[LineRecord]:
    """Assigne les splits sans fuite page entre train et test.

    Args:
        records: Lignes sans champ split ou à réassigner.
        train_ratio: Proportion train.
        val_ratio: Proportion validation (reste = test).
        seed: Graine pour reproductibilité.
        by_page: Groupe par page_id pour éviter la fuite.

    Returns:
        Nouvelle liste avec splits mis à jour.
    """
    fixer_seeds(seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio doit être <= 1")

    keys = [r.page_id if by_page else r.line_id for r in records]
    unique_keys = sorted(set(keys))
    random.shuffle(unique_keys)

    n = len(unique_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])
    test_keys = set(unique_keys[n_train + n_val :])

    def _split_for(key: str) -> Split:
        if key in train_keys:
            return "train"
        if key in val_keys:
            return "val"
        return "test"

    out: list[LineRecord] = []
    for rec in records:
        key = rec.page_id if by_page else rec.line_id
        out.append(
            LineRecord(
                line_id=rec.line_id,
                page_id=rec.page_id,
                image_path=rec.image_path,
                text=rec.text,
                split=_split_for(key),
                source=rec.source,
                mirror_normalized=rec.mirror_normalized,
            )
        )
    return out
