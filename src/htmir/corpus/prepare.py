"""Préparation du corpus Vinci : manifeste, splits, hash du test scellé."""

import argparse
from pathlib import Path

import yaml

from htmir.corpus.manifest import LineRecord, load_manifest, save_manifest
from htmir.corpus.split import assign_splits
from htmir.aggregation.data_contract import sha256_file
from htmir.utils.logger import get_logger
from htmir.utils.seeds import fixer_seeds

logger = get_logger(__name__)


def build_manifest_from_annotations(
    annotations_dir: Path,
    image_root: Path,
    source: str = "vinci-notebooks",
) -> list[LineRecord]:
    """Construit un manifeste depuis des paires image + .txt (une ligne par fichier).

    Structure attendue :
        annotations_dir/
            folio_001_line_001.txt
        image_root/
            folio_001_line_001.png
    """
    records: list[LineRecord] = []
    for txt_path in sorted(annotations_dir.glob("*.txt")):
        stem = txt_path.stem
        page_id = "_".join(stem.split("_")[:2]) if "_" in stem else stem
        img_candidates = list(image_root.glob(f"{stem}.*"))
        if not img_candidates:
            logger.warning(f"Image manquante pour {stem}")
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        records.append(
            LineRecord(
                line_id=stem,
                page_id=page_id,
                image_path=str(img_candidates[0].relative_to(image_root.parent)),
                text=text,
                split="train",
                source=source,
            )
        )
    return records


def prepare_corpus(config_path: Path) -> dict:
    """Exécute la préparation selon configs/corpus.yaml."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fixer_seeds(cfg.get("seed", 42))
    manifest_in = Path(cfg.get("manifest_in", "data/annotations/manifest.json"))
    manifest_out = Path(cfg.get("manifest_out", "data/processed/manifest.json"))
    test_hash_out = Path(cfg.get("test_hash_out", "data/processed/test_set.sha256"))

    if manifest_in.exists():
        records = load_manifest(manifest_in)
    else:
        ann = Path(cfg.get("annotations_dir", "data/annotations/lines"))
        img = Path(cfg.get("image_root", "data/raw"))
        records = build_manifest_from_annotations(ann, img, source=cfg.get("corpus_id", "vinci-notebooks"))

    splits = cfg.get("splits", {})
    records = assign_splits(
        records,
        train_ratio=splits.get("train", 0.8),
        val_ratio=splits.get("val", 0.1),
        seed=cfg.get("seed", 42),
        by_page=cfg.get("split_by_page", True),
    )
    save_manifest(records, manifest_out)

    test_lines = [r for r in records if r.split == "test"]
    test_manifest = manifest_out.parent / "test_manifest.json"
    save_manifest(test_lines, test_manifest)
    test_hash = sha256_file(test_manifest)
    test_hash_out.parent.mkdir(parents=True, exist_ok=True)
    test_hash_out.write_text(test_hash + "\n", encoding="utf-8")

    stats = {
        "total": len(records),
        "train": sum(1 for r in records if r.split == "train"),
        "val": sum(1 for r in records if r.split == "val"),
        "test": len(test_lines),
        "test_hash_sha256": test_hash,
    }
    logger.info(f"Corpus préparé : {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare le corpus HTR (manifeste + test scellé).")
    parser.add_argument("--config", type=Path, default=Path("configs/corpus_vinci.yaml"))
    args = parser.parse_args()
    prepare_corpus(args.config)


if __name__ == "__main__":
    main()
