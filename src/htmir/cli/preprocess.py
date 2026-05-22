"""CLI : prétraitement batch des images de manuscrits."""

import argparse
from pathlib import Path

from htmir.preprocessing.pipeline import PreprocessingConfig, batch_preprocess
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prétraitement d'images de manuscrits (deskew, CLAHE, Sauvola).")
    parser.add_argument("--input", type=Path, required=True, help="Dossier d'images sources")
    parser.add_argument("--output", type=Path, required=True, help="Dossier de sortie (binaire PNG)")
    parser.add_argument("--no-deskew", action="store_true", help="Désactiver la correction d'inclinaison")
    args = parser.parse_args()

    config = PreprocessingConfig(auto_deskew=not args.no_deskew)
    paths = batch_preprocess(args.input, args.output, config=config)
    logger.info(f"{len(paths)} image(s) écrite(s) dans {args.output}")


if __name__ == "__main__":
    main()
