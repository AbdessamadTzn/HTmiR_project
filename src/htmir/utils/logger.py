"""Logger centralisé pour le projet HTR."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Crée et retourne un logger configuré.

    Args:
        name: Nom du logger (typiquement __name__).
        log_file: Chemin optionnel vers un fichier de log.

    Returns:
        Logger configuré avec handler console et optionnellement fichier.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline démarré")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
