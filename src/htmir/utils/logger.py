"""Logger commun à tous les modules HTmiR."""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré (format uniforme, sortie stdout).

    Args:
        name: Nom du logger (typiquement ``__name__``).

    Returns:
        Instance de :class:`logging.Logger` prête à l'emploi.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
