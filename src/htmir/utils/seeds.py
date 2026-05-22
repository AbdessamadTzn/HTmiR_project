"""Gestion des seeds pour la reproductibilité complète."""

import random
import numpy as np
import torch


def fixer_seeds(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléatoire du projet.

    Args:
        seed: Valeur du seed. Défaut : 42.

    Example:
        >>> fixer_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
