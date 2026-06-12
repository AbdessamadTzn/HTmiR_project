"""Chargement et agrégation des données pour le dashboard Streamlit.

Fonctions **pures** (sans Streamlit) qui lisent les artefacts du projet et
renvoient des structures simples — donc entièrement testables :

- ``load_dataset_manifest`` : le manifeste produit par ``prepare_catmus``.
- ``compute_length_stats``  : distribution longueurs de lignes (caractères/mots).
- ``sample_lines``          : échantillon (image, transcription) pour aperçu.
- ``parse_ketos_log``       : extrait les métriques par epoch du log ``ketos train``.
- ``load_training_metrics`` : lit un CSV de métriques d'entraînement.
- ``load_eval_report``      : le rapport JSON d'évaluation.
"""

import json
import re
from pathlib import Path


def load_dataset_manifest(path: Path) -> dict:
    """Charge ``dataset_manifest.json`` (filtre + comptes par split).

    Returns:
        Le dictionnaire du manifeste, ou ``{}`` si absent.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_length_stats(split_dir: Path) -> dict:
    """Calcule les distributions de longueur sur les ``*.gt.txt`` d'un split.

    Args:
        split_dir: Répertoire contenant les ``line_*.gt.txt``.

    Returns:
        ``{n_lines, char_lengths, word_counts}`` où les deux listes sont
        parallèles (une valeur par ligne).
    """
    split_dir = Path(split_dir)
    char_lengths: list[int] = []
    word_counts: list[int] = []
    for txt in sorted(split_dir.glob("*.gt.txt")):
        content = txt.read_text(encoding="utf-8").strip()
        char_lengths.append(len(content))
        word_counts.append(len(content.split()))
    return {
        "n_lines": len(char_lengths),
        "char_lengths": char_lengths,
        "word_counts": word_counts,
    }


def sample_lines(split_dir: Path, n: int = 8) -> list[dict]:
    """Retourne un échantillon de couples (image, transcription) d'un split.

    Args:
        split_dir: Répertoire des lignes.
        n: Nombre d'exemples à renvoyer.

    Returns:
        Liste de ``{image_path, text}`` (au plus ``n`` éléments).
    """
    split_dir = Path(split_dir)
    out: list[dict] = []
    for png in sorted(split_dir.glob("*.png"))[:n]:
        txt = png.with_suffix("").with_suffix(".gt.txt")
        text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        out.append({"image_path": str(png), "text": text})
    return out


# Pattern du rapport d'accuracy de Kraken (ketos train), ex :
#   "Accuracy report (3) 0.8921 12345 1331"
#                    ↑epoch ↑char_accuracy ↑chars ↑errors
_ACC_RE = re.compile(
    r"Accuracy report \((\d+)\)\s+([\d.]+)(?:\s+(\d+)\s+(\d+))?"
)


def parse_ketos_log(log_text: str) -> list[dict]:
    """Extrait les métriques par epoch depuis la sortie texte de ``ketos train``.

    Args:
        log_text: Contenu stdout/stderr capturé pendant l'entraînement.

    Returns:
        Liste ``[{epoch, accuracy, cer}, ...]`` triée par epoch. Le CER est
        dérivé de ``1 - accuracy`` (Kraken reporte l'accuracy caractère).
    """
    rows: list[dict] = []
    for m in _ACC_RE.finditer(log_text):
        epoch = int(m.group(1))
        accuracy = float(m.group(2))
        rows.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "cer": round(1.0 - accuracy, 4),
        })
    rows.sort(key=lambda r: r["epoch"])
    return rows


def load_training_metrics(path: Path) -> list[dict]:
    """Lit un CSV de métriques d'entraînement (epoch, accuracy, cer, ...).

    Args:
        path: Chemin du CSV (colonnes en en-tête).

    Returns:
        Liste de dicts (une par ligne), ou ``[]`` si le fichier est absent.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for r in reader:
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_eval_report(path: Path) -> dict:
    """Charge le rapport JSON d'évaluation (sortie de ``evaluate.run``)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_early_stopping(metrics: list[dict], patience: int) -> dict:
    """Analyse l'early-stopping à partir des métriques par epoch.

    Identifie le meilleur epoch (CER validation minimal — c'est le modèle
    conservé), l'epoch où la patience aurait déclenché l'arrêt, et la liste
    des epochs « sans amélioration ».

    Args:
        metrics: Liste ``[{epoch, cer, ...}]`` triée par epoch.
        patience: Nombre d'epochs sans amélioration toléré (``--lag``).

    Returns:
        ``{best_epoch, best_cer, stop_epoch, stalled_epochs}``. Les champs
        sont ``None`` si ``metrics`` est vide.
    """
    if not metrics:
        return {"best_epoch": None, "best_cer": None,
                "stop_epoch": None, "stalled_epochs": []}

    best_cer = float("inf")
    best_epoch = None
    stalls_since_best = 0
    stop_epoch = None
    stalled_epochs: list[int] = []

    for row in metrics:
        cer = row.get("cer", float("inf"))
        epoch = row["epoch"]
        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch
            stalls_since_best = 0
        else:
            stalls_since_best += 1
            stalled_epochs.append(epoch)
            if stop_epoch is None and stalls_since_best >= patience:
                stop_epoch = epoch

    return {
        "best_epoch": best_epoch,
        "best_cer": best_cer,
        "stop_epoch": stop_epoch,
        "stalled_epochs": stalled_epochs,
    }


def char_frequency(split_dir: Path, only_special: bool = False) -> dict[str, int]:
    """Compte la fréquence des caractères sur les transcriptions d'un split.

    Args:
        split_dir: Répertoire des ``*.gt.txt``.
        only_special: Si ``True``, ne garde que les caractères non-ASCII
            (abréviations médiévales : ``⁊``, ``ẽ``, ``õ``, ``ꝯ``…).

    Returns:
        Dictionnaire ``{caractère: occurrences}`` trié par fréquence desc.
    """
    from collections import Counter

    counter: Counter = Counter()
    for txt in Path(split_dir).glob("*.gt.txt"):
        counter.update(txt.read_text(encoding="utf-8").strip())

    items = counter.items()
    if only_special:
        items = [(c, n) for c, n in items if ord(c) > 127]
    return dict(sorted(items, key=lambda kv: -kv[1]))


def load_predictions(path: Path) -> list[dict]:
    """Charge un CSV de prédictions ligne à ligne (image_path, ref, hyp).

    Args:
        path: CSV avec au moins les colonnes ``ref`` et ``hyp`` (et
            optionnellement ``image_path``).

    Returns:
        Liste de dicts, ou ``[]`` si absent.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def worst_predictions(predictions: list[dict], n: int = 10) -> list[dict]:
    """Retourne les ``n`` lignes au CER le plus élevé (diagnostic qualitatif).

    Args:
        predictions: Liste ``[{ref, hyp, image_path?}, ...]``.
        n: Nombre de pires lignes à renvoyer.

    Returns:
        Sous-liste enrichie d'un champ ``cer``, triée par CER décroissant.
    """
    from htmir.eval.evaluate import char_error_rate

    scored = []
    for p in predictions:
        ref = p.get("ref", "")
        hyp = p.get("hyp", "")
        scored.append({**p, "cer": char_error_rate(ref, hyp)})
    scored.sort(key=lambda r: r["cer"], reverse=True)
    return scored[:n]
