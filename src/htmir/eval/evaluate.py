"""Évaluation d'un modèle HTR : CER (Character Error Rate) et WER (Word Error Rate).

Deux usages complémentaires :

- Fonctions de métriques **pures** (distance de Levenshtein, CER, WER, agrégat
  corpus) — utilisables sur n'importe quelle liste de couples (référence,
  hypothèse), entièrement testables sans Kraken.
- Wrapper ``ketos test`` pour évaluer directement un ``.mlmodel`` sur un dataset
  Arrow et écrire un rapport JSON.

Le CER est la métrique principale du projet (cible : CER < 8 %).

Usage :
    python -m htmir.eval.evaluate --model htmir-french-13c_best.mlmodel \\
        --test-arrow data/catmus-french-13c/test.arrow
"""

import argparse
import json
import subprocess
from pathlib import Path

from htmir.training.train_kraken import _ketos_bin
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def levenshtein(a: list | str, b: list | str) -> int:
    """Distance d'édition de Levenshtein entre deux séquences.

    Args:
        a: Première séquence (chaîne ou liste de tokens).
        b: Seconde séquence.

    Returns:
        Nombre minimal d'insertions/suppressions/substitutions.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_error_rate(ref: str, hyp: str) -> float:
    """CER d'une ligne : distance caractère / longueur de la référence.

    Args:
        ref: Transcription de référence (ground truth).
        hyp: Transcription prédite.

    Returns:
        CER dans ``[0, +inf[`` (0 = parfait). Si ``ref`` est vide, retourne
        0.0 si ``hyp`` vide sinon 1.0.
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def word_error_rate(ref: str, hyp: str) -> float:
    """WER d'une ligne : distance au niveau des mots / nombre de mots de réf."""
    ref_w = ref.split()
    hyp_w = hyp.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return levenshtein(ref_w, hyp_w) / len(ref_w)


def corpus_metrics(pairs: list[tuple[str, str]]) -> dict:
    """Agrège CER et WER sur un corpus (micro-moyenne pondérée).

    Le CER/WER corpus est ``somme(distances) / somme(longueurs_référence)`` —
    plus correct qu'une moyenne des CER par ligne.

    Args:
        pairs: Liste de couples ``(référence, hypothèse)``.

    Returns:
        Dictionnaire ``{cer, wer, n_lines, n_chars, n_words}``.
    """
    tot_char_dist = tot_chars = 0
    tot_word_dist = tot_words = 0
    for ref, hyp in pairs:
        tot_char_dist += levenshtein(ref, hyp)
        tot_chars += len(ref)
        tot_word_dist += levenshtein(ref.split(), hyp.split())
        tot_words += len(ref.split())

    return {
        "cer": tot_char_dist / tot_chars if tot_chars else 0.0,
        "wer": tot_word_dist / tot_words if tot_words else 0.0,
        "n_lines": len(pairs),
        "n_chars": tot_chars,
        "n_words": tot_words,
    }


def build_ketos_test_cmd(model: Path, test_arrow: Path, device: str = "cpu") -> list[str]:
    """Construit la commande ``ketos test`` pour évaluer un modèle.

    Args:
        model: Chemin du ``.mlmodel`` à évaluer.
        test_arrow: Dataset Arrow de test.
        device: Périphérique (``"cpu"`` ou ``"cuda:0"``).

    Returns:
        Liste argv prête pour ``subprocess.run``.
    """
    return [
        _ketos_bin(),
        "-d", device,
        "test",
        "-m", str(model),
        "-f", "binary",
        str(test_arrow),
    ]


def run(model: Path, test_arrow: Path, device: str, report_path: Path) -> dict:
    """Évalue le modèle via ``ketos test`` et écrit un rapport JSON.

    Args:
        model: Modèle ``.mlmodel`` entraîné.
        test_arrow: Dataset Arrow de test.
        device: Périphérique d'inférence.
        report_path: Chemin du rapport JSON de sortie.

    Returns:
        Dictionnaire récapitulatif (sortie brute ketos + métadonnées).
    """
    cmd = build_ketos_test_cmd(model, test_arrow, device)
    logger.info(f"Évaluation : {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    report = {
        "model": str(model),
        "test_set": str(test_arrow),
        "ketos_output": proc.stdout,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Rapport écrit : {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue un modèle HTR Kraken (CER/WER) sur le test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-arrow", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, default=Path("eval_report.json"))
    args = parser.parse_args()

    run(args.model, args.test_arrow, args.device, args.report)


if __name__ == "__main__":
    main()
