"""Évaluation CER/WER + bootstrap + McNemar sur un manifeste."""

import argparse
import json
from pathlib import Path

from htmir.evaluation.metrics import corpus_cer, corpus_wer
from htmir.evaluation.bootstrap import bootstrap_cer
from htmir.evaluation.mcnemar import mcnemar_test
from htmir.corpus.manifest import load_manifest
from htmir.utils.logger import get_logger

logger = get_logger(__name__)

CER_VALIDATION = 0.15
CER_EXCELLENCE = 0.08
WER_VALIDATION = 0.25
WER_EXCELLENCE = 0.15


def evaluate_predictions(
    references: list[str],
    hypotheses: list[str],
    alt_hypotheses: list[str] | None = None,
    n_bootstrap: int = 500,
) -> dict:
    """Calcule les métriques du brief sur des listes alignées."""
    if len(references) != len(hypotheses):
        raise ValueError("references et hypotheses doivent avoir la même longueur")

    cer = corpus_cer(hypotheses, references)
    wer = corpus_wer(hypotheses, references)
    result = {
        "cer": cer,
        "wer": wer,
        "cer_bootstrap": bootstrap_cer(hypotheses, references, n_iterations=n_bootstrap),
        "passes_validation": cer < CER_VALIDATION and wer < WER_VALIDATION,
        "passes_excellence": cer < CER_EXCELLENCE and wer < WER_EXCELLENCE,
    }

    if alt_hypotheses and len(alt_hypotheses) == len(hypotheses):
        errors_a = [compute_line_error(h, r) for h, r in zip(hypotheses, references)]
        errors_b = [compute_line_error(h, r) for h, r in zip(alt_hypotheses, references)]
        result["mcnemar"] = mcnemar_test(errors_a, errors_b)

    return result


def compute_line_error(hyp: str, ref: str) -> bool:
    """True si la ligne est considérée en erreur (CER ligne > 0)."""
    from htmir.evaluation.metrics import compute_cer
    return compute_cer(hyp, ref) > 0


def evaluate_from_files(
    manifest_path: Path,
    predictions_path: Path,
    split: str = "test",
) -> dict:
    """Évalue un fichier predictions.jsonl {line_id, text} contre le manifeste."""
    records = [r for r in load_manifest(manifest_path) if r.split == split]
    ref_by_id = {r.line_id: r.text for r in records}

    preds: dict[str, str] = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds[row["line_id"]] = row["text"]

    ids = [r.line_id for r in records if r.line_id in preds]
    refs = [ref_by_id[i] for i in ids]
    hyps = [preds[i] for i in ids]
    return evaluate_predictions(refs, hyps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue des prédictions HTR (CER, WER, bootstrap).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("experiments/eval_last.json"))
    args = parser.parse_args()

    metrics = evaluate_from_files(args.manifest, args.predictions, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques : CER={metrics['cer']:.4f}, WER={metrics['wer']:.4f}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
