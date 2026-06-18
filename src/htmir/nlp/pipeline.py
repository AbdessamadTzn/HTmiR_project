"""Pipeline NLP : data contract → CER avant/après sur une page ou un manuscrit.

Le CER est mesuré contre la vérité terrain HTRomance (ALTO ``.chocomufin.xml``)
selon trois variantes, du plus strict au plus pertinent :

- ``cer_raw``        : toutes les lignes Kraken vs GT (brut) ;
- ``cer_reviewed``   : après filtrage des lignes ``needs_review`` (confiance) ;
- ``cer_normalized`` : règles CER appliquées des deux côtés (u/v, i/j, NFC…).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from htmir.eval.evaluate import corpus_metrics
from htmir.nlp.data_contract import alto_to_contract
from htmir.nlp.normalize import (
    count_residual_abbreviations,
    normalize_for_cer,
)

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def gt_lines_from_alto(path: Path) -> list[str]:
    """Extrait les lignes de texte d'un ALTO de vérité terrain."""
    root = ET.parse(path).getroot()
    out = []
    for tl in root.iter(f"{{{_ALTO_NS}}}TextLine"):
        words = [s.attrib.get("CONTENT", "")
                 for s in tl.iter(f"{{{_ALTO_NS}}}String")
                 if s.attrib.get("CONTENT")]
        if words:
            out.append(" ".join(words))
    return out


def _page_cer(hyp_text: str, gt_text: str) -> float:
    return corpus_metrics([(gt_text, hyp_text)])["cer"]


def evaluate_page(hyp_alto: str, gt_alto_path: Path, model: str | None = None) -> dict:
    """Évalue une page : construit le contract et calcule les CER avant/après.

    Args:
        hyp_alto: ALTO XML produit par Kraken (hypothèse).
        gt_alto_path: Chemin de l'ALTO de vérité terrain.
        model: Identifiant du modèle (informatif).

    Returns:
        Dictionnaire de métriques + le contract sous ``"contract"``.
    """
    contract = alto_to_contract(hyp_alto, model=model)
    all_lines = contract["lines"]
    kept = [ln for ln in all_lines if not ln["needs_review"]]
    gt = gt_lines_from_alto(gt_alto_path)

    gt_text = " ".join(gt)
    hyp_all = " ".join(ln["text"] for ln in all_lines)
    hyp_kept = " ".join(ln["text"] for ln in kept)

    cer_raw = _page_cer(hyp_all, gt_text)
    cer_reviewed = _page_cer(hyp_kept, gt_text)
    cer_norm = _page_cer(normalize_for_cer(hyp_kept), normalize_for_cer(gt_text))

    return {
        "contract": contract,
        "n_lines": len(all_lines),
        "n_lines_kept": len(kept),
        "n_lines_gt": len(gt),
        "n_needs_review": len(all_lines) - len(kept),
        "mean_confidence": round(
            sum(ln["mean_confidence"] for ln in all_lines) / len(all_lines), 4
        ) if all_lines else 0.0,
        "residual_abbreviations": count_residual_abbreviations(hyp_all),
        "cer_raw": round(cer_raw, 4),
        "cer_reviewed": round(cer_reviewed, 4),
        "cer_normalized": round(cer_norm, 4),
    }


def aggregate_pages(page_results: list[dict]) -> dict:
    """Agrège les métriques de plusieurs pages (moyenne pondérée simple)."""
    n = len(page_results)
    if n == 0:
        return {}
    def avg(k):
        return round(sum(p[k] for p in page_results) / n, 4)
    return {
        "n_pages": n,
        "n_lines": sum(p["n_lines"] for p in page_results),
        "n_needs_review": sum(p["n_needs_review"] for p in page_results),
        "mean_confidence": avg("mean_confidence"),
        "needs_review_rate": round(
            sum(p["n_needs_review"] for p in page_results)
            / sum(p["n_lines"] for p in page_results), 4
        ),
        "cer_raw": avg("cer_raw"),
        "cer_reviewed": avg("cer_reviewed"),
        "cer_normalized": avg("cer_normalized"),
    }
