"""Évaluation de la segmentation Kraken sur des pages HTRomance (ALTO XML).

Pipeline :
  1. parse_alto_baselines()  → lit les baselines GT depuis le XML ALTO
  2. run_kraken_segment()    → lance `kraken segment` sur l'image JPG
  3. parse_kraken_baselines()→ extrait les baselines prédites (JSON ketos)
  4. match_and_iou()         → apparie GT ↔ prédictions par distance, calcule IoU
  5. evaluate_manuscript()   → agrège sur toutes les pages d'un manuscrit
"""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from htmir.eval.evaluate import polygon_iou
from htmir.training.train_kraken import _ketos_bin
from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_alto_baselines(xml_path: Path) -> dict:
    """Extrait les baselines et le nom d'image d'un fichier ALTO XML HTRomance.

    Returns:
        ``{"image": str, "baselines": [[x0,y0, x1,y1, ...], ...], "texts": [str, ...]}``.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _ALTO_NS

    src = root.find(f".//{{{ns}}}fileName")
    image_name = src.text.strip() if src is not None and src.text else xml_path.stem + ".jpg"

    baselines = []
    texts = []
    for line in root.iter(f"{{{ns}}}TextLine"):
        bl_str = line.attrib.get("BASELINE", "")
        if not bl_str:
            continue
        coords = list(map(int, bl_str.split()))
        pts = [[coords[i], coords[i + 1]] for i in range(0, len(coords) - 1, 2)]
        baselines.append(pts)

        string = line.find(f"{{{ns}}}String")
        texts.append(string.attrib.get("CONTENT", "") if string is not None else "")

    return {"image": image_name, "baselines": baselines, "texts": texts}


def baseline_to_bbox(baseline: list[list[int]]) -> list[list[int]]:
    """Convertit une baseline en bounding-box polygon (±20px verticalement)."""
    margin = 20
    xs = [p[0] for p in baseline]
    ys = [p[1] for p in baseline]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys) - margin, max(ys) + margin
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ── Kraken segmentation ──────────────────────────────────────────────────────

def run_kraken_segment(image_path: Path, output_json: Path) -> bool:
    """Lance `kraken segment` sur une image et écrit le résultat JSON.

    Returns:
        True si succès.
    """
    # Priorité au kraken HTR du venv (évite /usr/bin/kraken bio-informatique)
    venv_kraken = Path(__file__).resolve().parents[4] / ".venv/bin/kraken"
    kraken = str(venv_kraken) if venv_kraken.exists() else _ketos_bin().replace("ketos", "kraken")

    cmd = [kraken, "-i", str(image_path), str(output_json), "segment", "-bl"]
    logger.info(f"Segmentation : {image_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Kraken segment échoué sur {image_path.name}: {exc}")
        return False


def parse_kraken_baselines(json_path: Path) -> list[list[list[int]]]:
    """Extrait les baselines prédites depuis la sortie JSON de `kraken segment`."""
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    baselines = []
    for line in data.get("lines", []):
        bl = line.get("baseline")
        if bl:
            baselines.append([[int(p[0]), int(p[1])] for p in bl])
    return baselines


# ── Appariement & IoU ────────────────────────────────────────────────────────

def match_and_iou(
    gt_baselines: list[list[list[int]]],
    pred_baselines: list[list[list[int]]],
    threshold: float = 0.75,
) -> dict:
    """Apparie GT ↔ prédictions par matching greedy (meilleur IoU en premier).

    Chaque GT est apparié à la prédiction non encore utilisée qui maximise l'IoU.
    Les GT sans prédiction correspondante contribuent avec IoU=0.

    Returns:
        ``{mean_iou, pct_above_threshold, threshold, n_lines}``.
    """
    if not gt_baselines or not pred_baselines:
        return {"mean_iou": 0.0, "pct_above_threshold": 0.0, "threshold": threshold, "n_lines": len(gt_baselines)}

    gt_polys = [baseline_to_bbox(bl) for bl in gt_baselines]
    pred_polys = [baseline_to_bbox(bl) for bl in pred_baselines]

    # matrice IoU complète
    iou_matrix = [
        [polygon_iou(gt_polys[i], pred_polys[j]) for j in range(len(pred_polys))]
        for i in range(len(gt_polys))
    ]

    matched_pred = set()
    ious: list[float] = []
    for i in range(len(gt_polys)):
        best_iou = 0.0
        best_j = -1
        for j in range(len(pred_polys)):
            if j not in matched_pred and iou_matrix[i][j] > best_iou:
                best_iou = iou_matrix[i][j]
                best_j = j
        if best_j >= 0:
            matched_pred.add(best_j)
        ious.append(best_iou)

    n = len(ious)
    mean_iou = sum(ious) / n if n else 0.0
    pct = sum(1 for v in ious if v >= threshold) / n if n else 0.0
    return {
        "mean_iou": round(mean_iou, 4),
        "pct_above_threshold": round(pct, 4),
        "threshold": threshold,
        "n_lines": n,
    }


# ── Évaluation d'un manuscrit ────────────────────────────────────────────────

def evaluate_manuscript(manuscript_dir: Path, tmp_dir: Path, threshold: float = 0.75) -> dict:
    """Évalue la segmentation Kraken sur toutes les pages d'un manuscrit.

    Args:
        manuscript_dir: Dossier contenant ``*.chocomufin.xml`` + ``*.jpg``.
        tmp_dir: Dossier temporaire pour les JSON Kraken.
        threshold: Seuil IoU (défaut 0.75).

    Returns:
        Statistiques agrégées ``{manuscript, pages, mean_iou, pct_above_threshold, n_lines}``.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(manuscript_dir.glob("*.chocomufin.xml"))
    if not xml_files:
        return {}

    page_ious: list[float] = []
    page_pcts: list[float] = []
    total_lines = 0

    for xml_path in xml_files:
        gt = parse_alto_baselines(xml_path)
        image_path = xml_path.parent / gt["image"]
        if not image_path.exists():
            logger.warning(f"Image absente : {image_path}")
            continue

        out_json = tmp_dir / (xml_path.stem + "_seg.json")
        ok = run_kraken_segment(image_path, out_json)
        if not ok:
            continue

        pred_bls = parse_kraken_baselines(out_json)
        gt_bls = gt["baselines"]
        page_stats = match_and_iou(gt_bls, pred_bls, threshold)
        page_ious.append(page_stats.get("mean_iou", 0.0))
        page_pcts.append(page_stats.get("pct_above_threshold", 0.0))
        total_lines += page_stats.get("n_lines", 0)

    if not page_ious:
        stats = {}
    else:
        stats = {
            "mean_iou": round(sum(page_ious) / len(page_ious), 4),
            "pct_above_threshold": round(sum(page_pcts) / len(page_pcts), 4),
            "threshold": threshold,
            "n_lines": total_lines,
        }
    stats["manuscript"] = manuscript_dir.name
    stats["pages"] = len(xml_files)
    return stats
