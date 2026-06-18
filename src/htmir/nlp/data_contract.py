"""Data contract : pont entre le HTR (ALTO Kraken) et le NLP.

Le contrat est un JSON décrivant, page par page et ligne par ligne :

- ``text``              : la transcription de la ligne ;
- ``polygon``          : le polygone englobant de la ligne (segmentation) ;
- ``baseline``         : la ligne de base ;
- ``char_confidences`` : la confiance par glyphe (ordre de lecture, hors espaces) ;
- ``mean_confidence``  : moyenne des confiances de la ligne ;
- ``needs_review``     : drapeau si ``mean_confidence`` sous le seuil.

C'est l'**entrée obligatoire** de tout le pipeline NLP. Le JSON est validé par
:func:`validate_contract` (schéma minimal, sans dépendance si ``jsonschema``
est absent).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"

REVIEW_THRESHOLD = 0.70  # mean_confidence sous ce seuil → needs_review

# ── Schéma du data contract (JSON Schema, draft simple) ──────────────────────
CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["source_image", "lines"],
    "properties": {
        "source_image": {"type": "string"},
        "model": {"type": ["string", "null"]},
        "page": {
            "type": "object",
            "properties": {
                "width": {"type": ["integer", "null"]},
                "height": {"type": ["integer", "null"]},
            },
        },
        "review_threshold": {"type": "number"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "text", "char_confidences",
                             "mean_confidence", "needs_review"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "polygon": {"type": "array"},
                    "baseline": {"type": "array"},
                    "char_confidences": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "mean_confidence": {"type": "number"},
                    "needs_review": {"type": "boolean"},
                    "candidates": {"type": "array"},
                },
            },
        },
    },
}


def _points(points_str: str) -> list[list[int]]:
    """``"x1 y1 x2 y2 ..."`` → ``[[x1,y1],[x2,y2],...]``."""
    vals = points_str.split()
    return [[int(float(vals[i])), int(float(vals[i + 1]))]
            for i in range(0, len(vals) - 1, 2)]


def alto_to_contract(
    alto_xml: str,
    source_image: str | None = None,
    model: str | None = None,
    review_threshold: float = REVIEW_THRESHOLD,
) -> dict:
    """Convertit un ALTO XML Kraken en data contract.

    Args:
        alto_xml: Contenu ALTO XML (chaîne).
        source_image: Nom de l'image source (sinon lu dans l'ALTO).
        model: Identifiant du modèle HTR (informatif).
        review_threshold: Seuil de ``mean_confidence`` pour ``needs_review``.

    Returns:
        Le dictionnaire du data contract.
    """
    ns = _ALTO_NS
    root = ET.fromstring(alto_xml)

    fn = root.find(f".//{{{ns}}}fileName")
    img = source_image or (fn.text.strip() if fn is not None and fn.text else "unknown")

    page_el = root.find(f".//{{{ns}}}Page")
    page = {}
    if page_el is not None:
        w = page_el.attrib.get("WIDTH")
        h = page_el.attrib.get("HEIGHT")
        page = {"width": int(w) if w else None, "height": int(h) if h else None}

    lines: list[dict] = []
    for i, tl in enumerate(root.iter(f"{{{ns}}}TextLine")):
        # texte : on joint les String (mots) par une espace
        words, confs = [], []
        for el in tl:
            tag = el.tag.split("}")[-1]
            if tag == "String":
                content = el.attrib.get("CONTENT", "")
                if content:
                    words.append(content)
                for glyph in el.iter(f"{{{ns}}}Glyph"):
                    gc = glyph.attrib.get("GC")
                    if gc is not None:
                        confs.append(round(float(gc), 4))
        text = " ".join(words)
        if not text:
            continue

        # polygone de la ligne (TextLine > Shape > Polygon)
        polygon = []
        shape = tl.find(f"{{{ns}}}Shape/{{{ns}}}Polygon")
        if shape is not None and shape.attrib.get("POINTS"):
            polygon = _points(shape.attrib["POINTS"])
        baseline = _points(tl.attrib["BASELINE"]) if tl.attrib.get("BASELINE") else []

        mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
        lines.append({
            "id": tl.attrib.get("ID", f"line_{i}"),
            "text": text,
            "polygon": polygon,
            "baseline": baseline,
            "char_confidences": confs,
            "mean_confidence": mean_conf,
            "needs_review": mean_conf < review_threshold,
            "candidates": [],
        })

    return {
        "source_image": img,
        "model": model,
        "page": page,
        "review_threshold": review_threshold,
        "lines": lines,
    }


def validate_contract(contract: dict) -> None:
    """Valide le data contract contre :data:`CONTRACT_SCHEMA`.

    Utilise ``jsonschema`` si disponible (validation complète), sinon effectue
    des vérifications minimales. Lève ``ValueError`` si invalide.
    """
    try:
        import jsonschema
    except ImportError:
        jsonschema = None

    if jsonschema is not None:
        try:
            jsonschema.validate(contract, CONTRACT_SCHEMA)
            return
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Contrat invalide : {exc.message}") from exc

    # Validation minimale de repli (sans jsonschema)
    if "source_image" not in contract or "lines" not in contract:
        raise ValueError("Contrat invalide : champs 'source_image'/'lines' requis.")
    for ln in contract["lines"]:
        missing = {"id", "text", "char_confidences", "mean_confidence",
                   "needs_review"} - set(ln)
        if missing:
            raise ValueError(f"Ligne invalide, champs manquants : {missing}")


def save_contract(contract: dict, path: Path) -> None:
    """Écrit le data contract (validé) en JSON."""
    validate_contract(contract)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
