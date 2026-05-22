"""Export des segmentations au format PAGE XML (standard humanités numériques)."""

from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from htmir.utils.logger import get_logger

logger = get_logger(__name__)

XMLNS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XMLNS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
XSI_SCHEMA = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)


def polygon_to_coords_str(polygon: np.ndarray) -> str:
    """Convertit un polygone numpy en chaîne PAGE XML.

    Args:
        polygon: Array de shape (N, 2) avec colonnes (x, y).

    Returns:
        Chaîne au format 'x1,y1 x2,y2 ...'

    Example:
        >>> s = polygon_to_coords_str(np.array([[0,0],[10,0],[10,20],[0,20]]))
        >>> assert s == "0,0 10,0 10,20 0,20"
    """
    return " ".join(f"{int(x)},{int(y)}" for x, y in polygon)


def build_page_xml(
    image_filename: str,
    image_width: int,
    image_height: int,
    text_regions: list[dict],
) -> ET.ElementTree:
    """Construit un arbre PAGE XML à partir des régions segmentées.

    Args:
        image_filename: Nom du fichier image source.
        image_width: Largeur de l'image en pixels.
        image_height: Hauteur de l'image en pixels.
        text_regions: Liste de dicts avec clés :
            - 'id' (str) : identifiant de la région
            - 'polygon' (np.ndarray) : contour de la région, shape (N,2)
            - 'lines' (list[dict]) : lignes avec 'id', 'baseline', 'polygon', 'text'

    Returns:
        ElementTree PAGE XML prêt à sérialiser.

    Example:
        >>> tree = build_page_xml("ms_001.tif", 2480, 3508, regions)
        >>> tree.write("ms_001.xml", encoding="utf-8", xml_declaration=True)
    """
    ET.register_namespace("", XMLNS)
    root = ET.Element("PcGts", {
        "xmlns": XMLNS,
        "xmlns:xsi": XMLNS_XSI,
        "xsi:schemaLocation": XSI_SCHEMA,
    })

    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "htr-medieval-manuscripts-2026"
    ET.SubElement(metadata, "Created").text = datetime.utcnow().isoformat()
    ET.SubElement(metadata, "LastChange").text = datetime.utcnow().isoformat()

    page = ET.SubElement(root, "Page", {
        "imageFilename": image_filename,
        "imageWidth": str(image_width),
        "imageHeight": str(image_height),
    })

    for region in text_regions:
        tr = ET.SubElement(page, "TextRegion", {"id": region["id"], "type": "paragraph"})
        coords = ET.SubElement(tr, "Coords")
        coords.set("points", polygon_to_coords_str(region["polygon"]))

        for line in region.get("lines", []):
            tl = ET.SubElement(tr, "TextLine", {"id": line["id"]})
            lc = ET.SubElement(tl, "Coords")
            lc.set("points", polygon_to_coords_str(line["polygon"]))

            if "baseline" in line:
                bl = ET.SubElement(tl, "Baseline")
                bl.set("points", polygon_to_coords_str(line["baseline"]))

            if line.get("text"):
                te = ET.SubElement(tl, "TextEquiv")
                ET.SubElement(te, "Unicode").text = line["text"]

    return ET.ElementTree(root)


def save_page_xml(tree: ET.ElementTree, output_path: Path) -> None:
    """Sauvegarde un arbre PAGE XML avec déclaration UTF-8.

    Args:
        tree: ElementTree PAGE XML.
        output_path: Chemin de sortie (.xml).

    Example:
        >>> save_page_xml(tree, Path("segmentations/ms_001.xml"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    logger.info(f"PAGE XML sauvegardé : {output_path}")
