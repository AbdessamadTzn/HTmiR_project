"""Collecteur de manuscrits via l'API IIIF Gallica (Bibliothèque nationale de France).

Flux :
  1. Recherche via SRU Dublin Core → liste d'identifiants ARK.
  2. Récupération du manifeste IIIF pour chaque manuscrit → liste de canvases.
  3. Téléchargement folio par folio avec gestion du rate-limiting et des retry.
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import requests

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_IIIF = "https://gallica.bnf.fr/iiif"
_SRU_ENDPOINT = "https://gallica.bnf.fr/SRU"

# User-Agent explicite pour usage académique (requis par Gallica pour éviter le 403)
_HEADERS = {
    "User-Agent": (
        "HTmiR-Research/1.0 (HETIC Master Data/IA 2026; "
        "academic HTR project on Leonardo da Vinci manuscripts; "
        "non-commercial)"
    ),
    "Accept": "application/xml, text/xml",
}

_NS = {
    "srw": "http://www.loc.gov/zing/srw/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

VINCI_SRU_QUERY = 'dc.creator all "Léonard de Vinci" and dc.type all "manuscrit"'


@dataclass
class GallicaFolio:
    """Un folio téléchargé depuis Gallica.

    Args:
        ark_id: Identifiant ARK sans préfixe (ex. btv1b10022860x).
        folio_idx: Rang du folio dans le manifeste IIIF (1-based).
        iiif_url: URL IIIF source utilisée pour le téléchargement.
        width: Largeur réelle de l'image téléchargée (pixels).
        height: Hauteur réelle de l'image téléchargée (pixels).
        s3_uri: URI S3 après upload (vide avant l'upload).
    """

    ark_id: str
    folio_idx: int
    iiif_url: str = ""
    width: int = 0
    height: int = 0
    s3_uri: str = ""

    @property
    def folio_id(self) -> str:
        """Identifiant stable du folio : {ark}_{folio:04d}."""
        return f"{self.ark_id}_f{self.folio_idx:04d}"


# ── Recherche SRU ─────────────────────────────────────────────────────────────


def search_vinci_manuscripts(
    query: str = VINCI_SRU_QUERY,
    max_results: int = 100,
    rate_limit: float = 1.0,
) -> list[str]:
    """Recherche les manuscrits de Vinci sur Gallica via l'API SRU.

    Parcourt les pages de résultats SRU par tranches de 50 et extrait
    les identifiants ARK depuis les champs Dublin Core ``dc:identifier``.

    Args:
        query: Requête SRU (Dublin Core).
        max_results: Nombre maximum d'ARK à retourner.
        rate_limit: Pause en secondes entre chaque requête paginée.

    Returns:
        Liste d'identifiants ARK dédupliqués (ex. ["btv1b10022860x", ...]).
    """
    ark_ids: list[str] = []
    start = 1
    page_size = min(50, max_results)

    while len(ark_ids) < max_results:
        params = {
            "operation": "searchRetrieve",
            "version": "1.2",
            "query": query,
            "maximumRecords": page_size,
            "startRecord": start,
        }
        url = f"{_SRU_ENDPOINT}?{urlencode(params)}"
        logger.info(f"SRU Gallica page {start} : {url}")

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Erreur SRU Gallica (startRecord={start}) : {exc}")
            break

        root = ET.fromstring(resp.content)
        identifiers = root.findall(".//dc:identifier", _NS)
        if not identifiers:
            logger.info("SRU : fin des résultats")
            break

        batch_count = 0
        for elem in identifiers:
            text = (elem.text or "").strip()
            if "gallica.bnf.fr/ark:/12148/" not in text:
                continue
            ark = text.split("ark:/12148/")[-1].rstrip("/")
            if ark and ark not in ark_ids:
                ark_ids.append(ark)
                batch_count += 1

        logger.info(f"Lot {start} : +{batch_count} ARK (total {len(ark_ids)})")
        start += page_size
        if len(identifiers) < page_size:
            break
        time.sleep(rate_limit)

    logger.info(f"Gallica SRU terminé : {len(ark_ids)} manuscrit(s)")
    return ark_ids[:max_results]


# ── Manifeste IIIF ────────────────────────────────────────────────────────────


def fetch_iiif_manifest(ark_id: str, timeout: int = 30) -> dict:
    """Charge le manifeste IIIF Presentation 2.x d'un document Gallica.

    Args:
        ark_id: Identifiant ARK (sans préfixe ``ark:/12148/``).
        timeout: Timeout HTTP en secondes.

    Returns:
        Manifeste IIIF sous forme de dictionnaire.

    Raises:
        requests.HTTPError: Si le manifeste est inaccessible (404, 5xx…).
    """
    url = f"{_BASE_IIIF}/ark:/12148/{ark_id}/manifest.json"
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def count_folios(ark_id: str) -> int:
    """Retourne le nombre de folios d'un manuscrit via son manifeste IIIF.

    Args:
        ark_id: Identifiant ARK.

    Returns:
        Nombre de canvases, ou 0 si le manifeste est inaccessible.
    """
    try:
        manifest = fetch_iiif_manifest(ark_id)
        canvases = manifest.get("sequences", [{}])[0].get("canvases", [])
        return len(canvases)
    except Exception as exc:
        logger.error(f"Impossible de compter les folios de {ark_id} : {exc}")
        return 0


# ── Téléchargement d'un folio ─────────────────────────────────────────────────


def download_folio(
    ark_id: str,
    folio_idx: int,
    out_path: Path,
    width: str = "2000,",
    retries: int = 3,
    rate_limit: float = 1.5,
) -> GallicaFolio | None:
    """Télécharge un folio en JPEG via le protocole IIIF Image API 2.x.

    URL construite : ``{base}/ark:/12148/{ark}/f{n}/full/{width}/0/native.jpg``

    Args:
        ark_id: Identifiant ARK du manuscrit.
        folio_idx: Index du folio (1-based).
        out_path: Chemin local de destination (le répertoire parent est créé).
        width: Paramètre de taille IIIF (ex. ``"2000,"`` = max 2000 px en largeur,
               ``"full"`` = résolution native).
        retries: Tentatives en cas d'erreur réseau transitoire.
        rate_limit: Pause (s) entre tentatives successives.

    Returns:
        :class:`GallicaFolio` renseigné avec les dimensions réelles,
        ou ``None`` si toutes les tentatives ont échoué.
    """
    iiif_url = f"{_BASE_IIIF}/ark:/12148/{ark_id}/f{folio_idx}/full/{width}/0/native.jpg"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(iiif_url, headers=_HEADERS, timeout=60, stream=True)
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)

            from PIL import Image as PILImage
            with PILImage.open(out_path) as img:
                w, h = img.size

            time.sleep(rate_limit)
            logger.debug(f"Téléchargé : {ark_id}/f{folio_idx} ({w}×{h}px)")
            return GallicaFolio(
                ark_id=ark_id,
                folio_idx=folio_idx,
                iiif_url=iiif_url,
                width=w,
                height=h,
            )

        except Exception as exc:
            logger.warning(f"Tentative {attempt}/{retries} échouée {ark_id}/f{folio_idx} : {exc}")
            if attempt < retries:
                time.sleep(rate_limit * attempt * 2)

    logger.error(f"Abandon téléchargement {ark_id}/f{folio_idx} après {retries} tentatives")
    return None
