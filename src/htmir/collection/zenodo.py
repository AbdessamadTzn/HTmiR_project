"""Collecteur de datasets via l'API REST Zenodo.

Permet de rechercher et télécharger des jeux de données publics pour le
pré-entraînement HTR (CREMMA Medieval, CATMuS, etc.) ainsi que tout dépôt
en rapport avec les manuscrits de Léonard de Vinci.
"""

import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_ZENODO_API = "https://zenodo.org/api/records"


@dataclass
class ZenodoRecord:
    """Un enregistrement Zenodo.

    Args:
        record_id: Identifiant numérique Zenodo.
        title: Titre du dataset.
        doi: DOI Zenodo (ex. 10.5281/zenodo.12345).
        licence: Identifiant de licence (ex. cc-by-4.0).
        files: Liste de dicts ``{key, url, size_bytes}``.
    """

    record_id: int
    title: str
    doi: str = ""
    licence: str = "unknown"
    files: list[dict] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        """Identifiant stable pour le manifeste de collecte."""
        return f"zenodo-{self.record_id}"

    @property
    def total_size_mb(self) -> float:
        return sum(f.get("size_bytes", 0) for f in self.files) / 1_048_576


# ── Fetch par ID (méthode fiable) ────────────────────────────────────────────


def fetch_record_by_id(record_id: int, rate_limit: float = 0.5) -> ZenodoRecord | None:
    """Récupère un enregistrement Zenodo par son identifiant numérique.

    Plus fiable que la recherche textuelle : garantit qu'on obtient exactement
    le dataset voulu (CREMMA, HTR-United, e-NDP…).

    Args:
        record_id: Identifiant Zenodo (ex. 10552907).
        rate_limit: Pause (s) avant la requête.

    Returns:
        :class:`ZenodoRecord` ou ``None`` si introuvable.
    """
    time.sleep(rate_limit)
    url = f"https://zenodo.org/api/records/{record_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Erreur Zenodo fetch_by_id({record_id}) : {exc}")
        return None

    hit = resp.json()
    meta = hit.get("metadata", {})
    lic_raw = meta.get("license", {})
    lic = lic_raw.get("id", "unknown") if isinstance(lic_raw, dict) else str(lic_raw)
    files = [
        {"key": f["key"], "url": f["links"]["self"], "size_bytes": f.get("size", 0)}
        for f in hit.get("files", [])
    ]
    record = ZenodoRecord(
        record_id=int(hit["id"]),
        title=meta.get("title", ""),
        doi=hit.get("doi", ""),
        licence=lic,
        files=files,
    )
    logger.info(f"Zenodo record {record_id} : {record.title!r} ({len(files)} fichier(s))")
    return record


def fetch_records_by_ids(
    record_ids: list[int], rate_limit: float = 0.5
) -> list[ZenodoRecord]:
    """Récupère une liste d'enregistrements Zenodo par leurs IDs.

    Args:
        record_ids: Liste d'identifiants Zenodo.
        rate_limit: Pause entre chaque requête.

    Returns:
        Liste de :class:`ZenodoRecord` valides (les échecs sont ignorés).
    """
    results = []
    for rid in record_ids:
        rec = fetch_record_by_id(rid, rate_limit=rate_limit)
        if rec:
            results.append(rec)
    return results


# ── Recherche textuelle (fallback) ────────────────────────────────────────────


def search_records(
    query: str,
    size: int = 10,
    type_filter: str = "dataset",
    rate_limit: float = 0.5,
) -> list[ZenodoRecord]:
    """Interroge l'API Zenodo REST et retourne les enregistrements publiés.

    Args:
        query: Requête de recherche (syntaxe Elasticsearch Zenodo).
        size: Nombre maximum de résultats.
        type_filter: Type de ressource Zenodo (``"dataset"``, ``"software"``…).
        rate_limit: Pause (s) avant la requête (respect du rate-limit Zenodo).

    Returns:
        Liste de :class:`ZenodoRecord` correspondant aux résultats.
    """
    time.sleep(rate_limit)
    params = {
        "q": query,
        "size": size,
        "type": type_filter,
        "status": "published",
        "sort": "mostrecent",
    }
    try:
        resp = requests.get(_ZENODO_API, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Erreur API Zenodo ({query!r}) : {exc}")
        return []

    records: list[ZenodoRecord] = []
    for hit in resp.json().get("hits", {}).get("hits", []):
        meta = hit.get("metadata", {})

        lic_raw = meta.get("license", {})
        lic = lic_raw.get("id", "unknown") if isinstance(lic_raw, dict) else str(lic_raw)

        files = [
            {
                "key": f["key"],
                "url": f["links"]["self"],
                "size_bytes": f.get("size", 0),
            }
            for f in hit.get("files", [])
        ]

        records.append(
            ZenodoRecord(
                record_id=int(hit["id"]),
                title=meta.get("title", ""),
                doi=hit.get("doi", ""),
                licence=lic,
                files=files,
            )
        )

    logger.info(f"Zenodo : {len(records)} résultat(s) pour {query!r}")
    return records


def deduplicate(records: list[ZenodoRecord]) -> list[ZenodoRecord]:
    """Supprime les doublons par ``record_id``."""
    seen: set[int] = set()
    unique: list[ZenodoRecord] = []
    for r in records:
        if r.record_id not in seen:
            seen.add(r.record_id)
            unique.append(r)
    return unique


# ── Téléchargement ────────────────────────────────────────────────────────────


def download_record(
    record: ZenodoRecord,
    out_dir: Path,
    rate_limit: float = 0.5,
    extract_zip: bool = True,
    skip_existing: bool = True,
) -> list[Path]:
    """Télécharge les fichiers d'un enregistrement Zenodo.

    Les archives ZIP sont décompressées automatiquement si ``extract_zip``
    est activé. Les fichiers déjà présents sont ignorés si ``skip_existing``.

    Args:
        record: Enregistrement Zenodo à télécharger.
        out_dir: Répertoire racine de destination.
        rate_limit: Pause (s) entre chaque fichier.
        extract_zip: Décompresse les archives ``.zip`` après téléchargement.
        skip_existing: Saute les fichiers déjà téléchargés.

    Returns:
        Chemins de tous les fichiers disponibles après téléchargement
        (archives incluses + fichiers extraits si ``extract_zip``).
    """
    record_dir = out_dir / str(record.record_id)
    record_dir.mkdir(parents=True, exist_ok=True)
    available: list[Path] = []

    for file_info in record.files:
        dest = record_dir / file_info["key"]

        if skip_existing and dest.exists():
            logger.debug(f"Déjà présent : {dest.name}")
            available.append(dest)
        else:
            size_kb = file_info["size_bytes"] // 1024
            logger.info(f"Téléchargement Zenodo {record.record_id}/{file_info['key']} ({size_kb} KB)")
            try:
                resp = requests.get(file_info["url"], timeout=120, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65_536):
                        fh.write(chunk)
                available.append(dest)
                time.sleep(rate_limit)
            except requests.RequestException as exc:
                logger.error(f"Échec {file_info['key']} (record {record.record_id}) : {exc}")
                continue

        if extract_zip and dest.suffix.lower() == ".zip" and dest.exists():
            extract_dir = record_dir / dest.stem
            if not extract_dir.exists():
                extract_dir.mkdir()
                try:
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(extract_dir)
                    logger.info(f"Extrait : {dest.name} → {extract_dir.name}/")
                except zipfile.BadZipFile as exc:
                    logger.error(f"Archive corrompue {dest.name} : {exc}")
            extracted = [p for p in extract_dir.rglob("*") if p.is_file()]
            available.extend(extracted)

    return available
