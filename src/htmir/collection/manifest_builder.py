"""Registre de collecte : trace chaque folio de sa source jusqu'à S3.

Le manifeste de collecte (``CollectionManifest``) est distinct du manifeste
d'entraînement HTR (``LineRecord`` dans ``htmir.corpus.manifest``).
Il joue le rôle de *data lineage* : on y consigne l'origine, la licence,
le hash SHA-256 et l'état de traitement de chaque image brute.

Stockage :
    - Localement : JSON via ``save(path)``
    - S3 : ``s3://htmir-data/manifests/collection_manifest.json``
"""

import json
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_S3_MANIFEST_KEY = "manifests/collection_manifest.json"
_SCHEMA_VERSION = "1.1"


@dataclass
class FolioRecord:
    """Un folio dans le manifeste de collecte.

    Args:
        folio_id: Identifiant unique stable (ex. ``btv1b10022860x_f0001``).
        source: Origine (``gallica``, ``zenodo``, ``local``).
        s3_uri: URI S3 de l'image brute validée.
        ark_id: Identifiant ARK Gallica si applicable.
        zenodo_id: Identifiant Zenodo si applicable.
        date_downloaded: Timestamp ISO 8601 UTC du téléchargement.
        licence: Licence SPDX ou label libre (ex. ``cc-by-4.0``).
        width: Largeur de l'image en pixels.
        height: Hauteur de l'image en pixels.
        sha256: Empreinte SHA-256 du fichier téléchargé.
        status: État courant (``downloaded``, ``validated``, ``rejected``,
                ``preprocessed``).
        rejection_reason: Motif de rejet éventuel (vide si validé).
    """

    folio_id: str
    source: str
    s3_uri: str
    ark_id: str = ""
    zenodo_id: str = ""
    date_downloaded: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    licence: str = "unknown"
    width: int = 0
    height: int = 0
    sha256: str = ""
    status: str = "downloaded"
    rejection_reason: str = ""

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


class CollectionManifest:
    """Registre complet des folios collectés pour le projet HTmiR.

    Les enregistrements sont indexés par ``folio_id`` ; un appel à
    :meth:`add` met à jour silencieusement un folio existant.
    """

    def __init__(self) -> None:
        self._records: dict[str, FolioRecord] = {}

    # ── Interface de base ─────────────────────────────────────────────────────

    def add(self, record: FolioRecord) -> None:
        """Ajoute ou met à jour un folio."""
        self._records[record.folio_id] = record

    def get(self, folio_id: str) -> FolioRecord | None:
        return self._records.get(folio_id)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records.values())

    def __contains__(self, folio_id: str) -> bool:
        return folio_id in self._records

    # ── Filtres ───────────────────────────────────────────────────────────────

    def filter_status(self, status: str) -> list[FolioRecord]:
        """Retourne tous les enregistrements avec le statut donné."""
        return [r for r in self._records.values() if r.status == status]

    def stats(self) -> dict[str, int]:
        """Compte des enregistrements par statut."""
        counts: dict[str, int] = {}
        for r in self._records.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    # ── Persistance locale ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Sauvegarde le manifeste en JSON localement.

        Args:
            path: Chemin de destination (les répertoires parents sont créés).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "total": len(self._records),
            "stats": self.stats(),
            "records": [asdict(r) for r in self._records.values()],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info(f"Manifeste sauvegardé : {path} ({len(self._records)} folio(s))")

    @classmethod
    def load(cls, path: Path) -> "CollectionManifest":
        """Charge un manifeste JSON depuis un fichier local.

        Args:
            path: Chemin du fichier JSON.

        Returns:
            :class:`CollectionManifest` peuplé.
        """
        manifest = cls()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data.get("records", []):
            # Compatibilité ascendante : ignorer les champs inconnus
            known = {f.name for f in FolioRecord.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            filtered = {k: v for k, v in item.items() if k in known}
            manifest.add(FolioRecord(**filtered))
        logger.info(f"Manifeste chargé : {len(manifest)} enregistrement(s) depuis {path}")
        return manifest

    # ── Persistance S3 ────────────────────────────────────────────────────────

    def push_to_s3(self, storage, s3_key: str = _S3_MANIFEST_KEY) -> str:
        """Sérialise et uploade le manifeste vers S3.

        Args:
            storage: Instance :class:`~htmir.collection.s3_storage.S3Storage`.
            s3_key: Clé S3 de destination.

        Returns:
            URI S3 du manifeste uploadé.
        """
        with tempfile.NamedTemporaryFile(suffix="_manifest.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.save(tmp_path)
            uri = storage.upload(tmp_path, s3_key)
        finally:
            tmp_path.unlink(missing_ok=True)
        logger.info(f"Manifeste poussé sur S3 : {uri}")
        return uri

    @classmethod
    def load_from_s3(
        cls,
        storage,
        s3_key: str = _S3_MANIFEST_KEY,
    ) -> "CollectionManifest":
        """Charge le manifeste depuis S3 ; retourne un manifeste vide si absent.

        Args:
            storage: Instance :class:`~htmir.collection.s3_storage.S3Storage`.
            s3_key: Clé S3 du manifeste.

        Returns:
            :class:`CollectionManifest` existant ou nouveau manifeste vide.
        """
        with tempfile.NamedTemporaryFile(suffix="_manifest.json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if storage.exists(s3_key):
                storage.download(s3_key, tmp_path)
                return cls.load(tmp_path)
            logger.info("Aucun manifeste existant sur S3 — nouveau manifeste créé")
            return cls()
        finally:
            tmp_path.unlink(missing_ok=True)
