"""CLI htmir-collect : collecte et upload des folios Vinci depuis Gallica et Zenodo.

Usage (local) :
    htmir-collect --config configs/collection.yaml

Usage (SageMaker) :
    Ce script est aussi le point d'entrée du Processing Job défini dans
    ``infrastructure/sagemaker_collect.py``.

Options :
    --config PATH       Fichier de configuration YAML (défaut : configs/collection.yaml)
    --dry-run           Simule sans téléchargement ni upload
    --source            gallica | zenodo | all (défaut : all)
    --skip-validation   Désactive la validation qualité (utile en développement)
"""

import argparse
import tempfile
from pathlib import Path

import yaml

from htmir.collection.gallica import (
    search_vinci_manuscripts,
    download_folio,
    count_folios,
)
from htmir.collection.zenodo import (
    search_records,
    fetch_records_by_ids,
    download_record,
    deduplicate,
)
from htmir.collection.s3_storage import S3Storage
from htmir.collection.manifest_builder import CollectionManifest, FolioRecord
from htmir.collection.image_validator import validate_folio_image, is_acceptable
from htmir.aggregation.data_contract import sha256_file
from htmir.utils.logger import get_logger

logger = get_logger(__name__)


# ── Gallica ───────────────────────────────────────────────────────────────────


def collect_gallica(
    cfg: dict,
    storage: S3Storage,
    manifest: CollectionManifest,
    validate: bool = True,
) -> int:
    """Collecte les folios Gallica et les uploade sur S3.

    Pour chaque ARK trouvé via SRU, télécharge folio par folio,
    valide la qualité, calcule le SHA-256 et uploade sur S3.
    Alimente ``manifest`` en temps réel.

    Args:
        cfg: Configuration de collecte (contenu de ``collection.yaml``).
        storage: Instance S3Storage configurée.
        manifest: Manifeste de collecte en cours de construction.
        validate: Active la validation qualité avant upload.

    Returns:
        Nombre de folios correctement uploadés.
    """
    gcfg = cfg["sources"]["gallica"]
    if not gcfg.get("enabled", True):
        logger.info("Source Gallica désactivée dans la configuration")
        return 0

    vcfg = cfg.get("validation", {})
    ark_ids = search_vinci_manuscripts(
        query=gcfg.get("search_query"),
        max_results=gcfg.get("max_results", 50),
        rate_limit=gcfg.get("rate_limit_seconds", 1.5),
    )

    # Ajouter les ARK de secours non encore dans la liste
    for ark in gcfg.get("fallback_arks", []):
        if ark not in ark_ids:
            ark_ids.append(ark)

    uploaded = 0

    for ark_id in ark_ids:
        n_folios = count_folios(ark_id)
        if n_folios == 0:
            logger.warning(f"Manuscrit {ark_id} : 0 folio détecté, ignoré")
            continue
        logger.info(f"Manuscrit {ark_id} : {n_folios} folio(s) à traiter")

        for idx in range(1, n_folios + 1):
            folio_id = f"{ark_id}_f{idx:04d}"

            # Ne pas re-télécharger les folios déjà validés
            existing = manifest.get(folio_id)
            if existing and existing.status == "validated":
                logger.debug(f"Folio déjà validé : {folio_id}")
                continue

            s3_key = f"{gcfg['s3_prefix']}/{ark_id}/folio_{idx:04d}.jpg"
            if storage.exists(s3_key):
                logger.debug(f"Déjà sur S3 : {s3_key}")
                continue

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / f"folio_{idx:04d}.jpg"
                folio = download_folio(
                    ark_id,
                    idx,
                    tmp_path,
                    width=gcfg.get("iiif_width", "2000,"),
                    rate_limit=gcfg.get("rate_limit_seconds", 1.5),
                )
                if folio is None:
                    manifest.add(
                        FolioRecord(
                            folio_id=folio_id,
                            source="gallica",
                            s3_uri="",
                            ark_id=ark_id,
                            licence="gallica-non-commercial",
                            status="rejected",
                            rejection_reason="download_failed",
                        )
                    )
                    continue

                status = "validated"
                rejection_reason = ""
                if validate:
                    quality = validate_folio_image(
                        tmp_path,
                        min_width=vcfg.get("min_width_px", 1200),
                        min_height=vcfg.get("min_height_px", 800),
                        max_blank_ratio=vcfg.get("max_blank_ratio", 0.95),
                        min_ink_density=vcfg.get("min_ink_density", 0.005),
                    )
                    if not is_acceptable(quality, vcfg.get("max_bleedthrough_score", 0.4)):
                        status = "rejected"
                        rejection_reason = quality.rejection_reason
                        logger.warning(f"Folio rejeté {folio_id} : {rejection_reason}")

                sha = sha256_file(tmp_path) if status == "validated" else ""
                s3_uri = storage.upload(tmp_path, s3_key) if status == "validated" else ""

                manifest.add(
                    FolioRecord(
                        folio_id=folio_id,
                        source="gallica",
                        s3_uri=s3_uri,
                        ark_id=ark_id,
                        licence="gallica-non-commercial",
                        width=folio.width,
                        height=folio.height,
                        sha256=sha,
                        status=status,
                        rejection_reason=rejection_reason,
                    )
                )
                if status == "validated":
                    uploaded += 1

    logger.info(f"Gallica terminé : {uploaded} folio(s) uploadé(s)")
    return uploaded


# ── Zenodo ────────────────────────────────────────────────────────────────────


def collect_zenodo(
    cfg: dict,
    storage: S3Storage,
    manifest: CollectionManifest,
) -> int:
    """Collecte les datasets Zenodo et les uploade sur S3.

    Exécute chaque requête configurée, déduplique les résultats et uploade
    tous les fichiers (avec décompression ZIP) vers S3.

    Args:
        cfg: Configuration de collecte.
        storage: Instance S3Storage configurée.
        manifest: Manifeste de collecte en cours de construction.

    Returns:
        Nombre de fichiers correctement uploadés.
    """
    zcfg = cfg["sources"]["zenodo"]
    if not zcfg.get("enabled", True):
        logger.info("Source Zenodo désactivée dans la configuration")
        return 0

    all_records = []

    # Priorité 1 : IDs Zenodo explicites (datasets HTR connus)
    known_ids = zcfg.get("known_record_ids", [])
    if known_ids:
        all_records.extend(
            fetch_records_by_ids(known_ids, rate_limit=zcfg.get("rate_limit_seconds", 0.5))
        )

    # Priorité 2 : recherche textuelle en fallback
    for query in zcfg.get("queries", []):
        records = search_records(
            query,
            size=zcfg.get("max_results_per_query", 5),
            rate_limit=zcfg.get("rate_limit_seconds", 0.5),
        )
        all_records.extend(records)

    all_records = deduplicate(all_records)
    logger.info(f"Zenodo : {len(all_records)} enregistrement(s) uniques à télécharger")

    uploaded = 0

    for record in all_records:
        with tempfile.TemporaryDirectory() as tmp:
            files = download_record(record, Path(tmp), rate_limit=zcfg.get("rate_limit_seconds", 0.5))

            for file_path in files:
                if not file_path.is_file():
                    continue
                # Clé S3 relative au dossier tmp
                try:
                    relative = file_path.relative_to(tmp)
                except ValueError:
                    relative = Path(file_path.name)

                s3_key = f"{zcfg['s3_prefix']}/{record.record_id}/{relative.as_posix()}"
                folio_id = f"zenodo-{record.record_id}-{relative.as_posix().replace('/', '_')}"

                if storage.exists(s3_key):
                    logger.debug(f"Déjà sur S3 : {s3_key}")
                    continue

                try:
                    s3_uri = storage.upload(file_path, s3_key)
                except Exception as exc:
                    logger.error(f"Upload échoué {file_path.name} : {exc}")
                    continue

                manifest.add(
                    FolioRecord(
                        folio_id=folio_id,
                        source="zenodo",
                        s3_uri=s3_uri,
                        zenodo_id=str(record.record_id),
                        licence=record.licence,
                        status="downloaded",
                    )
                )
                uploaded += 1

    logger.info(f"Zenodo terminé : {uploaded} fichier(s) uploadé(s)")
    return uploaded


# ── Entrée principale ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTmiR — Collecte et upload des données vers S3 (Gallica + Zenodo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection.yaml"),
        help="Fichier de configuration YAML",
    )
    parser.add_argument(
        "--source",
        choices=["gallica", "zenodo", "all"],
        default="all",
        help="Source(s) à collecter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule sans téléchargement ni upload S3",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Désactive la validation qualité (pour les tests)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.dry_run:
        logger.info("Mode dry-run activé — aucun téléchargement ni upload effectué")
        return

    storage = S3Storage(bucket=cfg["bucket"], region=cfg.get("region", "eu-west-3"))
    manifest = CollectionManifest.load_from_s3(storage)

    n_gallica, n_zenodo = 0, 0

    if args.source in ("gallica", "all"):
        n_gallica = collect_gallica(cfg, storage, manifest, validate=not args.skip_validation)

    if args.source in ("zenodo", "all"):
        n_zenodo = collect_zenodo(cfg, storage, manifest)

    manifest.push_to_s3(storage)

    stats = manifest.stats()
    logger.info(
        f"Collecte terminée — Gallica : {n_gallica} folio(s) | "
        f"Zenodo : {n_zenodo} fichier(s) | "
        f"Stats manifeste : {stats}"
    )


if __name__ == "__main__":
    main()
