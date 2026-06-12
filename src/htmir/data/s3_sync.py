"""Synchronisation du dataset préparé vers / depuis S3.

La source de vérité du dataset est S3 (``s3://htmir-data/datasets/...``). On
archive chaque split en un ``.tar.gz`` unique avant l'upload — bien plus
efficace que des dizaines de milliers de petits ``PUT`` (un par ligne).

Fonctions :
    - ``upload_dataset`` : tar par split + manifeste → S3.
    - ``download_dataset`` : récupère et extrait les tarballs depuis S3.

Les chemins/clés S3 sont construits par des fonctions pures (testables) ;
les appels réseau passent par un client boto3 injecté.
"""

import tarfile
from pathlib import Path

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def split_tar_key(s3_prefix: str, split: str) -> str:
    """Clé S3 du tarball d'un split (ex. ``datasets/.../train.tar.gz``)."""
    return f"{s3_prefix.rstrip('/')}/{split}.tar.gz"


def manifest_key(s3_prefix: str) -> str:
    """Clé S3 du manifeste du dataset."""
    return f"{s3_prefix.rstrip('/')}/dataset_manifest.json"


def make_split_archive(split_dir: Path, archive_path: Path) -> Path:
    """Archive un répertoire de split en ``.tar.gz``.

    Args:
        split_dir: Répertoire contenant ``line_*.png`` + ``line_*.gt.txt``.
        archive_path: Chemin du ``.tar.gz`` à créer.

    Returns:
        Le chemin de l'archive créée.
    """
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(split_dir, arcname=split_dir.name)
    return archive_path


def upload_dataset(local_dir: Path, bucket: str, s3_prefix: str, s3_client) -> list[str]:
    """Archive chaque split et l'uploade sur S3, plus le manifeste.

    Args:
        local_dir: Racine du dataset (sous-dossiers ``train/`` etc. + manifeste).
        bucket: Nom du bucket S3.
        s3_prefix: Préfixe S3 de destination.
        s3_client: Client boto3 S3 (``boto3.client("s3")``).

    Returns:
        Liste des clés S3 uploadées.
    """
    local_dir = Path(local_dir)
    uploaded: list[str] = []

    for split_dir in sorted(p for p in local_dir.iterdir() if p.is_dir()):
        if not any(split_dir.glob("*.png")):
            continue
        archive = local_dir / f"{split_dir.name}.tar.gz"
        make_split_archive(split_dir, archive)
        key = split_tar_key(s3_prefix, split_dir.name)
        s3_client.upload_file(str(archive), bucket, key)
        archive.unlink(missing_ok=True)
        uploaded.append(key)
        logger.info(f"Uploadé : s3://{bucket}/{key}")

    manifest = local_dir / "dataset_manifest.json"
    if manifest.exists():
        key = manifest_key(s3_prefix)
        s3_client.upload_file(str(manifest), bucket, key)
        uploaded.append(key)
        logger.info(f"Uploadé : s3://{bucket}/{key}")

    return uploaded


def download_dataset(bucket: str, s3_prefix: str, local_dir: Path,
                     splits: list[str], s3_client) -> Path:
    """Télécharge et extrait les tarballs de splits depuis S3.

    Args:
        bucket: Nom du bucket.
        s3_prefix: Préfixe S3 source.
        local_dir: Répertoire local de destination.
        splits: Splits à récupérer (``["train", "validation", "test"]``).
        s3_client: Client boto3 S3.

    Returns:
        Le répertoire local peuplé.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        key = split_tar_key(s3_prefix, split)
        archive = local_dir / f"{split}.tar.gz"
        try:
            s3_client.download_file(bucket, key, str(archive))
        except Exception as exc:
            logger.warning(f"Split '{split}' absent sur S3 ({key}) : {exc}")
            continue
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(local_dir, filter="data")
        archive.unlink(missing_ok=True)
        logger.info(f"Extrait : {split}")

    return local_dir
