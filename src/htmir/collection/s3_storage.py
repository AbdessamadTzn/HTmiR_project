"""Interface S3 pour le stockage centralisé des données HTmiR.

Encapsule boto3 pour exposer les opérations usuelles du projet :
upload, download, existence, listage de préfixe.
"""

from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class S3Storage:
    """Client S3 wrappé pour le projet HTmiR.

    Args:
        bucket: Nom du bucket S3 (ex. ``htmir-data``).
        region: Région AWS (ex. ``eu-west-3``).
    """

    bucket: str
    region: str = "eu-west-3"
    _client: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._client = boto3.client("s3", region_name=self.region)

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload(self, local_path: Path, s3_key: str) -> str:
        """Upload un fichier local et retourne son URI ``s3://``.

        Args:
            local_path: Chemin local existant.
            s3_key: Clé de destination dans le bucket.

        Returns:
            URI complète ``s3://{bucket}/{key}``.

        Raises:
            FileNotFoundError: Si ``local_path`` n'existe pas.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {local_path}")
        self._client.upload_file(str(local_path), self.bucket, s3_key)
        uri = f"s3://{self.bucket}/{s3_key}"
        logger.debug(f"S3 upload : {local_path.name} → {uri}")
        return uri

    def upload_directory(self, local_dir: Path, s3_prefix: str) -> list[str]:
        """Upload récursivement tous les fichiers d'un dossier.

        Args:
            local_dir: Répertoire local à uploader.
            s3_prefix: Préfixe S3 de destination (sans slash final).

        Returns:
            Liste des URI ``s3://`` créées.
        """
        uris: list[str] = []
        for file_path in sorted(Path(local_dir).rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{s3_prefix.rstrip('/')}/{relative.as_posix()}"
            uris.append(self.upload(file_path, key))
        logger.info(f"S3 upload dossier : {len(uris)} fichier(s) → s3://{self.bucket}/{s3_prefix}")
        return uris

    # ── Téléchargement ────────────────────────────────────────────────────────

    def download(self, s3_key: str, local_path: Path) -> Path:
        """Télécharge un objet S3 vers un fichier local.

        Args:
            s3_key: Clé S3 source.
            local_path: Chemin local de destination.

        Returns:
            ``local_path`` après téléchargement.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, s3_key, str(local_path))
        logger.debug(f"S3 download : s3://{self.bucket}/{s3_key} → {local_path}")
        return local_path

    # ── Existence / listage ───────────────────────────────────────────────────

    def exists(self, s3_key: str) -> bool:
        """Vérifie si une clé existe dans le bucket (HEAD request).

        Args:
            s3_key: Clé S3 à vérifier.

        Returns:
            ``True`` si l'objet existe, ``False`` sinon.

        Raises:
            ClientError: Pour toute erreur autre qu'un 404.
        """
        try:
            self._client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        """Liste toutes les clés S3 sous un préfixe donné (paginé).

        Args:
            prefix: Préfixe S3 (ex. ``"raw/gallica/"``).

        Returns:
            Liste complète des clés sous ce préfixe.
        """
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def uri(self, s3_key: str) -> str:
        """Construit l'URI S3 complète pour une clé donnée."""
        return f"s3://{self.bucket}/{s3_key}"
