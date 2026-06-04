"""Collecteur de datasets HTR depuis HuggingFace Hub.

Télécharge CATMuS Medieval et CREMMA Medieval, sauvegarde les paires
(image JPEG + transcription .txt) et les uploade sur S3.

Datasets ciblés :
    - CATMuS/medieval  : ~160 000 lignes, manuscrits VIIIe-XVIIe s., 10 langues
    - HTR-United/cremma-medieval : 15 manuscrits français XIIIe-XVe s.
"""

from dataclasses import dataclass
from pathlib import Path

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

# Datasets HuggingFace avec leur split et nombre max de lignes à collecter
_HF_DATASETS: list[dict] = [
    {
        "repo_id": "CATMuS/medieval",
        "config": None,
        "split": "train",
        "max_samples": 3000,
        "s3_prefix": "raw/huggingface/catmus-medieval",
    },
    {
        "repo_id": "HTR-United/cremma-medieval",
        "config": None,
        "split": "train",
        "max_samples": 2000,
        "s3_prefix": "raw/huggingface/cremma-medieval",
    },
]


@dataclass
class HFSample:
    """Une ligne de manuscrit extraite d'un dataset HuggingFace.

    Args:
        sample_id: Identifiant unique de la ligne.
        image_s3_uri: URI S3 de l'image JPEG.
        text_s3_uri: URI S3 du fichier .txt de transcription.
        text: Transcription brute.
        repo_id: Identifiant du dataset HuggingFace source.
    """

    sample_id: str
    image_s3_uri: str
    text_s3_uri: str
    text: str
    repo_id: str


def _save_sample_locally(
    idx: int,
    sample: dict,
    out_dir: Path,
    repo_id: str,
) -> tuple[Path, Path] | None:
    """Sauvegarde une image et sa transcription dans out_dir.

    Args:
        idx: Index du sample dans le dataset.
        sample: Dictionnaire HuggingFace avec clés 'image' et 'text'.
        out_dir: Répertoire de sortie local.
        repo_id: Identifiant du dataset (pour les logs).

    Returns:
        Tuple (image_path, text_path) ou None si le sample est invalide.
    """
    text = (sample.get("text") or sample.get("transcription") or "").strip()
    image = sample.get("image")

    if not text or image is None:
        return None

    stem = f"line_{idx:06d}"
    img_path = out_dir / f"{stem}.jpg"
    txt_path = out_dir / f"{stem}.txt"

    try:
        if hasattr(image, "save"):
            image.convert("RGB").save(img_path, "JPEG", quality=92)
        else:
            logger.warning(f"[{repo_id}] sample {idx} : image non PIL, ignoré")
            return None
    except Exception as exc:
        logger.warning(f"[{repo_id}] sample {idx} : erreur image : {exc}")
        return None

    txt_path.write_text(text, encoding="utf-8")
    return img_path, txt_path


def collect_hf_dataset(
    repo_id: str,
    s3_prefix: str,
    storage,
    split: str = "train",
    max_samples: int = 3000,
    config: str | None = None,
) -> list[HFSample]:
    """Télécharge un dataset HuggingFace et uploade les paires image/texte sur S3.

    Args:
        repo_id: Identifiant HuggingFace (ex. ``"CATMuS/medieval"``).
        s3_prefix: Préfixe S3 de destination.
        storage: Instance :class:`~htmir.collection.s3_storage.S3Storage`.
        split: Split à charger (``"train"``, ``"validation"``, ``"test"``).
        max_samples: Nombre maximum de lignes à uploader.
        config: Sous-configuration du dataset si nécessaire.

    Returns:
        Liste de :class:`HFSample` uploadés avec succès.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        logger.error("La bibliothèque 'datasets' n'est pas installée.")
        return []

    logger.info(f"Chargement HuggingFace : {repo_id} (split={split}, max={max_samples})")

    try:
        ds = load_dataset(repo_id, config, split=split, trust_remote_code=False)
    except Exception as exc:
        logger.error(f"Impossible de charger {repo_id} : {exc}")
        return []

    # Détecter les colonnes image et texte (noms variables selon le dataset)
    available = set(ds.column_names)
    IMAGE_COLS = ["image", "im", "img", "scan", "page_image"]
    TEXT_COLS  = ["text", "transcription", "ground_truth", "label"]
    image_col = next((c for c in IMAGE_COLS if c in available), None)
    text_col  = next((c for c in TEXT_COLS  if c in available), None)
    if not image_col or not text_col:
        logger.error(f"{repo_id} : colonnes manquantes (disponibles : {available})")
        return []
    logger.info(f"{repo_id} : image='{image_col}', text='{text_col}'")

    import tempfile
    samples_out: list[HFSample] = []
    dataset_name = repo_id.replace("/", "_")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / dataset_name
        out_dir.mkdir()
        count = 0

        for idx, row in enumerate(ds):
            if count >= max_samples:
                break

            # Normaliser les noms de colonnes pour _save_sample_locally
            normalized = dict(row)
            if image_col != "image":
                normalized["image"] = normalized.pop(image_col)
            if text_col != "text":
                normalized["text"] = normalized.pop(text_col)

            # Skip si déjà sur S3
            img_key = f"{s3_prefix}/line_{idx:06d}.jpg"
            if storage.exists(img_key):
                logger.debug(f"Déjà sur S3 : {img_key}")
                count += 1
                continue

            result = _save_sample_locally(idx, normalized, out_dir, repo_id)
            if result is None:
                continue

            img_path, txt_path = result

            try:
                img_uri = storage.upload(img_path, img_key)
                txt_uri = storage.upload(txt_path, f"{s3_prefix}/line_{idx:06d}.txt")
            except Exception as exc:
                logger.error(f"Upload échoué sample {idx} : {exc}")
                continue

            samples_out.append(
                HFSample(
                    sample_id=f"{dataset_name}_line_{idx:06d}",
                    image_s3_uri=img_uri,
                    text_s3_uri=txt_uri,
                    text=row.get("text") or row.get("transcription", ""),
                    repo_id=repo_id,
                )
            )
            count += 1

            # Nettoyage local pour ne pas saturer /tmp
            img_path.unlink(missing_ok=True)
            txt_path.unlink(missing_ok=True)

            if count % 200 == 0:
                logger.info(f"{repo_id} : {count}/{max_samples} lignes uploadées")

    logger.info(f"{repo_id} terminé : {len(samples_out)} lignes uploadées sur S3")
    return samples_out


def collect_all_hf_datasets(
    storage,
    hf_configs: list[dict] | None = None,
) -> dict[str, list[HFSample]]:
    """Collecte tous les datasets HuggingFace configurés.

    Args:
        storage: Instance :class:`~htmir.collection.s3_storage.S3Storage`.
        hf_configs: Liste de configs (remplace ``_HF_DATASETS`` si fournie).

    Returns:
        Dictionnaire ``{repo_id: [HFSample, ...]}``.
    """
    configs = hf_configs or _HF_DATASETS
    results: dict[str, list[HFSample]] = {}

    for cfg in configs:
        samples = collect_hf_dataset(
            repo_id=cfg["repo_id"],
            s3_prefix=cfg["s3_prefix"],
            storage=storage,
            split=cfg.get("split", "train"),
            max_samples=cfg.get("max_samples", 2000),
            config=cfg.get("config"),
        )
        results[cfg["repo_id"]] = samples

    return results
