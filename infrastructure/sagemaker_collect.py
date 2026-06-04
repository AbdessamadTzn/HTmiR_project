#!/usr/bin/env python3
"""Lance un SageMaker Processing Job pour la collecte de données HTmiR.

Ce script est exécuté **localement** (depuis votre poste ou CI/CD) pour
démarrer un job AWS SageMaker qui effectue le scraping et l'upload S3
dans le cloud, sans monopoliser votre machine.

Prérequis :
    - AWS CLI configuré (``aws configure``) ou variables d'env AWS_* définies.
    - Le rôle IAM SageMaker doit avoir les politiques :
        AmazonSageMakerFullAccess + AmazonS3FullAccess (ou politique fine-grained).
    - La dépendance ``boto3`` doit être installée (``pip install boto3``).

Usage :
    python infrastructure/sagemaker_collect.py --config configs/collection.yaml
    python infrastructure/sagemaker_collect.py --config configs/collection.yaml --wait
    python infrastructure/sagemaker_collect.py --status htmir-collect-1717000000
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Charger le .env avant tout autre import (sans dépendance python-dotenv)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Le script est dans infrastructure/ — ajouter src/ au path pour les imports htmir
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3  # noqa: E402
import yaml  # noqa: E402
from htmir.utils.logger import get_logger

logger = get_logger(__name__)

# Compte AWS DLC (Deep Learning Containers) — stable dans toutes les régions AWS
_DLC_ACCOUNT = "763104351884"


# ── Construction des arguments du job ────────────────────────────────────────


def _container_image(cfg: dict) -> str:
    """Retourne l'URI de l'image conteneur à utiliser.

    Priorité : valeur dans config > image DLC PyTorch CPU AWS.
    """
    if override := cfg.get("sagemaker", {}).get("container_image", ""):
        return override
    region = cfg.get("region", "eu-west-3")
    return (
        f"{_DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/"
        "pytorch-training:2.1.0-cpu-py310-ubuntu20.04-ec2"
    )


def build_processing_job_args(cfg: dict, job_name: str) -> dict:
    """Construit le dictionnaire d'arguments pour ``create_processing_job``.

    Le job :
      - Monte la config depuis S3 en lecture.
      - Exécute ``htmir-collect`` (installé via ``pip install .``).
      - Écrit le manifeste mis à jour dans S3 (via le module collect lui-même).

    Args:
        cfg: Contenu du fichier ``collection.yaml``.
        job_name: Nom unique du job (utilisé par SageMaker pour l'idempotence).

    Returns:
        Dictionnaire prêt pour ``boto3.client("sagemaker").create_processing_job(**args)``.
    """
    sm = cfg.get("sagemaker", {})
    bucket = cfg["bucket"]
    region = cfg.get("region", "eu-west-3")

    role_arn: str = sm.get("role_arn") or os.environ.get("SAGEMAKER_ROLE_ARN", "")
    if not role_arn:
        raise ValueError(
            "role_arn introuvable. Définissez SAGEMAKER_ROLE_ARN dans .env "
            "ou renseignez sagemaker.role_arn dans collection.yaml."
        )

    return {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": sm.get("instance_type", "ml.t3.medium"),
                "InstanceCount": sm.get("instance_count", 1),
                "VolumeSizeInGB": sm.get("volume_size_gb", 50),
            }
        },
        "AppSpecification": {
            "ImageUri": _container_image(cfg),
            # Script dédié — évite la limite SageMaker de 256 chars par argument
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/infrastructure/container_entrypoint.py",
            ],
        },
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/code/htmir/",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "config",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/configs/collection.yaml",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        # Le manifeste est pushé directement vers S3 par le script collect.py.
        # SageMaker exige au moins une sortie — on mappe le dossier de logs.
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "logs",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/logs/collection/",
                        "LocalPath": "/opt/ml/processing/output/logs",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "RoleArn": role_arn,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": sm.get("max_runtime_seconds", 14_400)
        },
        "Environment": {
            "HTMIR_BUCKET": bucket,
            "AWS_DEFAULT_REGION": region,
            "PYTHONUNBUFFERED": "1",
        },
    }


# ── Upload du code source ─────────────────────────────────────────────────────


def upload_code_and_config(cfg_path: Path, cfg: dict) -> None:
    """Uploade le code source et la config vers S3 avant le lancement.

    Le code est uploadé sous ``s3://{bucket}/code/htmir/`` de façon à être
    monté dans le conteneur SageMaker à ``/opt/ml/processing/input/code``.

    Args:
        cfg_path: Chemin local du fichier ``collection.yaml``.
        cfg: Contenu parsé de ``collection.yaml``.
    """
    bucket = cfg["bucket"]
    region = cfg.get("region", "eu-west-3")
    s3 = boto3.client("s3", region_name=region)

    # Config
    s3.upload_file(str(cfg_path), bucket, "configs/collection.yaml")
    logger.info(f"Config uploadée : s3://{bucket}/configs/collection.yaml")

    # Code source (src/ + infrastructure/ + pyproject.toml)
    project_root = cfg_path.parent.parent
    code_files = (
        list((project_root / "src").rglob("*.py"))
        + list((project_root / "src").rglob("*.toml"))
        + list((project_root / "infrastructure").rglob("*.py"))
        + [project_root / "pyproject.toml"]
        + [project_root / "README.md"]
    )
    for file_path in code_files:
        if not file_path.exists():
            continue
        try:
            relative = file_path.relative_to(project_root)
        except ValueError:
            continue
        key = f"code/htmir/{relative.as_posix()}"
        s3.upload_file(str(file_path), bucket, key)

    logger.info(f"Code source uploadé : s3://{bucket}/code/htmir/")


# ── Lancement et suivi ────────────────────────────────────────────────────────


def launch_collection_job(
    config_path: Path,
    wait: bool = False,
    dry_run: bool = False,
) -> str:
    """Prépare et lance le SageMaker Processing Job de collecte.

    Args:
        config_path: Chemin vers ``configs/collection.yaml``.
        wait: Bloque jusqu'à la fin du job si ``True``.
        dry_run: Affiche les arguments sans lancer le job.

    Returns:
        Nom du job lancé (ou simulé en dry-run).
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    region = cfg.get("region", "eu-west-3")
    job_name = f"htmir-collect-{int(time.time())}"

    job_args = build_processing_job_args(cfg, job_name)

    if dry_run:
        import json
        logger.info("Dry-run — arguments du job :")
        print(json.dumps(job_args, indent=2))
        return job_name

    upload_code_and_config(config_path, cfg)

    client = boto3.client("sagemaker", region_name=region)
    client.create_processing_job(**job_args)
    logger.info(
        f"Job lancé : {job_name}\n"
        f"  Console : https://{region}.console.aws.amazon.com/sagemaker/home"
        f"?region={region}#/processing-jobs/{job_name}"
    )

    if wait:
        logger.info("Attente de la fin du job (Ctrl+C pour annuler l'attente)…")
        waiter = client.get_waiter("processing_job_completed_or_stopped")
        waiter.wait(ProcessingJobName=job_name, WaiterConfig={"Delay": 30, "MaxAttempts": 480})
        desc = client.describe_processing_job(ProcessingJobName=job_name)
        final_status = desc["ProcessingJobStatus"]
        logger.info(f"Job terminé avec le statut : {final_status}")
        if final_status != "Completed":
            logger.error(f"Échec du job : {desc.get('FailureReason', 'N/A')}")

    return job_name


def describe_job(job_name: str, region: str = "eu-west-3") -> None:
    """Affiche l'état courant d'un Processing Job existant."""
    import json
    client = boto3.client("sagemaker", region_name=region)
    desc = client.describe_processing_job(ProcessingJobName=job_name)
    keys = [
        "ProcessingJobName", "ProcessingJobStatus",
        "CreationTime", "ProcessingEndTime", "FailureReason",
    ]
    summary = {k: str(desc.get(k, "N/A")) for k in keys}
    print(json.dumps(summary, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance (ou inspecte) un SageMaker Processing Job de collecte HTmiR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # Sous-commande launch (défaut)
    launch_p = sub.add_parser("launch", help="Démarre un nouveau job de collecte")
    launch_p.add_argument("--config", type=Path, default=Path("configs/collection.yaml"))
    launch_p.add_argument("--wait", action="store_true", help="Attend la fin du job")
    launch_p.add_argument("--dry-run", action="store_true", help="Affiche les args sans lancer")

    # Sous-commande status
    status_p = sub.add_parser("status", help="Affiche l'état d'un job existant")
    status_p.add_argument("job_name", help="Nom du Processing Job")
    status_p.add_argument("--region", default="eu-west-3")

    args = parser.parse_args()

    if args.command == "status":
        describe_job(args.job_name, args.region)
    else:
        # launch est la commande par défaut
        config_path = getattr(args, "config", Path("configs/collection.yaml"))
        wait = getattr(args, "wait", False)
        dry_run = getattr(args, "dry_run", False)
        job_name = launch_collection_job(config_path, wait=wait, dry_run=dry_run)
        print(f"\nJob : {job_name}")


if __name__ == "__main__":
    main()
