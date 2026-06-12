#!/usr/bin/env python3
"""Lance un SageMaker Processing Job GPU pour l'entraînement HTmiR.

Exécuté **localement**, ce script démarre un job AWS qui, dans le conteneur :
  1. prépare les données CATMuS (français XIIIe) ;
  2. fine-tune le modèle Kraken ;
  3. évalue (CER/WER) sur le test set ;
  4. uploade modèle + log + rapport sur S3.

On réutilise un Processing Job (et non un Training Job) pour pouvoir surcharger
l'``ContainerEntrypoint`` — cohérent avec ``sagemaker_collect.py``.

Usage :
    python infrastructure/sagemaker_train.py launch --config configs/training.yaml
    python infrastructure/sagemaker_train.py launch --config configs/training.yaml --wait
    python infrastructure/sagemaker_train.py status htmir-train-1717000000
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Charger .env avant les imports AWS (sans python-dotenv)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3  # noqa: E402
import yaml  # noqa: E402

from htmir.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_training_job_args(cfg: dict, job_name: str) -> dict:
    """Construit les arguments ``create_processing_job`` pour l'entraînement.

    Args:
        cfg: Contenu de ``training.yaml``.
        job_name: Nom unique du job.

    Returns:
        Dictionnaire prêt pour ``boto3.client("sagemaker").create_processing_job``.

    Raises:
        ValueError: si le rôle IAM SageMaker est introuvable.
    """
    sm = cfg.get("sagemaker", {})
    bucket = cfg["output"]["bucket"]
    region = cfg["output"].get("region", "eu-west-3")

    role_arn = sm.get("role_arn") or os.environ.get("SAGEMAKER_ROLE_ARN", "")
    if not role_arn:
        raise ValueError(
            "role_arn introuvable. Définissez SAGEMAKER_ROLE_ARN dans .env "
            "ou sagemaker.role_arn dans training.yaml."
        )

    return {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": sm.get("instance_type", "ml.g4dn.xlarge"),
                "InstanceCount": sm.get("instance_count", 1),
                "VolumeSizeInGB": sm.get("volume_size_gb", 50),
            }
        },
        "AppSpecification": {
            "ImageUri": sm["container_image"],
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/infrastructure/train_entrypoint.py",
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
                    "S3Uri": f"s3://{bucket}/configs/training.yaml",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "model",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/models/{job_name}/",
                        "LocalPath": "/opt/ml/processing/output/model",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "RoleArn": role_arn,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": sm.get("max_runtime_seconds", 86_400)
        },
        "Environment": {
            "HTMIR_BUCKET": bucket,
            "AWS_DEFAULT_REGION": region,
            "PYTHONUNBUFFERED": "1",
        },
    }


def upload_code_and_config(cfg_path: Path, cfg: dict) -> None:
    """Uploade le code source et la config vers S3 avant le lancement.

    Args:
        cfg_path: Chemin local de ``training.yaml``.
        cfg: Contenu parsé.
    """
    bucket = cfg["output"]["bucket"]
    region = cfg["output"].get("region", "eu-west-3")
    s3 = boto3.client("s3", region_name=region)

    s3.upload_file(str(cfg_path), bucket, "configs/training.yaml")
    logger.info(f"Config uploadée : s3://{bucket}/configs/training.yaml")

    project_root = cfg_path.parent.parent
    code_files = (
        list((project_root / "src").rglob("*.py"))
        + list((project_root / "infrastructure").rglob("*.py"))
        + [project_root / "pyproject.toml", project_root / "README.md"]
    )
    for fp in code_files:
        if not fp.exists():
            continue
        try:
            rel = fp.relative_to(project_root)
        except ValueError:
            continue
        s3.upload_file(str(fp), bucket, f"code/htmir/{rel.as_posix()}")
    logger.info(f"Code source uploadé : s3://{bucket}/code/htmir/")


def launch(config_path: Path, wait: bool = False, dry_run: bool = False) -> str:
    """Prépare et lance le job d'entraînement SageMaker.

    Args:
        config_path: Chemin vers ``training.yaml``.
        wait: Bloque jusqu'à la fin si ``True``.
        dry_run: Affiche les arguments sans rien lancer.

    Returns:
        Nom du job.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    region = cfg["output"].get("region", "eu-west-3")
    job_name = f"htmir-train-{int(time.time())}"
    job_args = build_training_job_args(cfg, job_name)

    if dry_run:
        import json
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
        waiter = client.get_waiter("processing_job_completed_or_stopped")
        waiter.wait(ProcessingJobName=job_name,
                    WaiterConfig={"Delay": 60, "MaxAttempts": 1440})
        desc = client.describe_processing_job(ProcessingJobName=job_name)
        logger.info(f"Statut final : {desc['ProcessingJobStatus']}")

    return job_name


def status(job_name: str, region: str = "eu-west-3") -> None:
    """Affiche l'état d'un job existant."""
    import json
    client = boto3.client("sagemaker", region_name=region)
    desc = client.describe_processing_job(ProcessingJobName=job_name)
    keys = ["ProcessingJobName", "ProcessingJobStatus", "FailureReason"]
    print(json.dumps({k: str(desc.get(k, "N/A")) for k in keys}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance l'entraînement HTmiR sur SageMaker.")
    sub = parser.add_subparsers(dest="command")

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p_launch.add_argument("--wait", action="store_true")
    p_launch.add_argument("--dry-run", action="store_true")

    p_status = sub.add_parser("status")
    p_status.add_argument("job_name")
    p_status.add_argument("--region", default="eu-west-3")

    args = parser.parse_args()
    if args.command == "status":
        status(args.job_name, args.region)
    else:
        config = getattr(args, "config", Path("configs/training.yaml"))
        job = launch(config, wait=getattr(args, "wait", False),
                     dry_run=getattr(args, "dry_run", False))
        print(f"\nJob : {job}")


if __name__ == "__main__":
    main()
