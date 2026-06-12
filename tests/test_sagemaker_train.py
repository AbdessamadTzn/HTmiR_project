"""Tests unitaires pour le lanceur infrastructure/sagemaker_train.py."""

import importlib.util
from pathlib import Path

import pytest

# Import du module hors package (dans infrastructure/)
_SPEC = importlib.util.spec_from_file_location(
    "sagemaker_train",
    Path(__file__).parent.parent / "infrastructure" / "sagemaker_train.py",
)
sagemaker_train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sagemaker_train)


@pytest.fixture
def cfg():
    return {
        "output": {"bucket": "htmir-data", "region": "eu-west-3"},
        "training": {"device": "cuda:0"},
        "sagemaker": {
            "role_arn": "arn:aws:iam::123:role/SageMakerRole",
            "instance_type": "ml.g4dn.xlarge",
            "instance_count": 1,
            "volume_size_gb": 50,
            "max_runtime_seconds": 86400,
            "container_image": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch:gpu",
        },
    }


def test_build_args_basic_structure(cfg):
    args = sagemaker_train.build_training_job_args(cfg, "htmir-train-1")
    assert args["ProcessingJobName"] == "htmir-train-1"
    assert args["RoleArn"].endswith("SageMakerRole")
    assert args["AppSpecification"]["ImageUri"].endswith("pytorch:gpu")


def test_build_args_uses_gpu_instance(cfg):
    args = sagemaker_train.build_training_job_args(cfg, "j")
    cluster = args["ProcessingResources"]["ClusterConfig"]
    assert cluster["InstanceType"] == "ml.g4dn.xlarge"


def test_build_args_entrypoint_is_train_script(cfg):
    args = sagemaker_train.build_training_job_args(cfg, "j")
    entry = args["AppSpecification"]["ContainerEntrypoint"]
    assert entry[-1].endswith("infrastructure/train_entrypoint.py")


def test_build_args_output_path_includes_job_name(cfg):
    args = sagemaker_train.build_training_job_args(cfg, "htmir-train-42")
    out = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    assert "models/htmir-train-42/" in out


def test_build_args_role_from_env(cfg, monkeypatch):
    """Si role_arn absent de la config, on prend SAGEMAKER_ROLE_ARN."""
    cfg["sagemaker"]["role_arn"] = ""
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::999:role/EnvRole")
    args = sagemaker_train.build_training_job_args(cfg, "j")
    assert args["RoleArn"].endswith("EnvRole")


def test_build_args_raises_without_role(cfg, monkeypatch):
    """Sans rôle nulle part, on lève ValueError explicite."""
    cfg["sagemaker"]["role_arn"] = ""
    monkeypatch.delenv("SAGEMAKER_ROLE_ARN", raising=False)
    with pytest.raises(ValueError, match="role_arn"):
        sagemaker_train.build_training_job_args(cfg, "j")


def test_build_args_environment_vars(cfg):
    args = sagemaker_train.build_training_job_args(cfg, "j")
    env = args["Environment"]
    assert env["HTMIR_BUCKET"] == "htmir-data"
    assert env["AWS_DEFAULT_REGION"] == "eu-west-3"
