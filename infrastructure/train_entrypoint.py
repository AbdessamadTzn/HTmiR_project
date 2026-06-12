"""Point d'entrée du conteneur SageMaker pour l'entraînement HTmiR.

Exécuté comme :
    python3 /opt/ml/processing/input/code/infrastructure/train_entrypoint.py

Étapes :
  1. installe le package + Kraken ;
  2. prépare les données CATMuS (français XIIIe) en local ;
  3. fine-tune le modèle Kraken (log capturé) ;
  4. évalue (CER/WER) ;
  5. copie modèle + log + rapport dans le dossier de sortie SageMaker (→ S3).
"""

import subprocess
import sys
from pathlib import Path

CODE_DIR = "/opt/ml/processing/input/code"
CONFIG_PATH = "/opt/ml/processing/input/config/training.yaml"
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
DATA_DIR = Path("/opt/ml/processing/data/catmus-french-13c")
TRAIN_LOG = OUTPUT_DIR / "train.log"


def _run(cmd: list[str], log_file=None) -> None:
    """Lance une commande, en streamant éventuellement la sortie vers un log."""
    print(f"+ {' '.join(cmd)}", flush=True)
    if log_file is None:
        subprocess.run(cmd, check=True)
        return
    with open(log_file, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Installation (package sans deps lourdes + kraken)
    _run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", CODE_DIR])
    _run([sys.executable, "-m", "pip", "install", "--quiet",
          "duckdb>=1.0.0", "pillow>=10.3.0", "pyyaml>=6.0",
          "requests>=2.32.0", "boto3>=1.34.0", "kraken>=5.2"])

    sys.path.insert(0, str(Path(CODE_DIR) / "src"))
    import boto3
    import yaml
    from htmir.data.prepare_catmus import prepare
    from htmir.data.s3_sync import download_dataset
    from htmir.training.train_kraken import run as train_run
    from htmir.eval.evaluate import run as eval_run

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    out_cfg = cfg["output"]
    splits = cfg["dataset"].get("splits", ["train", "validation", "test"])
    s3 = boto3.client("s3", region_name=out_cfg.get("region", "eu-west-3"))

    # 2. Dataset : on lit la source de vérité S3 ; si absente, on prépare
    #    depuis HuggingFace puis on persiste sur S3 (idempotent).
    download_dataset(out_cfg["bucket"], out_cfg["s3_prefix"], DATA_DIR, splits, s3)
    if not (DATA_DIR / "train").exists():
        print("Dataset absent de S3 → préparation depuis HuggingFace + push S3", flush=True)
        prepare(cfg, overrides={"output_dir": str(DATA_DIR), "push_s3": True})

    # 3. Fine-tuning (log capturé pour le dashboard)
    model_prefix = train_run(cfg, DATA_DIR)

    # 4. Évaluation sur le test set
    test_arrow = DATA_DIR / "test.arrow"
    if test_arrow.exists():
        model_file = Path(f"{model_prefix}_best.mlmodel")
        if not model_file.exists():
            candidates = sorted(Path().glob(f"{model_prefix}*.mlmodel"))
            model_file = candidates[-1] if candidates else None
        if model_file:
            eval_run(model_file, test_arrow, cfg["training"].get("device", "cuda:0"),
                     OUTPUT_DIR / "eval_report.json")

    # 5. Copie des artefacts vers la sortie SageMaker (→ S3)
    for model in Path().glob("*.mlmodel"):
        (OUTPUT_DIR / model.name).write_bytes(model.read_bytes())

    print(f"Artefacts dans {OUTPUT_DIR} → upload S3 automatique en fin de job", flush=True)


if __name__ == "__main__":
    main()
