"""Pipeline d'entraînement HTmiR **en local** (machine avec GPU).

Une seule commande qui enchaîne tout, sans SageMaker :

  1. récupère le dataset (download S3 si présent, sinon prepare HF + push S3) ;
  2. fine-tune Kraken sur GPU (``device: cuda:0`` dans la config) ;
  3. évalue (CER/WER) sur le test set ;
  4. (option) uploade le modèle entraîné sur S3.

Contrairement au conteneur SageMaker, l'environnement local installe Kraken
proprement (``pip install -e ".[train]"``) — pas de conflit torch/torchvision.

Usage :
    htmir-train-local --config configs/training.yaml
    htmir-train-local --config configs/training.yaml --skip-prepare   # data déjà là
    htmir-train-local --config configs/training.yaml --upload-model
"""

import argparse
from pathlib import Path

import yaml

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dataset(cfg: dict, data_dir: Path, use_s3: bool = True) -> None:
    """Garantit la présence du dataset en local.

    Tente d'abord un download depuis S3 ; si le split train est absent,
    prépare depuis HuggingFace puis (si ``use_s3``) pousse sur S3.

    Args:
        cfg: Config ``training.yaml``.
        data_dir: Répertoire local cible.
        use_s3: Active le download/upload S3.
    """
    from htmir.data.prepare_catmus import prepare

    out = cfg["output"]
    splits = cfg["dataset"].get("splits", ["train", "validation", "test"])

    if use_s3:
        import boto3
        from htmir.data.s3_sync import download_dataset
        s3 = boto3.client("s3", region_name=out.get("region", "eu-west-3"))
        logger.info("Tentative de récupération du dataset depuis S3…")
        download_dataset(out["bucket"], out["s3_prefix"], data_dir, splits, s3)

    if (data_dir / "train").exists() and any((data_dir / "train").glob("*.png")):
        logger.info(f"Dataset prêt en local : {data_dir}")
        return

    logger.info("Dataset incomplet → préparation depuis HuggingFace")
    prepare(cfg, overrides={"output_dir": str(data_dir), "push_s3": use_s3})


def find_best_model(model_prefix: Path) -> Path | None:
    """Trouve le meilleur modèle produit par ``ketos train``.

    Kraken écrit ``<prefix>_best.mlmodel`` ; à défaut on prend le dernier
    checkpoint ``<prefix>_*.mlmodel``.

    Args:
        model_prefix: Préfixe du modèle (sans extension).

    Returns:
        Chemin du modèle, ou ``None`` si aucun trouvé.
    """
    best = Path(f"{model_prefix}_best.mlmodel")
    if best.exists():
        return best
    candidates = sorted(Path().glob(f"{model_prefix.name}*.mlmodel"))
    return candidates[-1] if candidates else None


def run_pipeline(cfg: dict, data_dir: Path, use_s3: bool,
                 skip_prepare: bool, upload_model: bool) -> dict:
    """Exécute le pipeline local complet.

    Args:
        cfg: Config ``training.yaml``.
        data_dir: Répertoire des données.
        use_s3: Active les échanges S3 (download data / upload modèle).
        skip_prepare: Saute la récupération des données (déjà présentes).
        upload_model: Uploade le modèle entraîné sur S3 en fin de run.

    Returns:
        Récapitulatif ``{model, eval_report}``.
    """
    from htmir.training.train_kraken import run as train_run
    from htmir.eval.evaluate import run as eval_run

    if not skip_prepare:
        ensure_dataset(cfg, data_dir, use_s3=use_s3)

    # Entraînement (GPU via cfg["training"]["device"])
    model_prefix = train_run(cfg, data_dir)
    model = find_best_model(model_prefix)

    # Évaluation
    report_path = data_dir / "eval_report.json"
    test_arrow = data_dir / "test.arrow"
    if model and test_arrow.exists():
        eval_run(model, test_arrow, cfg["training"].get("device", "cuda:0"), report_path)
    else:
        logger.warning("Évaluation sautée (modèle ou test.arrow manquant)")

    # Upload modèle sur S3
    if upload_model and model and use_s3:
        import boto3
        out = cfg["output"]
        s3 = boto3.client("s3", region_name=out.get("region", "eu-west-3"))
        key = f"models/{cfg['model'].get('output_name', 'htmir')}/{model.name}"
        s3.upload_file(str(model), out["bucket"], key)
        logger.info(f"Modèle uploadé : s3://{out['bucket']}/{key}")

    return {"model": str(model) if model else None, "eval_report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement HTmiR en local (GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Répertoire des données (défaut : output.local_dir)")
    parser.add_argument("--no-s3", action="store_true",
                        help="Désactive tout échange S3 (data depuis HF, pas d'upload)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Suppose le dataset déjà présent en local")
    parser.add_argument("--upload-model", action="store_true",
                        help="Uploade le modèle entraîné sur S3")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_dir = args.data_dir or Path(cfg["output"]["local_dir"])
    result = run_pipeline(
        cfg, data_dir,
        use_s3=not args.no_s3,
        skip_prepare=args.skip_prepare,
        upload_model=args.upload_model,
    )
    logger.info(f"Pipeline terminé : {result}")


if __name__ == "__main__":
    main()
