"""Fine-tuning d'un modèle HTR Kraken sur les lignes CATMuS français XIIIe.

Enchaîne deux étapes de l'outil ``ketos`` (fourni par Kraken) :

1. ``ketos compile`` — compile les couples ``*.png`` / ``*.gt.txt`` d'un split
   en un dataset binaire Arrow (chargement rapide à l'entraînement).
2. ``ketos train`` — fine-tune le modèle de base CATMuS sur le split train,
   évalue sur le split validation, avec early stopping.

Les constructeurs de commandes (``build_compile_cmd`` / ``build_train_cmd``)
sont des fonctions pures — l'exécution réelle passe par ``subprocess``.

Usage :
    python -m htmir.training.train_kraken --config configs/training.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from htmir.utils.logger import get_logger

logger = get_logger(__name__)


def run_logged_command(cmd: list[str], log_path: Path) -> None:
    """Exécute une commande en « tee-ant » sa sortie vers la console et un fichier.

    Le log d'entraînement est ensuite parsé par le dashboard pour tracer les
    courbes CER/loss et les markers d'early-stopping.

    Args:
        cmd: Commande à exécuter (argv).
        log_path: Fichier où enregistrer stdout+stderr.

    Raises:
        subprocess.CalledProcessError: si la commande échoue.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def build_compile_cmd(image_dir: Path, output_arrow: Path) -> list[str]:
    """Construit la commande ``ketos compile`` pour un répertoire de lignes.

    Args:
        image_dir: Répertoire contenant ``line_*.png`` + ``line_*.gt.txt``.
        output_arrow: Chemin du dataset Arrow de sortie.

    Returns:
        Liste argv prête pour ``subprocess.run``.
    """
    images = sorted(str(p) for p in image_dir.glob("*.png"))
    return [
        "ketos", "compile",
        "--format-type", "path",   # cherche les .gt.txt frères des images
        "--output", str(output_arrow),
        *images,
    ]


def build_train_cmd(
    train_arrow: Path,
    val_arrow: Path | None,
    output_name: str,
    hp: dict,
    base_model: str | None = None,
) -> list[str]:
    """Construit la commande ``ketos train`` (fine-tuning ou from scratch).

    Args:
        train_arrow: Dataset Arrow d'entraînement.
        val_arrow: Dataset Arrow de validation (``None`` → split interne).
        output_name: Préfixe du modèle de sortie.
        hp: Hyperparamètres (``epochs``, ``lr``, ``batch_size``,
            ``early_stopping_patience``, ``device``, ``workers``).
        base_model: Chemin du ``.mlmodel`` de base à fine-tuner (``None`` →
            entraînement from scratch).

    Returns:
        Liste argv prête pour ``subprocess.run``.
    """
    cmd = [
        "ketos", "train",
        "--format-type", "binary",
        "--output", output_name,
        "--device", str(hp.get("device", "cpu")),
        "--epochs", str(hp.get("epochs", 50)),
        "--lag", str(hp.get("early_stopping_patience", 10)),
        "--lrate", str(hp.get("lr", 1e-4)),
        "--batch-size", str(hp.get("batch_size", 32)),
        "--workers", str(hp.get("workers", 4)),
    ]
    if base_model:
        # Fine-tuning : charge le modèle de base et fusionne les alphabets
        cmd += ["--load", base_model, "--resize", "union"]
    if val_arrow is not None:
        cmd += ["--evaluation-files", str(val_arrow)]
    cmd.append(str(train_arrow))
    return cmd


def fetch_base_model(doi: str, dest_dir: Path) -> Path | None:
    """Télécharge le modèle de base Kraken via ``kraken get <DOI>``.

    Args:
        doi: DOI Zenodo du modèle (ex. ``"10.5281/zenodo.10592716"``).
        dest_dir: Répertoire où chercher le ``.mlmodel`` téléchargé.

    Returns:
        Chemin du ``.mlmodel`` récupéré, ou ``None`` en cas d'échec.
    """
    logger.info(f"Téléchargement du modèle de base : {doi}")
    try:
        subprocess.run(["kraken", "get", doi], check=True, cwd=dest_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error(f"Échec de `kraken get {doi}` : {exc}")
        return None
    models = sorted(dest_dir.glob("*.mlmodel"))
    return models[0] if models else None


def run(cfg: dict, data_dir: Path) -> Path:
    """Pipeline complet : compile des splits puis fine-tuning.

    Args:
        cfg: Contenu de ``training.yaml``.
        data_dir: Répertoire produit par ``prepare_catmus`` (sous-dossiers
            ``train/`` et ``validation/``).

    Returns:
        Chemin (préfixe) du modèle entraîné.
    """
    mcfg = cfg["model"]
    hp = cfg["training"]

    # 1. Compilation des datasets Arrow
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    train_arrow = data_dir / "train.arrow"
    val_arrow = data_dir / "val.arrow" if val_dir.exists() else None

    logger.info("Compilation du split train…")
    subprocess.run(build_compile_cmd(train_dir, train_arrow), check=True)
    if val_arrow is not None:
        logger.info("Compilation du split validation…")
        subprocess.run(build_compile_cmd(val_dir, val_arrow), check=True)

    # 2. Résolution du modèle de base
    base_model = mcfg.get("base_model_path") or None
    if not base_model and mcfg.get("base_model_doi"):
        fetched = fetch_base_model(mcfg["base_model_doi"], data_dir)
        base_model = str(fetched) if fetched else None

    # Le brief impose le fine-tuning : on refuse de partir from scratch en silence.
    require_finetuning = mcfg.get("require_finetuning", True)
    if base_model is None:
        if require_finetuning:
            raise RuntimeError(
                "Fine-tuning requis mais modèle de base introuvable "
                f"(doi={mcfg.get('base_model_doi')!r}, path={mcfg.get('base_model_path')!r}). "
                "Vérifiez le DOI/chemin, ou mettez model.require_finetuning=false "
                "pour autoriser un entraînement from scratch."
            )
        logger.warning("Modèle de base indisponible — entraînement from scratch (autorisé)")

    # 3. Fine-tuning (log capturé pour le dashboard)
    output_name = mcfg.get("output_name", "htmir-model")
    log_path = data_dir / "train.log"
    cmd = build_train_cmd(train_arrow, val_arrow, output_name, hp, base_model)
    logger.info(f"Lancement de l'entraînement (log → {log_path}) : {' '.join(cmd)}")
    run_logged_command(cmd, log_path)

    logger.info(f"Entraînement terminé — modèle : {output_name} | log : {log_path}")
    return Path(output_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune un modèle HTR Kraken sur CATMuS français XIIIe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Répertoire des lignes (défaut : output.local_dir de la config)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_dir = args.data_dir or Path(cfg["output"]["local_dir"])
    run(cfg, data_dir)


if __name__ == "__main__":
    main()
