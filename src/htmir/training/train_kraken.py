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
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from htmir.utils.logger import get_logger

logger = get_logger(__name__)

_COMPILE_MANIFEST = "_compile_manifest.txt"
_ARROW_MANIFEST_SUFFIX = "_ketos_manifest.txt"


def _subprocess_env() -> dict[str, str]:
    """Environnement subprocess avec encodage UTF-8 (console Windows)."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


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
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_subprocess_env(),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def _venv_script(name: str) -> str:
    """Résout un exécutable CLI du venv (``ketos``, ``kraken``, …)."""
    found = shutil.which(name)
    if found:
        return found
    exe = f"{name}.exe" if sys.platform == "win32" else name
    candidate = Path(sys.executable).with_name(exe)
    return str(candidate) if candidate.exists() else name


def _ketos_bin() -> str:
    return _venv_script("ketos")


def _kraken_bin() -> str:
    return _venv_script("kraken")


def _dataloader_workers(hp: dict) -> int:
    """Nombre de workers DataLoader (0 sous Windows : spawn multiprocessing fragile)."""
    requested = int(hp.get("workers", 4))
    if sys.platform == "win32" and requested > 0:
        logger.warning(
            "workers=%s ignoré sous Windows (DataLoader multiprocessing instable) — utilisation de 0",
            requested,
        )
        return 0
    return requested


def write_compile_manifest(image_dir: Path, manifest_path: Path | None = None) -> Path:
    """Écrit un manifeste texte (un chemin PNG par ligne) pour ``ketos compile``.

    Évite de dépasser la limite de longueur de ligne de commande Windows
    (~8k car.) quand un split contient des milliers d'images.

    Args:
        image_dir: Répertoire contenant ``line_*.png`` + ``line_*.gt.txt``.
        manifest_path: Fichier manifeste (défaut : ``<image_dir>/_compile_manifest.txt``).

    Returns:
        Chemin du manifeste écrit.
    """
    manifest = manifest_path or image_dir / _COMPILE_MANIFEST
    images = sorted(image_dir.glob("*.png"))
    manifest.write_text(
        "\n".join(str(p.resolve()) for p in images),
        encoding="utf-8",
    )
    logger.info(f"Manifeste compile : {len(images)} image(s) -> {manifest}")
    return manifest


def compile_split(image_dir: Path, output_arrow: Path) -> None:
    """Compile un répertoire de lignes ``*.png`` / ``*.gt.txt`` en dataset Arrow.

    Utilise l'API Python de Kraken (plus fiable que ``ketos compile`` sous
    Windows, où la ligne de commande et la barre de progression Rich plantent).
    """
    from kraken.lib.arrow_dataset import build_binary_dataset

    images = sorted(str(p.resolve()) for p in image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"Aucune image PNG dans {image_dir}")
    logger.info(f"Compilation Arrow : {len(images)} ligne(s) -> {output_arrow}")
    build_binary_dataset(
        files=images,
        output_file=str(output_arrow),
        format_type="path",
        num_workers=0,
        skip_empty_lines=True,
    )


def write_arrow_manifest(arrow_path: Path) -> Path:
    """Écrit un manifeste texte pointant vers un dataset Arrow (Kraken 7 ``-e``).

    ``ketos train -e`` attend un fichier texte (un chemin par ligne), pas un
    ``.arrow`` binaire directement.
    """
    manifest = arrow_path.with_name(f"{arrow_path.stem}{_ARROW_MANIFEST_SUFFIX}")
    manifest.write_text(f"{arrow_path.resolve()}\n", encoding="utf-8")
    return manifest


def build_compile_cmd(manifest_path: Path, output_arrow: Path) -> list[str]:
    """Construit la commande ``ketos compile`` à partir d'un manifeste.

    Args:
        manifest_path: Fichier texte listant les PNG (un par ligne).
        output_arrow: Chemin du dataset Arrow de sortie.

    Returns:
        Liste argv prête pour ``subprocess.run``.
    """
    return [
        _ketos_bin(), "compile",
        "--format-type", "path",   # cherche les .gt.txt frères des images
        "--output", str(output_arrow),
        "--files", str(manifest_path),
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
    # Kraken 7+ : device/workers sont des options globales de ``ketos``,
    # pas de la sous-commande ``train`` (cf. ``ketos -d cuda:0 train ...``).
    cmd = [
        _ketos_bin(),
        "-d", str(hp.get("device", "cpu")),
        "--workers", str(hp.get("workers", 4)),
        "train",
        "-f", "binary",
        "-o", output_name,
        "-N", str(hp.get("epochs", 50)),
        "--lag", str(hp.get("early_stopping_patience", 10)),
        "-r", str(hp.get("lr", 1e-4)),
        "-B", str(hp.get("batch_size", 32)),
    ]
    if base_model:
        # Fine-tuning : charge le modèle de base et fusionne les alphabets
        cmd += ["-i", base_model, "--resize", "union"]
    if val_arrow is not None:
        cmd += ["-e", str(write_arrow_manifest(val_arrow))]
    cmd.append(str(train_arrow.resolve()))
    return cmd


def _patch_kraken_binary_dataloader() -> None:
    """Corrige le chargement des datasets Arrow binaires sous Kraken 7.

    ``VGSLRecognitionDataModule._build_dataset`` passe ``im_transforms=None``,
    ce qui fait échouer silencieusement ``ArrowIPCRecognitionDataset.add()``
    (``self.transforms.valid_norm`` sur ``None``) et produit un jeu vide.
    """
    from kraken.train.vgsl import VGSLRecognitionDataModule
    from torchvision.transforms import v2

    if getattr(VGSLRecognitionDataModule, "_htmir_patched", False):
        return

    def _build_dataset_fixed(self, dataset_cls, training_data, **kwargs):
        dataset = dataset_cls(
            normalization=self.hparams.data_config.normalization,
            whitespace_normalization=self.hparams.data_config.normalize_whitespace,
            reorder=self.hparams.data_config.bidi_reordering,
            im_transforms=v2.Identity(),
            **kwargs,
        )
        for sample in training_data:
            try:
                dataset.add(**sample)
            except Exception as exc:
                logger.warning(str(exc))
        dc = self.hparams.data_config
        if dc.format_type == "binary" and (
            dc.normalization or dc.normalize_whitespace or dc.bidi_reordering
        ):
            dataset.rebuild_alphabet()
        return dataset

    VGSLRecognitionDataModule._build_dataset = _build_dataset_fixed  # type: ignore[method-assign]
    VGSLRecognitionDataModule._htmir_patched = True


class _TeeStream:
    """Duplique les écritures vers la console et un fichier log."""

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return self._stream.isatty()


def run_kraken_train(
    train_arrow: Path,
    val_arrow: Path | None,
    output_name: str,
    hp: dict,
    base_model: str | None,
    log_path: Path,
) -> None:
    """Lance l'entraînement Kraken via l'API Python (contourne bugs CLI Windows).

    Réplique ``ketos train`` avec le correctif du dataloader binaire et des
    chemins absolus pour les fichiers Arrow.
    """
    from pathlib import Path as PathCls
    from threading import local

    from kraken.configs import (
        VGSLRecognitionTrainingConfig,
        VGSLRecognitionTrainingDataConfig,
    )
    from kraken.ketos.util import to_ptl_device
    from kraken.models.convert import convert_models
    from kraken.train import KrakenTrainer, VGSLRecognitionDataModule, VGSLRecognitionModel
    from kraken.train.utils import KrakenOnExceptionCheckpoint
    from lightning.pytorch.callbacks import ModelCheckpoint
    from threadpoolctl import threadpool_limits

    _patch_kraken_binary_dataloader()

    train_path = str(train_arrow.resolve())
    eval_paths = [str(val_arrow.resolve())] if val_arrow is not None else None
    checkpoint_path = PathCls(output_name)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    params: dict = {
        "format_type": "binary",
        "training_data": [train_path],
        "evaluation_data": eval_paths,
        "partition": 1.0 if eval_paths else 0.9,
        "checkpoint_path": str(checkpoint_path),
        "epochs": hp.get("epochs", 50),
        "lag": hp.get("early_stopping_patience", 10),
        "lrate": hp.get("lr", 1e-4),
        "batch_size": hp.get("batch_size", 32),
        "num_workers": _dataloader_workers(hp),
        "resize": "union" if base_model else "fail",
        "quit": "early",
    }

    device = str(hp.get("device", "cpu"))
    accelerator, devices = to_ptl_device(device)

    freq = 1.0
    val_check_interval = (
        {"check_val_every_n_epoch": int(freq)}
        if freq > 1
        else {"val_check_interval": freq}
    )

    dm_config = VGSLRecognitionTrainingDataConfig(**params)
    m_config = VGSLRecognitionTrainingConfig(**params)

    data_module = VGSLRecognitionDataModule(dm_config)
    if len(data_module.train_set) == 0:
        raise RuntimeError(
            f"Jeu d'entraînement vide après chargement de {train_path}. "
            "Vérifiez que train.arrow est valide."
        )

    cbs = [
        KrakenOnExceptionCheckpoint(
            dirpath=str(checkpoint_path),
            filename="checkpoint_abort",
        ),
    ]
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=10,
        monitor="val_metric",
        mode="max",
        auto_insert_metric_name=False,
        filename="checkpoint_{epoch:02d}-{val_metric:.4f}",
    )
    cbs.append(checkpoint_callback)

    trainer = KrakenTrainer(
        accelerator=accelerator,
        devices=devices,
        precision="32-true",
        max_epochs=params["epochs"],
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=cbs,
        **val_check_interval,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_local = local()

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_local.tee = _TeeStream(sys.stdout, log_fh)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log_local.tee
        try:
            logger.info(
                "Entraînement Kraken (API) : %s, eval=%s, sortie=%s",
                train_path,
                eval_paths,
                checkpoint_path,
            )
            with trainer.init_module(empty_init=base_model is None):
                if base_model:
                    logger.info("Chargement du modèle de base : %s", base_model)
                    load = str(PathCls(base_model).resolve())
                    if load.endswith("ckpt"):
                        model = VGSLRecognitionModel.load_from_checkpoint(
                            load, config=m_config, weights_only=False,
                        )
                    else:
                        model = VGSLRecognitionModel.load_from_weights(load, config=m_config)
                else:
                    logger.info("Initialisation d'un nouveau modèle.")
                    model = VGSLRecognitionModel(m_config)

            with threadpool_limits(limits=1):
                trainer.fit(model, data_module)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    if checkpoint_callback.best_model_path:
        score = checkpoint_callback.best_model_score.item()
        weight_path = PathCls(checkpoint_callback.best_model_path).with_name(
            f"best_{score:.4f}.{params.get('weights_format', 'safetensors')}",
        )
        convert_models(
            [checkpoint_callback.best_model_path],
            weight_path,
            weights_format=params.get("weights_format", "safetensors"),
        )
        logger.info("Meilleur modèle : %s (score=%.4f)", weight_path, score)


def _find_mlmodel(dest_dir: Path) -> Path | None:
    models = sorted(dest_dir.glob("*.mlmodel"))
    return models[0] if models else None


def _zenodo_record_id(doi: str) -> str | None:
    prefix = "10.5281/zenodo."
    if doi.startswith(prefix):
        return doi[len(prefix):]
    return None


def fetch_base_model_zenodo(doi: str, dest_dir: Path) -> Path | None:
    """Télécharge un ``.mlmodel`` depuis l'API Zenodo (fallback si ``kraken get`` échoue)."""
    import requests

    record_id = _zenodo_record_id(doi)
    if not record_id:
        logger.warning(f"DOI non Zenodo, pas de fallback direct : {doi}")
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60).json()
    except requests.RequestException as exc:
        logger.error(f"API Zenodo indisponible pour {doi} : {exc}")
        return None

    files = [f for f in meta.get("files", []) if f.get("key", "").endswith(".mlmodel")]
    if not files:
        logger.error(f"Aucun .mlmodel dans l'enregistrement Zenodo {record_id}")
        return None

    entry = files[0]
    dest = dest_dir / entry["key"]
    expected_size = entry.get("size")
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        logger.info(f"Modèle déjà présent : {dest}")
        return dest

    url = entry["links"]["self"]
    logger.info(f"Téléchargement Zenodo : {entry['key']} ({expected_size or '?'} o)")
    try:
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        logger.error(f"Échec téléchargement Zenodo {entry['key']} : {exc}")
        dest.unlink(missing_ok=True)
        return None
    return dest


def fetch_base_model(doi: str, dest_dir: Path) -> Path | None:
    """Télécharge le modèle de base Kraken via ``kraken get``, puis Zenodo en secours.

    Args:
        doi: DOI Zenodo du modèle (ex. ``"10.5281/zenodo.10592716"``).
        dest_dir: Répertoire où chercher le ``.mlmodel`` téléchargé.

    Returns:
        Chemin du ``.mlmodel`` récupéré, ou ``None`` en cas d'échec.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = _find_mlmodel(dest_dir)
    if cached:
        logger.info(f"Modèle de base déjà en cache : {cached}")
        return cached

    logger.info(f"Téléchargement du modèle de base : {doi}")
    try:
        subprocess.run(
            [_kraken_bin(), "get", doi],
            check=True,
            cwd=dest_dir,
            env=_subprocess_env(),
        )
        found = _find_mlmodel(dest_dir)
        if found:
            return found
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"`kraken get {doi}` a échoué ({exc}) — tentative Zenodo")

    return fetch_base_model_zenodo(doi, dest_dir)


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

    if train_arrow.exists():
        logger.info(f"Réutilisation du dataset train existant : {train_arrow}")
    else:
        logger.info("Compilation du split train…")
        compile_split(train_dir, train_arrow)
    if val_arrow is not None:
        if val_arrow.exists():
            logger.info(f"Réutilisation du dataset validation existant : {val_arrow}")
        else:
            logger.info("Compilation du split validation…")
            compile_split(val_dir, val_arrow)

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

    # 3. Fine-tuning (API Python : évite bugs ketos CLI + dataloader binaire Kraken 7)
    output_name = mcfg.get("output_name", "htmir-model")
    log_path = data_dir / "train.log"
    logger.info(f"Lancement de l'entraînement (log → {log_path})")
    run_kraken_train(train_arrow, val_arrow, output_name, hp, base_model, log_path)

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
