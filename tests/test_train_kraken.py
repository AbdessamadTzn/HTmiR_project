"""Tests unitaires pour htmir.training.train_kraken."""

from pathlib import Path

import pytest

from htmir.training import train_kraken


# ── build_compile_cmd ───────────────────────────────────────────────────────


def test_write_compile_manifest_lists_sorted_pngs(tmp_path):
    """Le manifeste doit contenir les PNG triés, un par ligne."""
    (tmp_path / "line_000001.png").touch()
    (tmp_path / "line_000000.png").touch()
    manifest = train_kraken.write_compile_manifest(tmp_path)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines == sorted(lines)


def test_build_compile_cmd_uses_manifest_file(tmp_path):
    """La commande compile passe les chemins via --files (pas en argv)."""
    (tmp_path / "line_000000.png").touch()
    manifest = train_kraken.write_compile_manifest(tmp_path)
    out = tmp_path / "train.arrow"

    cmd = train_kraken.build_compile_cmd(manifest, out)

    assert cmd[1] == "compile"
    assert "--format-type" in cmd and "path" in cmd
    assert "--output" in cmd
    assert str(out) in cmd
    assert "--files" in cmd
    assert str(manifest) in cmd
    assert not any(c.endswith(".png") for c in cmd)


# ── build_train_cmd ─────────────────────────────────────────────────────────


def test_build_train_cmd_finetune_includes_base_model(tmp_path):
    """En fine-tuning, -i et --resize union doivent être présents (Kraken 7)."""
    hp = {"epochs": 30, "lr": 1e-4, "batch_size": 16, "device": "cuda:0",
          "early_stopping_patience": 5, "workers": 2}
    val_arrow = tmp_path / "val.arrow"
    val_arrow.touch()
    cmd = train_kraken.build_train_cmd(
        tmp_path / "train.arrow", val_arrow, "out", hp,
        base_model="base.mlmodel",
    )
    assert cmd.index("train") > cmd.index("-d")
    assert "-i" in cmd
    assert "base.mlmodel" in cmd
    assert "--resize" in cmd and "union" in cmd
    assert "-e" in cmd
    eval_manifest = tmp_path / "val_ketos_manifest.txt"
    assert str(eval_manifest) in cmd
    assert eval_manifest.read_text(encoding="utf-8").strip().endswith("val.arrow")
    assert cmd[-1] == str((tmp_path / "train.arrow").resolve())


def test_build_train_cmd_from_scratch_no_load():
    """Sans modèle de base, pas de -i ni --resize."""
    cmd = train_kraken.build_train_cmd(
        Path("train.arrow"), None, "out", {}, base_model=None,
    )
    assert "-i" not in cmd
    assert "--resize" not in cmd
    assert "-e" not in cmd   # pas de val


def test_build_train_cmd_hyperparams_propagated():
    """Les hyperparamètres de la config doivent passer dans la commande."""
    hp = {"epochs": 42, "lr": 0.0003, "batch_size": 64, "device": "cpu",
          "early_stopping_patience": 7, "workers": 8}
    cmd = train_kraken.build_train_cmd(Path("t.arrow"), None, "m", hp)
    assert "42" in cmd                      # epochs
    assert "0.0003" in cmd                  # lr
    assert "64" in cmd                      # batch
    assert "cpu" in cmd                     # device
    assert "7" in cmd                       # patience


# ── fetch_base_model ────────────────────────────────────────────────────────


def test_fetch_base_model_returns_none_on_missing_kraken(tmp_path, monkeypatch):
    """Si `kraken` n'est pas installé et Zenodo échoue, fetch_base_model retourne None."""
    def boom(*a, **k):
        raise FileNotFoundError("kraken introuvable")
    monkeypatch.setattr(train_kraken.subprocess, "run", boom)
    monkeypatch.setattr(train_kraken, "fetch_base_model_zenodo", lambda doi, d: None)
    assert train_kraken.fetch_base_model("10.5281/zenodo.1", tmp_path) is None


def test_fetch_base_model_zenodo_downloads_mlmodel(tmp_path, monkeypatch):
    """Le fallback Zenodo écrit le .mlmodel attendu."""
    import requests

    class MetaResp:
        def json(self):
            return {
                "files": [{
                    "key": "catmus.mlmodel",
                    "size": 11,
                    "links": {"self": "https://zenodo.example/dl"},
                }],
            }

    class DownloadResp:
        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=0):
            yield b"model-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_get(url, **kw):
        if "api/records" in url:
            return MetaResp()
        return DownloadResp()

    monkeypatch.setattr(requests, "get", fake_get)
    result = train_kraken.fetch_base_model_zenodo("10.5281/zenodo.99", tmp_path)
    assert result == tmp_path / "catmus.mlmodel"
    assert result.read_bytes() == b"model-bytes"


def test_fetch_base_model_finds_downloaded_model(tmp_path, monkeypatch):
    """Après un `kraken get` réussi, le .mlmodel téléchargé est retourné."""
    def fake_run(*a, **k):
        (tmp_path / "catmus.mlmodel").touch()
        class R:  # noqa: D401
            returncode = 0
        return R()
    monkeypatch.setattr(train_kraken.subprocess, "run", fake_run)
    result = train_kraken.fetch_base_model("10.5281/zenodo.1", tmp_path)
    assert result == tmp_path / "catmus.mlmodel"


# ── run (orchestration, subprocess mocké) ───────────────────────────────────


def test_run_compiles_and_trains(tmp_path, monkeypatch):
    """run() compile train+val puis entraîne via run_kraken_train."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "validation").mkdir(parents=True)
    (data_dir / "train" / "line_000000.png").touch()
    (data_dir / "validation" / "line_000000.png").touch()

    compiles = []
    trains = []
    monkeypatch.setattr(train_kraken, "compile_split",
                        lambda img_dir, out: compiles.append((img_dir, out)))
    monkeypatch.setattr(
        train_kraken,
        "run_kraken_train",
        lambda train_a, val_a, out, hp, base, log: trains.append(
            (train_a, val_a, out, hp, base, log),
        ),
    )

    cfg = {
        "model": {"base_model_path": "base.mlmodel", "output_name": "htmir-french-13c"},
        "training": {"epochs": 1, "device": "cpu"},
    }
    train_kraken.run(cfg, data_dir)

    # 2 compile (train + val)
    assert len(compiles) == 2
    assert compiles[0][0].name == "train"
    assert compiles[1][0].name == "validation"
    # 1 entraînement API, fine-tuning avec modèle de base
    assert len(trains) == 1
    train_a, val_a, out, _hp, base, log_path = trains[0]
    assert train_a == data_dir / "train.arrow"
    assert val_a == data_dir / "val.arrow"
    assert out == "htmir-french-13c"
    assert base == "base.mlmodel"
    assert log_path == data_dir / "train.log"


def test_run_logged_command_writes_log(tmp_path):
    """run_logged_command capture la sortie dans un fichier ET ne lève pas si OK."""
    log = tmp_path / "sub" / "train.log"
    train_kraken.run_logged_command(["echo", "epoch 1 done"], log)
    assert log.exists()
    assert "epoch 1 done" in log.read_text(encoding="utf-8")


def test_run_logged_command_raises_on_failure(tmp_path):
    """Une commande qui échoue lève CalledProcessError."""
    import subprocess
    log = tmp_path / "train.log"
    with pytest.raises(subprocess.CalledProcessError):
        train_kraken.run_logged_command(["false"], log)


def test_run_raises_when_finetuning_required_but_no_base(tmp_path, monkeypatch):
    """Le brief impose le fine-tuning : run() lève si le modèle de base manque."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "train" / "line_000000.png").touch()

    monkeypatch.setattr(train_kraken, "compile_split", lambda img_dir, out: None)
    # fetch_base_model échoue (pas de kraken)
    monkeypatch.setattr(train_kraken, "fetch_base_model", lambda doi, d: None)

    cfg = {
        "model": {"base_model_doi": "10.5281/zenodo.bad", "require_finetuning": True},
        "training": {"epochs": 1, "device": "cpu"},
    }
    with pytest.raises(RuntimeError, match="Fine-tuning requis"):
        train_kraken.run(cfg, data_dir)


def test_run_allows_scratch_when_explicitly_disabled(tmp_path, monkeypatch):
    """Si require_finetuning=False, l'entraînement from scratch est autorisé."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "train" / "line_000000.png").touch()

    monkeypatch.setattr(train_kraken, "compile_split", lambda img_dir, out: None)
    monkeypatch.setattr(train_kraken, "fetch_base_model", lambda doi, d: None)
    trains = []
    monkeypatch.setattr(
        train_kraken,
        "run_kraken_train",
        lambda train_a, val_a, out, hp, base, log: trains.append(base),
    )

    cfg = {
        "model": {"base_model_doi": "x", "require_finetuning": False},
        "training": {"epochs": 1, "device": "cpu"},
    }
    train_kraken.run(cfg, data_dir)
    # entraînement lancé sans modèle de base
    assert len(trains) == 1
    assert trains[0] is None
