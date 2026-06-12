"""Tests unitaires pour htmir.training.train_kraken."""

from pathlib import Path

import pytest

from htmir.training import train_kraken


# ── build_compile_cmd ───────────────────────────────────────────────────────


def test_build_compile_cmd_basic(tmp_path):
    """La commande compile doit cibler le format 'path' et lister les PNG."""
    (tmp_path / "line_000000.png").touch()
    (tmp_path / "line_000001.png").touch()
    out = tmp_path / "train.arrow"

    cmd = train_kraken.build_compile_cmd(tmp_path, out)

    assert cmd[:2] == ["ketos", "compile"]
    assert "--format-type" in cmd and "path" in cmd
    assert "--output" in cmd
    assert str(out) in cmd
    # Les deux images doivent être présentes et triées
    pngs = [c for c in cmd if c.endswith(".png")]
    assert len(pngs) == 2
    assert pngs == sorted(pngs)


# ── build_train_cmd ─────────────────────────────────────────────────────────


def test_build_train_cmd_finetune_includes_base_model():
    """En fine-tuning, --load et --resize union doivent être présents."""
    hp = {"epochs": 30, "lr": 1e-4, "batch_size": 16, "device": "cuda:0",
          "early_stopping_patience": 5, "workers": 2}
    cmd = train_kraken.build_train_cmd(
        Path("train.arrow"), Path("val.arrow"), "out", hp,
        base_model="base.mlmodel",
    )
    assert "--load" in cmd
    assert "base.mlmodel" in cmd
    assert "--resize" in cmd and "union" in cmd
    assert "--evaluation-files" in cmd and "val.arrow" in cmd
    assert cmd[-1] == "train.arrow"   # train en positionnel final


def test_build_train_cmd_from_scratch_no_load():
    """Sans modèle de base, pas de --load ni --resize."""
    cmd = train_kraken.build_train_cmd(
        Path("train.arrow"), None, "out", {}, base_model=None,
    )
    assert "--load" not in cmd
    assert "--resize" not in cmd
    assert "--evaluation-files" not in cmd   # pas de val


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
    """Si `kraken` n'est pas installé, fetch_base_model retourne None."""
    def boom(*a, **k):
        raise FileNotFoundError("kraken introuvable")
    monkeypatch.setattr(train_kraken.subprocess, "run", boom)
    assert train_kraken.fetch_base_model("10.5281/zenodo.1", tmp_path) is None


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
    """run() compile train+val (subprocess.run) puis entraîne (run_logged_command)."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "validation").mkdir(parents=True)
    (data_dir / "train" / "line_000000.png").touch()
    (data_dir / "validation" / "line_000000.png").touch()

    compiles = []
    trains = []
    monkeypatch.setattr(train_kraken.subprocess, "run",
                        lambda cmd, **k: compiles.append(cmd))
    monkeypatch.setattr(train_kraken, "run_logged_command",
                        lambda cmd, log: trains.append((cmd, log)))

    cfg = {
        "model": {"base_model_path": "base.mlmodel", "output_name": "htmir-french-13c"},
        "training": {"epochs": 1, "device": "cpu"},
    }
    train_kraken.run(cfg, data_dir)

    # 2 compile (train + val)
    assert len(compiles) == 2
    assert all(c[:2] == ["ketos", "compile"] for c in compiles)
    # 1 entraînement loggé, avec --load (fine-tuning) et log dans data_dir/train.log
    assert len(trains) == 1
    train_cmd, log_path = trains[0]
    assert train_cmd[:2] == ["ketos", "train"]
    assert "--load" in train_cmd
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

    monkeypatch.setattr(train_kraken.subprocess, "run", lambda cmd, **k: None)
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

    monkeypatch.setattr(train_kraken.subprocess, "run", lambda cmd, **k: None)
    monkeypatch.setattr(train_kraken, "fetch_base_model", lambda doi, d: None)
    trains = []
    monkeypatch.setattr(train_kraken, "run_logged_command",
                        lambda cmd, log: trains.append(cmd))

    cfg = {
        "model": {"base_model_doi": "x", "require_finetuning": False},
        "training": {"epochs": 1, "device": "cpu"},
    }
    train_kraken.run(cfg, data_dir)
    # entraînement lancé sans --load
    assert len(trains) == 1
    assert "--load" not in trains[0]
