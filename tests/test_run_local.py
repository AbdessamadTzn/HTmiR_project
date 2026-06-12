"""Tests unitaires pour htmir.cli.run_local."""

from pathlib import Path

import pytest

from htmir.cli import run_local


@pytest.fixture
def cfg(tmp_path):
    return {
        "dataset": {"splits": ["train", "validation", "test"]},
        "output": {"bucket": "htmir-data", "region": "eu-west-3",
                   "s3_prefix": "datasets/x", "local_dir": str(tmp_path / "data")},
        "model": {"output_name": "htmir-french-13c"},
        "training": {"device": "cuda:0"},
    }


# ── find_best_model ─────────────────────────────────────────────────────────


def test_find_best_model_prefers_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m_best.mlmodel").touch()
    (tmp_path / "m_5.mlmodel").touch()
    assert run_local.find_best_model(tmp_path / "m").name == "m_best.mlmodel"


def test_find_best_model_falls_back_to_last(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m_1.mlmodel").touch()
    (tmp_path / "m_9.mlmodel").touch()
    assert run_local.find_best_model(tmp_path / "m").name == "m_9.mlmodel"


def test_find_best_model_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_local.find_best_model(tmp_path / "m") is None


# ── ensure_dataset ──────────────────────────────────────────────────────────


def test_ensure_dataset_uses_local_if_present(cfg, monkeypatch):
    """Si le train existe déjà, prepare n'est pas appelé."""
    data_dir = Path(cfg["output"]["local_dir"])
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "train" / "line_000000.png").touch()

    called = {"prepare": False}
    import htmir.data.prepare_catmus as prep
    monkeypatch.setattr(prep, "prepare",
                        lambda *a, **k: called.__setitem__("prepare", True))

    run_local.ensure_dataset(cfg, data_dir, use_s3=False)
    assert called["prepare"] is False   # train présent → pas de prepare


def test_ensure_dataset_prepares_when_missing(cfg, monkeypatch):
    """Si le train est absent, prepare est appelé (avec push_s3=False en no-s3)."""
    data_dir = Path(cfg["output"]["local_dir"])
    captured = {}

    import htmir.data.prepare_catmus as prep
    monkeypatch.setattr(prep, "prepare",
                        lambda c, overrides: captured.update(overrides))

    run_local.ensure_dataset(cfg, data_dir, use_s3=False)
    assert captured["push_s3"] is False
    assert captured["output_dir"] == str(data_dir)


# ── run_pipeline ────────────────────────────────────────────────────────────


def test_run_pipeline_trains_and_evaluates(cfg, monkeypatch):
    """run_pipeline appelle train puis eval avec le bon modèle/device."""
    data_dir = Path(cfg["output"]["local_dir"])
    (data_dir).mkdir(parents=True)
    (data_dir / "test.arrow").touch()

    calls = {}

    import htmir.training.train_kraken as tk
    import htmir.eval.evaluate as ev
    monkeypatch.setattr(tk, "run", lambda c, d: Path("htmir-french-13c"))
    monkeypatch.setattr(run_local, "find_best_model",
                        lambda prefix: data_dir / "model_best.mlmodel")
    (data_dir / "model_best.mlmodel").touch()

    def fake_eval(model, test_arrow, device, report):
        calls["device"] = device
        calls["model"] = str(model)
        return {}
    monkeypatch.setattr(ev, "run", fake_eval)

    result = run_local.run_pipeline(cfg, data_dir, use_s3=False,
                                    skip_prepare=True, upload_model=False)

    assert calls["device"] == "cuda:0"
    assert calls["model"].endswith("model_best.mlmodel")
    assert result["model"].endswith("model_best.mlmodel")


def test_run_pipeline_skip_prepare_does_not_fetch(cfg, monkeypatch):
    """Avec skip_prepare, ensure_dataset n'est pas appelé."""
    data_dir = Path(cfg["output"]["local_dir"])
    data_dir.mkdir(parents=True)

    import htmir.training.train_kraken as tk
    monkeypatch.setattr(tk, "run", lambda c, d: Path("m"))
    monkeypatch.setattr(run_local, "find_best_model", lambda p: None)
    monkeypatch.setattr(run_local, "ensure_dataset",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ne doit pas être appelé")))

    run_local.run_pipeline(cfg, data_dir, use_s3=False,
                           skip_prepare=True, upload_model=False)
