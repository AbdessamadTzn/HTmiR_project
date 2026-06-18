"""Tests unitaires pour htmir.viz.data_loaders."""

import json

from htmir.viz import data_loaders as dl


# ── load_dataset_manifest ───────────────────────────────────────────────────


def test_load_manifest_missing(tmp_path):
    assert dl.load_dataset_manifest(tmp_path / "nope.json") == {}


def test_load_manifest_ok(tmp_path):
    p = tmp_path / "dataset_manifest.json"
    p.write_text(json.dumps({"total": 42, "splits": {"train": 42}}), encoding="utf-8")
    assert dl.load_dataset_manifest(p)["total"] == 42


# ── compute_length_stats ────────────────────────────────────────────────────


def test_compute_length_stats(tmp_path):
    (tmp_path / "line_000000.gt.txt").write_text("li rois", encoding="utf-8")        # 7c, 2m
    (tmp_path / "line_000001.gt.txt").write_text("de France ici", encoding="utf-8")  # 13c, 3m
    stats = dl.compute_length_stats(tmp_path)
    assert stats["n_lines"] == 2
    assert sorted(stats["char_lengths"]) == [7, 13]
    assert sorted(stats["word_counts"]) == [2, 3]


def test_compute_length_stats_empty(tmp_path):
    stats = dl.compute_length_stats(tmp_path)
    assert stats["n_lines"] == 0
    assert stats["char_lengths"] == []


# ── sample_lines ────────────────────────────────────────────────────────────


def test_sample_lines_pairs_image_and_text(tmp_path):
    (tmp_path / "line_000000.png").touch()
    (tmp_path / "line_000000.gt.txt").write_text("li rois", encoding="utf-8")
    samples = dl.sample_lines(tmp_path, n=8)
    assert samples[0]["text"] == "li rois"
    assert samples[0]["image_path"].endswith("line_000000.png")


def test_sample_lines_respects_n(tmp_path):
    for i in range(5):
        (tmp_path / f"line_{i:06d}.png").touch()
        (tmp_path / f"line_{i:06d}.gt.txt").write_text("x", encoding="utf-8")
    assert len(dl.sample_lines(tmp_path, n=3)) == 3


# ── parse_ketos_log ─────────────────────────────────────────────────────────


def test_parse_ketos_log_extracts_epochs():
    log = """
    Accuracy report (0) 0.7500 1000 250
    Accuracy report (1) 0.8200 1000 180
    Accuracy report (2) 0.9100 1000 90
    """
    rows = dl.parse_ketos_log(log)
    assert len(rows) == 3
    assert rows[0] == {"epoch": 0, "accuracy": 0.75, "cer": 0.25}
    assert rows[2]["cer"] == round(1 - 0.91, 4)


def test_parse_ketos_log_sorts_by_epoch():
    log = "Accuracy report (2) 0.9 0 0\nAccuracy report (0) 0.7 0 0"
    assert [r["epoch"] for r in dl.parse_ketos_log(log)] == [0, 2]


def test_parse_ketos_log_empty():
    assert dl.parse_ketos_log("rien ici") == []


# ── load_training_metrics ───────────────────────────────────────────────────


def test_load_training_metrics_csv(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text("epoch,accuracy,cer\n0,0.75,0.25\n1,0.91,0.09\n", encoding="utf-8")
    rows = dl.load_training_metrics(p)
    assert rows[0]["epoch"] == 0
    assert rows[1]["cer"] == 0.09


def test_load_training_metrics_missing(tmp_path):
    assert dl.load_training_metrics(tmp_path / "nope.csv") == []


# ── load_eval_report ────────────────────────────────────────────────────────


def test_load_eval_report(tmp_path):
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps({"model": "m.mlmodel"}), encoding="utf-8")
    assert dl.load_eval_report(p)["model"] == "m.mlmodel"


def test_load_eval_report_missing(tmp_path):
    assert dl.load_eval_report(tmp_path / "nope.json") == {}


# ── compute_early_stopping ──────────────────────────────────────────────────


def test_early_stopping_finds_best_and_stop():
    """Best epoch = CER min ; stop = patience epochs après le best."""
    metrics = [
        {"epoch": 0, "cer": 0.40},
        {"epoch": 1, "cer": 0.20},
        {"epoch": 2, "cer": 0.06},   # ← best
        {"epoch": 3, "cer": 0.07},   # stall 1
        {"epoch": 4, "cer": 0.065},  # stall 2
        {"epoch": 5, "cer": 0.066},  # stall 3 → stop (patience=3)
    ]
    es = dl.compute_early_stopping(metrics, patience=3)
    assert es["best_epoch"] == 2
    assert es["best_cer"] == 0.06
    assert es["stop_epoch"] == 5
    assert es["stalled_epochs"] == [3, 4, 5]


def test_early_stopping_no_stop_if_improving():
    """Si le CER s'améliore tout du long, pas d'arrêt déclenché."""
    metrics = [{"epoch": i, "cer": 0.5 - i * 0.1} for i in range(5)]
    es = dl.compute_early_stopping(metrics, patience=2)
    assert es["best_epoch"] == 4
    assert es["stop_epoch"] is None
    assert es["stalled_epochs"] == []


def test_early_stopping_empty():
    es = dl.compute_early_stopping([], patience=10)
    assert es["best_epoch"] is None
    assert es["stalled_epochs"] == []


# ── char_frequency ──────────────────────────────────────────────────────────


def test_char_frequency_counts_all(tmp_path):
    (tmp_path / "a.gt.txt").write_text("aab", encoding="utf-8")
    (tmp_path / "b.gt.txt").write_text("a", encoding="utf-8")
    freq = dl.char_frequency(tmp_path)
    assert freq["a"] == 3
    assert freq["b"] == 1
    # trié par fréquence décroissante
    assert list(freq.keys())[0] == "a"


def test_char_frequency_only_special(tmp_path):
    """only_special ne garde que les caractères médiévaux non-ASCII."""
    (tmp_path / "a.gt.txt").write_text("li rois ⁊ ẽ ⁊", encoding="utf-8")
    freq = dl.char_frequency(tmp_path, only_special=True)
    assert freq["⁊"] == 2
    assert "ẽ" in freq
    assert "r" not in freq   # ASCII exclu


# ── worst_predictions ───────────────────────────────────────────────────────


def test_worst_predictions_sorts_by_cer():
    preds = [
        {"ref": "bonjour", "hyp": "bonjour"},    # CER 0
        {"ref": "bonjour", "hyp": "bXnjXur"},    # CER 2/7
        {"ref": "abc", "hyp": "xyz"},            # CER 1.0
    ]
    worst = dl.worst_predictions(preds, n=2)
    assert len(worst) == 2
    assert worst[0]["cer"] == 1.0          # le pire en premier
    assert worst[0]["ref"] == "abc"


def test_load_predictions_missing(tmp_path):
    assert dl.load_predictions(tmp_path / "nope.csv") == []


def test_load_predictions_csv(tmp_path):
    p = tmp_path / "preds.csv"
    p.write_text("ref,hyp\nli rois,li reis\n", encoding="utf-8")
    rows = dl.load_predictions(p)
    assert rows[0]["ref"] == "li rois"
    assert rows[0]["hyp"] == "li reis"


# ── text_from_alto ──────────────────────────────────────────────────────────


def test_text_from_alto_basic():
    """Reconstitue le texte ligne par ligne depuis un ALTO Kraken."""
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = f"""<?xml version="1.0"?>
<alto xmlns="{ns}">
  <Layout><Page><PrintSpace>
    <TextLine><String CONTENT="li"/><String CONTENT="rois"/></TextLine>
    <TextLine><String CONTENT="de"/><String CONTENT="France"/></TextLine>
  </PrintSpace></Page></Layout>
</alto>"""
    assert dl.text_from_alto(alto) == "li rois\nde France"


def test_text_from_alto_invalid():
    assert dl.text_from_alto("pas du xml") == ""


def test_text_from_alto_empty_lines():
    ns = "http://www.loc.gov/standards/alto/ns-v4#"
    alto = f'<alto xmlns="{ns}"><Layout><Page><PrintSpace>' \
           '<TextLine><String CONTENT=""/></TextLine>' \
           '</PrintSpace></Page></Layout></alto>'
    assert dl.text_from_alto(alto) == ""
