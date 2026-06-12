"""Tests unitaires pour htmir.eval.evaluate."""

from pathlib import Path

from htmir.eval import evaluate


# ── levenshtein ─────────────────────────────────────────────────────────────


def test_levenshtein_identical():
    assert evaluate.levenshtein("abc", "abc") == 0


def test_levenshtein_empty():
    assert evaluate.levenshtein("", "abc") == 3
    assert evaluate.levenshtein("abc", "") == 3


def test_levenshtein_substitution_insertion():
    assert evaluate.levenshtein("chat", "chats") == 1     # insertion
    assert evaluate.levenshtein("chat", "chot") == 1      # substitution
    assert evaluate.levenshtein("kitten", "sitting") == 3


def test_levenshtein_token_lists():
    """Fonctionne aussi au niveau mots (listes de tokens)."""
    assert evaluate.levenshtein(["le", "roi"], ["le", "roi"]) == 0
    assert evaluate.levenshtein(["le", "roi"], ["le", "reine"]) == 1


# ── char_error_rate ─────────────────────────────────────────────────────────


def test_cer_perfect():
    assert evaluate.char_error_rate("bonjour", "bonjour") == 0.0


def test_cer_one_error():
    # 1 substitution sur 7 caractères
    assert evaluate.char_error_rate("bonjour", "bnnjour") == 1 / 7


def test_cer_empty_ref():
    assert evaluate.char_error_rate("", "") == 0.0
    assert evaluate.char_error_rate("", "x") == 1.0


# ── word_error_rate ─────────────────────────────────────────────────────────


def test_wer_perfect():
    assert evaluate.word_error_rate("le roi de France", "le roi de France") == 0.0


def test_wer_one_word_wrong():
    # 1 mot faux sur 4
    assert evaluate.word_error_rate("le roi de France", "le roi de Paris") == 1 / 4


# ── corpus_metrics ──────────────────────────────────────────────────────────


def test_corpus_metrics_micro_average():
    """Le CER corpus est somme(distances)/somme(longueurs), pas une moyenne."""
    pairs = [
        ("abcd", "abcd"),   # 0 erreur / 4 chars
        ("ef", "xf"),       # 1 erreur / 2 chars
    ]
    m = evaluate.corpus_metrics(pairs)
    assert m["n_lines"] == 2
    assert m["n_chars"] == 6
    assert m["cer"] == 1 / 6   # micro-moyenne, pas (0 + 0.5)/2 = 0.25


def test_corpus_metrics_empty():
    m = evaluate.corpus_metrics([])
    assert m["cer"] == 0.0
    assert m["wer"] == 0.0
    assert m["n_lines"] == 0


def test_corpus_metrics_perfect_corpus():
    pairs = [("li rois", "li rois"), ("de France", "de France")]
    m = evaluate.corpus_metrics(pairs)
    assert m["cer"] == 0.0
    assert m["wer"] == 0.0


# ── build_ketos_test_cmd ────────────────────────────────────────────────────


def test_build_ketos_test_cmd():
    cmd = evaluate.build_ketos_test_cmd(Path("m.mlmodel"), Path("test.arrow"), "cuda:0")
    assert cmd[:2] == ["ketos", "test"]
    assert "--model" in cmd and "m.mlmodel" in cmd
    assert "--format-type" in cmd and "binary" in cmd
    assert "--device" in cmd and "cuda:0" in cmd
    assert cmd[-1] == "test.arrow"


# ── run (subprocess mocké) ──────────────────────────────────────────────────


def test_run_writes_report(tmp_path, monkeypatch):
    """run() doit appeler ketos test et écrire un rapport JSON."""
    class FakeProc:
        stdout = "Average CER: 5.2%\n"
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *a, **k: FakeProc())

    report_path = tmp_path / "report.json"
    report = evaluate.run(Path("m.mlmodel"), Path("t.arrow"), "cpu", report_path)

    assert report_path.exists()
    import json
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "Average CER" in data["ketos_output"]
    assert data["model"] == "m.mlmodel"
