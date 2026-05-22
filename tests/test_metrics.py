"""Tests des métriques CER, WER, bootstrap et McNemar."""

import pytest
import numpy as np
from htmir.evaluation.metrics import compute_cer, compute_wer, corpus_cer, corpus_wer
from htmir.evaluation.bootstrap import bootstrap_cer
from htmir.evaluation.mcnemar import mcnemar_test


class TestCER:
    def test_identical_strings(self):
        assert compute_cer("abc", "abc") == 0.0

    def test_completely_different(self):
        assert compute_cer("xyz", "abc") == 1.0

    def test_partial_error(self):
        cer = compute_cer("le chat", "le chien")
        assert 0.0 < cer <= 1.0

    def test_empty_hypothesis(self):
        cer = compute_cer("", "abc")
        assert cer == 3 / 3

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            compute_cer("abc", "")


class TestWER:
    def test_identical(self):
        assert compute_wer("le chat noir", "le chat noir") == 0.0

    def test_one_word_error(self):
        wer = compute_wer("le chien noir", "le chat noir")
        assert abs(wer - 1/3) < 1e-6

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            compute_wer("test", "")


class TestCorpusMetrics:
    def test_corpus_cer_perfect(self):
        assert corpus_cer(["abc", "def"], ["abc", "def"]) == 0.0

    def test_corpus_cer_length_mismatch(self):
        with pytest.raises(ValueError):
            corpus_cer(["abc"], ["abc", "def"])

    def test_corpus_wer_perfect(self):
        assert corpus_wer(["le chat", "bon jour"], ["le chat", "bon jour"]) == 0.0


class TestBootstrap:
    def test_bootstrap_returns_dict(self):
        hyps = ["le chat", "bon jour", "manuscrit ancien"]
        refs = ["le chien", "bonjour", "manuscrit ancien"]
        result = bootstrap_cer(hyps, refs, n_iterations=100)
        assert all(k in result for k in ["mean", "lower", "upper", "std"])

    def test_bootstrap_interval_ordered(self):
        hyps = ["abc", "def", "ghi"]
        refs = ["abc", "xyz", "ghi"]
        result = bootstrap_cer(hyps, refs, n_iterations=200)
        assert result["lower"] <= result["mean"] <= result["upper"]

    def test_bootstrap_perfect_transcription(self):
        hyps = refs = ["le chat noir"] * 20
        result = bootstrap_cer(hyps, refs, n_iterations=100)
        assert result["mean"] == pytest.approx(0.0, abs=1e-6)


class TestMcNemar:
    def test_identical_errors_not_significant(self):
        errors = [True, False, True, False] * 10
        result = mcnemar_test(errors, errors)
        assert not result["significant"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True])

    def test_result_keys(self):
        a = [True, False, True, False, False] * 5
        b = [False, False, True, True, False] * 5
        result = mcnemar_test(a, b)
        assert all(k in result for k in ["statistic", "p_value", "significant", "n01", "n10"])

    def test_p_value_range(self):
        a = [True, False] * 20
        b = [False, True] * 20
        result = mcnemar_test(a, b)
        assert 0.0 <= result["p_value"] <= 1.0
