import pytest

from ngramlm.data import map_to_vocab
from ngramlm.evaluate import evaluate_topk, unigram_suggester
from ngramlm.model import BOS


def test_map_to_vocab():
    assert map_to_vocab([["a", "z"], ["b"]], {"a", "b"}) == [["a", "<unk>"], ["b"]]


def test_topk_with_constant_suggestions():
    sentences = [["a", "b"], ["a", "c"]]  # targets: a, b, a, c
    res = evaluate_topk(sentences, lambda history: ["a", "b"], ks=(1, 2))
    assert res["positions"] == 4
    assert res["top1_pct"] == pytest.approx(50.0)  # a, a
    assert res["top2_pct"] == pytest.approx(75.0)  # a, b, a


def test_unk_target_is_a_miss_but_excluded_from_in_vocab_figures():
    sentences = [["a", "<unk>"], ["b", "a"]]
    res = evaluate_topk(sentences, lambda history: ["a", "b"], ks=(1, 2))
    assert res["positions"] == 4 and res["in_vocab_positions"] == 3
    assert res["top2_pct"] == pytest.approx(75.0)  # 3 hits out of 4
    assert res["top2_in_vocab_pct"] == pytest.approx(100.0)  # 3 hits out of 3
    assert res["top1_pct"] == pytest.approx(50.0)  # a at pos 1, a at pos 4


def test_history_starts_with_bos_and_grows_with_the_true_words():
    seen = []

    def suggest(history):
        seen.append(list(history))
        return []

    evaluate_topk([["x", "y", "z"]], suggest, ks=(1,))
    assert seen == [[BOS], [BOS, "x"], [BOS, "x", "y"]]


def test_unigram_suggester_ignores_unk_and_out_of_vocab():
    counts = {"de": 10, "je": 8, "<unk>": 50, "rare": 99, "est": 8, "la": 3}
    vocab = {"de", "je", "est", "la", "<unk>"}  # "rare" is not in the vocabulary
    suggest = unigram_suggester(counts, vocab, k=3)
    assert suggest([BOS]) == ["de", "est", "je"]  # tie est/je broken alphabetically
    assert suggest([BOS, "anything"]) == ["de", "est", "je"]
