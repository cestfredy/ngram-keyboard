import pytest

from ngramlm.data import count_tokens, deduplicate, oov_stats, split, top_vocabulary


def make(n: int):
    return [[f"w{i}", "x"] for i in range(n)]


def test_split_sizes_and_disjointness():
    train, valid, test = split(make(1000), seed=42)
    assert (len(train), len(valid), len(test)) == (900, 50, 50)
    keys = [tuple(s) for part in (train, valid, test) for s in part]
    assert len(set(keys)) == 1000  # nothing lost, nothing shared between splits


def test_split_is_reproducible_and_seed_dependent():
    a = split(make(500), seed=42)
    b = split(make(500), seed=42)
    c = split(make(500), seed=7)
    assert a == b
    assert a != c


def test_split_rejects_ratios_without_test():
    with pytest.raises(ValueError):
        split(make(10), seed=1, train=0.9, valid=0.1)


def test_deduplicate_keeps_first_occurrence():
    sents = [["a", "b"], ["c", "d"], ["a", "b"]]
    assert deduplicate(sents) == [["a", "b"], ["c", "d"]]


def test_top_vocabulary_breaks_ties_alphabetically():
    counts = count_tokens([["b", "a", "c", "a"], ["b", "d"]])
    # a:2, b:2, c:1, d:1 -> top 3 = a, b, c (c before d)
    assert top_vocabulary(counts, 3) == {"a", "b", "c"}
    assert top_vocabulary(counts, 10) == {"a", "b", "c", "d"}


def test_oov_stats():
    stats = oov_stats([["a", "b"], ["a", "z"], ["y", "z"]], {"a", "b"})
    assert stats["tokens"] == 6 and stats["oov_tokens"] == 3
    assert stats["oov_rate_pct"] == pytest.approx(50.0)
    assert stats["sentences_with_oov_pct"] == pytest.approx(200 / 3)
