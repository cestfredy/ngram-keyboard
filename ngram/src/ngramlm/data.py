"""Dataset helpers: deduplication, reproducible split, vocabulary and out-of-vocabulary rate.

Order matters: deduplicate, then split, then build the vocabulary on the TRAIN split only.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Sequence

Sentence = Sequence[str]


def deduplicate(sentences: Iterable[Sentence]) -> list[list[str]]:
    """Drop sentences whose token sequence was already seen (keeps the first occurrence).

    Done on tokens, not raw text, so "Tom ?" and "Tom!" cannot end up in different splits.
    """
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for s in sentences:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            out.append(list(s))
    return out


def split(
    sentences: Sequence[Sentence], seed: int, train: float = 0.90, valid: float = 0.05
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Shuffle with a fixed seed, then cut into train / valid / test (test is the remainder)."""
    if not 0 < train < 1 or not 0 <= valid < 1 or train + valid >= 1:
        raise ValueError("ratios must leave a non-empty test split")
    shuffled = [list(s) for s in sentences]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_valid = int(n * valid)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_valid],
        shuffled[n_train + n_valid :],
    )


UNK = "<unk>"


def map_to_vocab(sentences: Iterable[Sentence], vocab: set[str], unk: str = UNK) -> list[list[str]]:
    """Replace every token outside the vocabulary by `<unk>`."""
    return [[t if t in vocab else unk for t in s] for s in sentences]


def count_tokens(sentences: Iterable[Sentence]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for s in sentences:
        counts.update(s)
    return counts


def top_vocabulary(counts: Counter[str], size: int) -> set[str]:
    """The `size` most frequent tokens (ties broken alphabetically, so it is deterministic)."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {tok for tok, _ in ranked[:size]}


def oov_stats(sentences: Iterable[Sentence], vocab: set[str]) -> dict[str, float]:
    """Out-of-vocabulary statistics of a split for a given vocabulary."""
    total = oov = n_sent = n_sent_oov = 0
    for s in sentences:
        n_sent += 1
        missing = sum(1 for tok in s if tok not in vocab)
        total += len(s)
        oov += missing
        n_sent_oov += missing > 0
    return {
        "tokens": total,
        "oov_tokens": oov,
        "oov_rate_pct": 100 * oov / total if total else 0.0,
        "sentences": n_sent,
        "sentences_with_oov_pct": 100 * n_sent_oov / n_sent if n_sent else 0.0,
    }
