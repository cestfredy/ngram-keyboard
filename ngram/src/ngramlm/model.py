"""Bigram counting and smoothing for next-word prediction.

Design choice: the model stores *counts* only. Probabilities are computed on
demand, so a smoothing method can be swapped or tuned (on the validation set)
without recounting the corpus.

Conventions
-----------
* Every sentence is wrapped as ``<s> w1 ... wn </s>``.
* ``<s>`` is only ever a context, never a predicted word.
* The vocabulary is the set of predicted words (``</s>`` included, ``<s>`` excluded).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Sequence

BOS = "<s>"
EOS = "</s>"

# A probability function maps (word, history) -> P(word | history).
ProbFn = Callable[[str, str], float]


class BigramModel:
    def __init__(self, sentences: Iterable[Sequence[str]]) -> None:
        self.unigrams: Counter[str] = Counter()  # C(w), predicted words only
        self.bigrams: Counter[tuple[str, str]] = Counter()  # C(h, w)
        self.context: Counter[str] = Counter()  # C(h): times h is a context
        self.followers: Counter[str] = Counter()  # N1+(h .): distinct words after h
        self.preceders: Counter[str] = Counter()  # N1+(. w): distinct words before w

        for words in sentences:
            seq = [BOS, *words, EOS]
            for prev, word in zip(seq, seq[1:]):
                self.unigrams[word] += 1
                self.bigrams[(prev, word)] += 1
                self.context[prev] += 1

        for prev, word in self.bigrams:
            self.followers[prev] += 1
            self.preceders[word] += 1

        self.total = sum(self.unigrams.values())  # N
        self.vocab = sorted(self.unigrams)
        self.size = len(self.vocab)  # V
        self.bigram_types = len(self.bigrams)  # N1+(. .)

    # ------------------------------------------------------------------ #
    # Probability estimators
    # ------------------------------------------------------------------ #
    def p_unigram(self, word: str) -> float:
        """P(w) = C(w) / N."""
        return self.unigrams[word] / self.total

    def p_mle(self, word: str, history: str) -> float:
        """Maximum likelihood: C(h, w) / C(h). Zero for unseen bigrams."""
        c_h = self.context[history]
        if c_h == 0:
            return 0.0
        return self.bigrams[(history, word)] / c_h

    def p_add_k(self, word: str, history: str, k: float = 1.0) -> float:
        """Add-k smoothing (Laplace when k = 1): (C(h, w) + k) / (C(h) + k V)."""
        return (self.bigrams[(history, word)] + k) / (self.context[history] + k * self.size)

    def p_interpolated(self, word: str, history: str, lam: float = 0.7) -> float:
        """Linear interpolation: lam * P_bigram + (1 - lam) * P_unigram."""
        return lam * self.p_mle(word, history) + (1.0 - lam) * self.p_unigram(word)

    def p_continuation(self, word: str) -> float:
        """Kneser-Ney continuation probability: N1+(. w) / N1+(. .)."""
        return self.preceders[word] / self.bigram_types

    def p_kneser_ney(self, word: str, history: str, d: float = 0.75) -> float:
        """Interpolated Kneser-Ney with absolute discount d."""
        c_h = self.context[history]
        if c_h == 0:
            return self.p_continuation(word)
        discounted = max(self.bigrams[(history, word)] - d, 0.0) / c_h
        backoff_weight = d * self.followers[history] / c_h
        return discounted + backoff_weight * self.p_continuation(word)

    # ------------------------------------------------------------------ #
    # Evaluation and prediction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def sentence_probability(words: Sequence[str], prob: ProbFn) -> float:
        """P(sentence), including the end-of-sentence marker."""
        seq = [BOS, *words, EOS]
        p = 1.0
        for prev, word in zip(seq, seq[1:]):
            p *= prob(word, prev)
        return p

    @staticmethod
    def perplexity(words: Sequence[str], prob: ProbFn) -> float:
        """exp(-1/N * sum ln P(w_i | h_i)) over the N predicted words (</s> included)."""
        seq = [BOS, *words, EOS]
        log_sum = 0.0
        for prev, word in zip(seq, seq[1:]):
            p = prob(word, prev)
            if p <= 0.0:
                return math.inf
            log_sum += math.log(p)
        return math.exp(-log_sum / (len(seq) - 1))

    def top_k(self, history: str, k: int, prob: ProbFn) -> list[str]:
        """The k most probable next words (ties broken alphabetically).

        ``</s>`` is never suggested and zero-probability words are dropped.
        """
        scored = [(prob(w, history), w) for w in self.vocab if w != EOS]
        scored = [(p, w) for p, w in scored if p > 0.0]
        scored.sort(key=lambda pw: (-pw[0], pw[1]))
        return [w for _, w in scored[:k]]
