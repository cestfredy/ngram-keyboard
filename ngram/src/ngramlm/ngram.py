"""N-gram language model of any order with interpolated Kneser-Ney smoothing.

Only counts are stored; probabilities are computed on demand.

Counts used at each order k (history h of length k-1, word w):
* highest order n : raw counts c(h, w)
* lower orders    : continuation counts N1+(. h w) = number of distinct words seen before (h, w),
                    except for n-grams that start with <s>, which keep raw counts (they can never
                    be extended to the left).

    P_k(w | h) = max(c'(h, w) - d, 0) / D(h)  +  d * N1+(h .) / D(h)  *  P_{k-1}(w | h')

with D(h) = sum over w of c'(h, w) and h' the history without its oldest word. If the history was
never seen, P_k(w | h) = P_{k-1}(w | h'). The recursion stops at the unigram level
P_1(w) = c'(w) / sum c'.

Sentences are padded with n-1 `<s>` markers and one `</s>`. `<s>` is never predicted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from .data import UNK
from .model import BOS, EOS


class NgramModel:
    def __init__(self, order: int, discount: float = 0.75) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if not 0.0 < discount < 1.0:
            raise ValueError("discount must be in (0, 1)")
        self.order = order
        self.discount = discount

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, sentences: Iterable[Sequence[str]]) -> "NgramModel":
        sentences = [list(s) for s in sentences]
        n = self.order

        # Token ids follow alphabetical order, so ties broken by id are ties broken alphabetically.
        tokens = sorted({BOS, EOS, *(t for s in sentences for t in s)})
        self.tokens = tokens
        self.ids = {t: i for i, t in enumerate(tokens)}
        bos, eos = self.ids[BOS], self.ids[EOS]
        self._bos = bos
        self._excluded = {bos, eos}
        if UNK in self.ids:
            self._excluded.add(self.ids[UNK])

        # highest order: raw counts of n-grams whose last token is a predicted word
        raw: dict[tuple[int, ...], int] = defaultdict(int)
        pad = [bos] * (n - 1)
        for s in sentences:
            seq = pad + [self.ids[t] for t in s] + [eos]
            for i in range(n - 1, len(seq)):
                raw[tuple(seq[i - n + 1 : i + 1])] += 1

        num: list[dict[tuple[int, ...], int]] = [{} for _ in range(n + 1)]
        num[n] = dict(raw)
        for k in range(n - 1, 0, -1):
            lower: dict[tuple[int, ...], int] = defaultdict(int)
            for gram, c in num[k + 1].items():
                suffix = gram[1:]
                lower[suffix] += c if suffix[0] == bos else 1
            num[k] = dict(lower)
        self.num = num

        self._build_index()
        return self

    def _build_index(self) -> None:
        """Derive, from the counts in ``self.num``, everything needed to compute probabilities."""
        num, n = self.num, self.order
        # per history: total mass D(h), number of distinct followers N1+(h .), followers list
        self.den: list[dict[tuple[int, ...], int]] = [{} for _ in range(n + 1)]
        self.fol: list[dict[tuple[int, ...], int]] = [{} for _ in range(n + 1)]
        self.succ: list[dict[tuple[int, ...], list[int]]] = [{} for _ in range(n + 1)]
        for k in range(1, n + 1):
            den: dict[tuple[int, ...], int] = defaultdict(int)
            fol: dict[tuple[int, ...], int] = defaultdict(int)
            succ: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for gram, c in num[k].items():
                h = gram[:-1]
                den[h] += c
                if k > 1:
                    fol[h] += 1
                    succ[h].append(gram[-1])
            self.den[k], self.fol[k], self.succ[k] = dict(den), dict(fol), dict(succ)

        self._unigram_total = self.den[1][()]
        # words ranked by unigram-level probability, used to complete the candidate set
        ranked = sorted(
            ((c, w) for (w,), c in num[1].items() if w not in self._excluded),
            key=lambda cw: (-cw[0], cw[1]),
        )
        self._rank1 = [w for _, w in ranked]

    def prune(self, min_counts: Sequence[int]) -> "NgramModel":
        """Drop n-grams whose count is below a threshold, one threshold per order 2..n.

        ``min_counts[i]`` applies to order ``i + 2`` (1 keeps everything). Unigrams are never
        removed. Counts are those used by the smoothing (raw at the top order, continuation
        counts below), the discounts are applied to what remains, so every distribution still
        sums to 1; a history whose entries were all removed falls back to the lower order.
        """
        if len(min_counts) != self.order - 1:
            raise ValueError(f"expected {self.order - 1} thresholds, got {len(min_counts)}")
        if any(m < 1 for m in min_counts):
            raise ValueError("thresholds must be at least 1")
        for k, m in zip(range(2, self.order + 1), min_counts):
            if m > 1:
                self.num[k] = {g: c for g, c in self.num[k].items() if c >= m}
        self._build_index()
        return self

    # ------------------------------------------------------------------ #
    # Probabilities
    # ------------------------------------------------------------------ #
    @property
    def words(self) -> list[str]:
        """Every word the model can predict (``<s>`` excluded, ``</s>`` included)."""
        return [t for t in self.tokens if t != BOS]

    def ngram_counts(self) -> dict[int, int]:
        return {k: len(self.num[k]) for k in range(1, self.order + 1)}

    def _history_ids(self, history: Sequence[str]) -> tuple[int, ...]:
        """Last n-1 words as ids, padded on the left with <s>; unknown words get id -1."""
        want = self.order - 1
        if want == 0:
            return ()
        ids = [self.ids.get(t, -1) for t in history[-want:]]
        return tuple([self._bos] * (want - len(ids)) + ids)

    def _prob(self, w: int, hist: tuple[int, ...]) -> float:
        d = self.discount
        p = self.num[1].get((w,), 0) / self._unigram_total
        for k in range(2, self.order + 1):
            h = hist[len(hist) - (k - 1) :]
            total = self.den[k].get(h)
            if not total:
                continue  # unseen history: keep the lower-order estimate
            c = self.num[k].get(h + (w,), 0)
            p = max(c - d, 0.0) / total + d * self.fol[k][h] / total * p
        return p

    def prob(self, word: str, history: Sequence[str]) -> float:
        """P(word | history). The history may start with ``<s>`` and may be shorter than n-1."""
        return self._prob(self.ids.get(word, -1), self._history_ids(history))

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """exp(-1/N sum ln P) over every predicted word, ``</s>`` included."""
        log_sum = 0.0
        count = 0
        pad = [BOS] * (self.order - 1)
        for s in sentences:
            seq = pad + list(s) + [EOS]
            for i in range(self.order - 1, len(seq)):
                p = self.prob(seq[i], seq[max(0, i - self.order + 1) : i])
                if p <= 0.0:
                    return math.inf
                log_sum += math.log(p)
                count += 1
        return math.exp(-log_sum / count)

    # ------------------------------------------------------------------ #
    # Suggestions
    # ------------------------------------------------------------------ #
    def suggest(self, history: Sequence[str], k: int = 3) -> list[str]:
        """The k most probable next words (never ``<s>``, ``</s>`` or ``<unk>``).

        Exact top-k without scoring the whole vocabulary: a word that was not seen after the
        history at any order only receives back-off mass proportional to its unigram-level
        probability, so among those words the best k are the first k of a precomputed ranking.
        """
        hist = self._history_ids(history)
        candidates: set[int] = set()
        for level in range(2, self.order + 1):
            candidates.update(self.succ[level].get(hist[len(hist) - (level - 1) :], ()))
        candidates -= self._excluded

        extra = 0
        for w in self._rank1:
            if w in candidates:
                continue
            candidates.add(w)
            extra += 1
            if extra >= k:
                break

        scored = [(-self._prob(w, hist), w) for w in candidates]
        scored.sort()
        return [self.tokens[w] for p, w in scored[:k] if p < 0.0]
