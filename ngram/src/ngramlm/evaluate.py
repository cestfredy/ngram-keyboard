"""Next-word prediction metrics: top-k accuracy with no letter typed yet.

At every word position of a sentence the model sees the words typed so far (starting with
``<s>``) and proposes a ranked list of suggestions. Top-k is a hit when the true word is among
the first k suggestions.

* The end-of-sentence marker is not evaluated (a keyboard has nothing to suggest after it).
* A true word outside the vocabulary (``<unk>``) can never be proposed, so it counts as a miss
  in the overall figures. The ``in_vocab`` figures leave those positions out.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .data import UNK
from .model import BOS

# A suggester maps the history (words typed so far, starting with <s>) to ranked suggestions.
Suggester = Callable[[Sequence[str]], Sequence[str]]


def evaluate_topk(
    sentences: Iterable[Sequence[str]],
    suggest: Suggester,
    ks: Sequence[int] = (1, 3),
    unk: str = UNK,
) -> dict[str, float]:
    k_max = max(ks)
    hits = {k: 0 for k in ks}
    hits_in_vocab = {k: 0 for k in ks}
    positions = 0
    in_vocab_positions = 0

    for sentence in sentences:
        history = [BOS]
        for target in sentence:
            suggestions = list(suggest(history))[:k_max]
            positions += 1
            known = target != unk
            in_vocab_positions += known
            if known:
                for k in ks:
                    if target in suggestions[:k]:
                        hits[k] += 1
                        hits_in_vocab[k] += 1
            history.append(target)

    result: dict[str, float] = {"positions": positions, "in_vocab_positions": in_vocab_positions}
    for k in ks:
        result[f"top{k}_pct"] = 100 * hits[k] / positions if positions else 0.0
        result[f"top{k}_in_vocab_pct"] = (
            100 * hits_in_vocab[k] / in_vocab_positions if in_vocab_positions else 0.0
        )
    return result


def unigram_suggester(counts: dict[str, int], vocab: set[str], k: int = 3) -> Suggester:
    """Baseline: always propose the k most frequent words of the vocabulary, whatever was typed."""
    ranked = sorted(((t, c) for t, c in counts.items() if t in vocab and t != UNK), key=lambda tc: (-tc[1], tc[0]))
    fixed = [t for t, _ in ranked[:k]]
    return lambda history: fixed
