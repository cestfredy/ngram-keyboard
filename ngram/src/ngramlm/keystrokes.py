"""Keystroke savings: how many key presses a prediction bar saves while typing real sentences.

Simulated user, for each word of a sentence:
  * types its letters one by one; before each letter (and before the first one) the keyboard
    shows the 3 most probable words that start with what has been typed so far;
  * as soon as the word is among them, taps it: 1 key press, which inserts the word and the
    space after it;
  * otherwise types the whole word and presses space.
Cost without prediction: len(word) + 1 (space). Elision tokens ending with an apostrophe
(`l'`) have no space. A tap is never counted when it costs more than typing.

    keystroke savings = 1 - (key presses with prediction) / (key presses without)

The history given to the model is the sentence so far, with out-of-vocabulary words replaced by
`<unk>`; an out-of-vocabulary target can never be suggested and is typed in full (its real
length is used). Digit sequences (`<num>`) are skipped: their length is unknown after cleaning.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .model import BOS

Suggester = Callable[[Sequence[str], str], list[str]]  # (history, prefix) -> suggestions

NUM = "<num>"
UNK = "<unk>"


def keystroke_savings(
    sentences: Iterable[Sequence[str]], suggest: Suggester, vocab: set[str]
) -> dict[str, float]:
    base_keys = keys = words = hits = hits_at_zero = letters_before_hit = skipped = 0
    for sentence in sentences:
        history = [BOS]
        for word in sentence:
            if word == NUM:
                skipped += 1
                history.append(NUM)
                continue
            baseline = len(word) + (0 if word.endswith("'") else 1)
            cost = baseline
            if word in vocab:
                for j in range(len(word)):
                    if word in suggest(history, word[:j]):
                        cost = min(j + 1, baseline)
                        hits += 1
                        hits_at_zero += j == 0
                        letters_before_hit += j
                        break
            words += 1
            base_keys += baseline
            keys += cost
            history.append(word if word in vocab else UNK)
    return {
        "words": words,
        "skipped_num": skipped,
        "keys_without": base_keys,
        "keys_with": keys,
        "savings_pct": 100 * (1 - keys / base_keys),
        "hit_without_typing_pct": 100 * hits_at_zero / words,
        "hit_before_end_pct": 100 * hits / words,
        "avg_letters_before_hit": letters_before_hit / hits if hits else 0.0,
    }
