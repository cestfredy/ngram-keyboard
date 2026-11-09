import random

import pytest

from ngramlm import BOS
from ngramlm.compact import CompactNgramModel, export
from ngramlm.keystrokes import keystroke_savings
from ngramlm.ngram import NgramModel


def test_hand_computed_example():
    # "le": suggested with no letter typed -> 1 tap instead of 3 keys
    # "chat": found after typing "c" -> 1 letter + 1 tap = 2 instead of 5
    # "zzz": out of vocabulary, typed in full = 4 keys, no saving
    def suggest(history, prefix):
        if history == [BOS] and prefix == "":
            return ["le"]
        if history == [BOS, "le"] and prefix == "c":
            return ["chat"]
        return []

    r = keystroke_savings([["le", "chat", "zzz"]], suggest, vocab={"le", "chat"})
    assert (r["keys_without"], r["keys_with"]) == (12, 7)
    assert r["savings_pct"] == pytest.approx(100 * 5 / 12)
    assert r["hit_without_typing_pct"] == pytest.approx(100 / 3)


def test_elision_has_no_space_and_a_tap_is_never_dearer_than_typing():
    r = keystroke_savings([["l'", "ami"]], lambda h, p: ["l'"] if p == "" else [], vocab={"l'", "ami"})
    assert (r["keys_without"], r["keys_with"]) == (2 + 4, 1 + 4)
    # a 1-letter word suggested after 0 letters costs 1 tap (vs 2 keys); suggested only at the
    # last letter it would cost 2 taps' worth, never more than typing it
    r = keystroke_savings([["a"]], lambda h, p: ["a"], vocab={"a"})
    assert (r["keys_without"], r["keys_with"]) == (2, 1)


def test_num_tokens_are_skipped_but_kept_in_the_history():
    seen = []
    keystroke_savings([["<num>", "le"]], lambda h, p: seen.append(list(h)) or [], vocab={"le"})
    assert seen[0] == [BOS, "<num>"]


def test_unknown_words_are_replaced_by_unk_in_the_history():
    seen = []
    keystroke_savings([["zzz", "le"]], lambda h, p: seen.append(list(h)) or [], vocab={"le"})
    assert seen[-1] == [BOS, "<unk>"]


def test_prefix_suggestions_match_brute_force(tmp_path):
    rng = random.Random(5)
    words = [f"w{i:02d}" for i in range(40)] + ["wa", "wab", "abc"]
    corpus = [rng.choices(words, [1 / (i + 1) for i in range(len(words))], k=rng.randint(2, 8)) for _ in range(300)]
    model = NgramModel(order=3).fit(corpus).prune([1, 2])
    export(model, tmp_path / "m.npz")
    compact = CompactNgramModel(tmp_path / "m.npz")
    for _ in range(80):
        s = rng.choice(corpus)
        history = [BOS] + s[: rng.randint(0, len(s))]
        for prefix in ("", "w", "w1", "wa", "wab", "abc", "z"):
            want = sorted(
                ((-model.prob(w, history), w) for w in model.words
                 if w.startswith(prefix) and w not in {BOS, "</s>", "<unk>"}),
            )
            want = [w for p, w in want if p < 0][:3]
            assert compact.suggest(history, 3, prefix=prefix) == want, (history, prefix)
