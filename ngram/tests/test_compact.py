import random

import pytest

from ngramlm import BOS
from ngramlm.compact import CompactNgramModel, export
from ngramlm.ngram import NgramModel


def corpus(seed=7, n=400):
    rng = random.Random(seed)
    words = [f"w{i:02d}" for i in range(40)] + ["<unk>"]
    weights = [1 / (i + 1) for i in range(41)]
    return [rng.choices(words, weights, k=rng.randint(2, 9)) for _ in range(n)]


@pytest.mark.parametrize("order,mins", [(2, [1]), (3, [1, 1]), (4, [1, 2, 2]), (4, [2, 2, 2])])
@pytest.mark.parametrize("compressed", [False, True])
def test_compact_model_matches_the_dict_model(tmp_path, order, mins, compressed):
    sentences = corpus()
    model = NgramModel(order=order).fit(sentences).prune(mins)
    path = tmp_path / "m.npz"
    export(model, path, compressed=compressed)
    compact = CompactNgramModel(path)

    rng = random.Random(1)
    for _ in range(120):
        s = rng.choice(sentences)
        history = [BOS] + s[: rng.randint(0, len(s))]
        assert compact.suggest(history, 3) == model.suggest(history, 3), history
        w = rng.choice(model.words)
        assert compact.prob(w, history) == pytest.approx(model.prob(w, history), rel=1e-12)
    assert compact.suggest(["inconnu", "autre"], 3) == model.suggest(["inconnu", "autre"], 3)
    assert compact.prob("inconnu", [BOS]) == 0.0


def test_compact_model_works_for_order_one(tmp_path):
    sentences = corpus()
    model = NgramModel(order=1).fit(sentences)
    export(model, tmp_path / "m.npz")
    compact = CompactNgramModel(tmp_path / "m.npz")
    assert compact.suggest([BOS], 3) == model.suggest([BOS], 3)
