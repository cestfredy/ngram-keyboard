"""Reference values computed by hand on the 4-sentence mini-corpus (see the course PDF).

If an implementation reproduces these numbers exactly, its counting and smoothing
logic is correct and can be trusted on the real corpus.
"""

import math
from pathlib import Path

import pytest

from ngramlm import BOS, EOS, BigramModel

CORPUS = Path(__file__).parent / "minicorpus.txt"
APPROX = dict(rel=1e-12)


@pytest.fixture(scope="module")
def model() -> BigramModel:
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    return BigramModel(line.split() for line in lines if line.strip())


# ---------------------------------------------------------------- counts
def test_counts(model):
    assert model.total == 22  # N
    assert model.size == 8  # V: 7 words + </s>
    assert model.bigram_types == 13
    assert model.context["chat"] == 3
    assert model.bigrams[("chat", "mange")] == 2
    assert model.bigrams[("chat", "souris")] == 0
    assert BOS not in model.vocab and EOS in model.vocab


# ------------------------------------------------------------------- MLE
def test_mle_bigram(model):
    assert model.p_mle("mange", "chat") == pytest.approx(2 / 3, **APPROX)
    assert model.p_mle("dort", "chat") == pytest.approx(1 / 3, **APPROX)
    assert model.p_mle("souris", "chat") == 0.0
    assert model.p_mle("la", BOS) == pytest.approx(1 / 4, **APPROX)
    assert model.p_mle("souris", "la") == pytest.approx(1.0, **APPROX)
    assert model.p_mle(EOS, "fromage") == pytest.approx(1.0, **APPROX)


def test_sentence_probability(model):
    p = model.sentence_probability("le chat dort".split(), model.p_mle)
    assert p == pytest.approx(3 / 20, **APPROX)
    p = model.sentence_probability("le chat mange le fromage".split(), model.p_mle)
    assert p == pytest.approx(2 / 25, **APPROX)


# --------------------------------------------------------------- Laplace
def test_laplace(model):
    assert model.p_add_k("souris", "chat") == pytest.approx(1 / 11, **APPROX)
    assert model.p_add_k("mange", "chat") == pytest.approx(3 / 11, **APPROX)
    assert model.p_add_k("dort", "chat") == pytest.approx(2 / 11, **APPROX)


# --------------------------------------------------------- interpolation
def test_interpolation(model):
    assert model.p_interpolated("souris", "chat", 0.7) == pytest.approx(3 / 110, **APPROX)
    assert model.p_interpolated("mange", "chat", 0.7) == pytest.approx(67 / 132, **APPROX)
    assert model.p_interpolated("souris", "chat", 0.5) == pytest.approx(1 / 22, **APPROX)
    assert model.p_interpolated("mange", "chat", 0.5) == pytest.approx(53 / 132, **APPROX)


# ----------------------------------------------------------- Kneser-Ney
def test_kneser_ney(model):
    assert model.p_continuation("souris") == pytest.approx(1 / 13, **APPROX)
    assert model.p_continuation("dort") == pytest.approx(1 / 13, **APPROX)
    assert model.p_kneser_ney("souris", "chat") == pytest.approx(1 / 26, **APPROX)
    assert model.p_kneser_ney("mange", "chat") == pytest.approx(77 / 156, **APPROX)
    assert model.p_kneser_ney("dort", "chat") == pytest.approx(19 / 156, **APPROX)


# ------------------------------------------- distributions sum to one
HISTORIES = [BOS, "le", "chat", "mange", "la", "souris", "fromage", "dort"]


@pytest.mark.parametrize("history", HISTORIES)
@pytest.mark.parametrize("name", ["mle", "add_k", "interpolated", "kneser_ney"])
def test_distribution_sums_to_one(model, history, name):
    prob = {
        "mle": model.p_mle,
        "add_k": model.p_add_k,
        "interpolated": model.p_interpolated,
        "kneser_ney": model.p_kneser_ney,
    }[name]
    total = sum(prob(w, history) for w in model.vocab)
    expected = 1.0
    if name == "mle" and model.context[history] == 0:
        expected = 0.0  # no data for this history
    assert total == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------ perplexity
def test_perplexity_values(model):
    ppl = model.perplexity("le chat dort".split(), model.p_mle)
    assert ppl == pytest.approx((20 / 3) ** 0.25, **APPROX)

    ppl = model.perplexity("la souris dort".split(), model.p_add_k)
    assert ppl == pytest.approx(math.sqrt(30), **APPROX)

    ppl = model.perplexity("le fromage dort".split(), model.p_add_k)
    assert ppl == pytest.approx(585**0.25, **APPROX)


def test_perplexity_is_infinite_without_smoothing(model):
    assert model.perplexity("la souris dort".split(), model.p_mle) == math.inf


# ----------------------------------------------------------------- top-k
def test_top_k(model):
    # After "le": chat (3/5) then fromage (2/5); zero-probability words are dropped.
    assert model.top_k("le", 3, model.p_mle) == ["chat", "fromage"]
    # Smoothed models fill the remaining slots, </s> is never suggested.
    suggestions = model.top_k("le", 3, model.p_kneser_ney)
    assert suggestions[:2] == ["chat", "fromage"]
    assert len(suggestions) == 3 and EOS not in suggestions
