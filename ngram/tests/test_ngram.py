import math
import random
from pathlib import Path

import pytest

from ngramlm import BOS, EOS, BigramModel
from ngramlm.ngram import NgramModel

CORPUS = Path(__file__).parent / "minicorpus.txt"
APPROX = dict(rel=1e-12)


@pytest.fixture(scope="module")
def sentences():
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines if line.strip()]


@pytest.fixture(scope="module")
def bigram(sentences):
    return NgramModel(order=2).fit(sentences)


@pytest.fixture(scope="module")
def trigram(sentences):
    return NgramModel(order=3).fit(sentences)


# ---------------------------------------------- order 2 = the course PDF
def test_order2_reproduces_the_pdf_kneser_ney_values(bigram):
    assert bigram.prob("souris", [BOS, "le", "chat"]) == pytest.approx(1 / 26, **APPROX)
    assert bigram.prob("mange", [BOS, "le", "chat"]) == pytest.approx(77 / 156, **APPROX)
    assert bigram.prob("dort", [BOS, "le", "chat"]) == pytest.approx(19 / 156, **APPROX)


def test_order2_matches_the_bigram_model_everywhere(sentences, bigram):
    reference = BigramModel(sentences)
    for history in [BOS, *reference.vocab]:
        for word in reference.vocab:
            assert bigram.prob(word, [history]) == pytest.approx(
                reference.p_kneser_ney(word, history), **APPROX
            )


def test_order2_perplexity_matches_the_bigram_model(sentences, bigram):
    reference = BigramModel(sentences)
    for sentence in (["le", "chat", "dort"], ["la", "souris", "mange", "le", "fromage"]):
        expected = reference.perplexity(sentence, reference.p_kneser_ney)
        assert bigram.perplexity([sentence]) == pytest.approx(expected, **APPROX)


# --------------------------------- order 3, derived by hand from the definition
def test_trigram_value_computed_by_hand(trigram):
    """P3(fromage | mange le), d = 0.75, sentences padded with two <s>.

    Order 3: context (mange le) is followed only by fromage, count 2, so
        P3 = (2 - .75)/2 + .75 * 1/2 * P2(fromage | le)
    Order 2: continuation counts after "le": chat 1 and fromage 1 (total 2, 2 followers), so
        P2 = (1 - .75)/2 + .75 * 2/2 * P1(fromage)
    Order 1: "fromage" follows 1 distinct word out of 13 distinct bigram types, so P1 = 1/13.
        P2 = 1/8 + 3/52 = 19/104        P3 = 5/8 + 3/8 * 19/104 = 577/832
    """
    assert trigram.prob("fromage", ["mange", "le"]) == pytest.approx(577 / 832, **APPROX)


def test_trigram_sentence_start_uses_raw_counts(trigram):
    # Every sentence starts after (<s> <s>): P(le | <s> <s>) must follow raw counts, not 1/#types.
    # raw: le 3, la 1 (4 sentences); followers 2: P3 = (3-.75)/4 + .75*2/4*P2(le | <s>), and
    # P2(le | <s>) = (3-.75)/4 + .75*2/4*P1(le) with raw counts propagated for <s>-initial grams.
    # "le" follows 2 distinct words (<s> and "mange") among the 13 bigram types of the order-2 level
    p1_le = 2 / 13
    assert p1_le == pytest.approx(trigram.num[1][(trigram.ids["le"],)] / trigram._unigram_total)
    p2 = (3 - 0.75) / 4 + 0.75 * 2 / 4 * p1_le
    p3 = (3 - 0.75) / 4 + 0.75 * 2 / 4 * p2
    assert trigram.prob("le", [BOS, BOS]) == pytest.approx(p3, **APPROX)


# --------------------------------------------------- proper distributions
@pytest.mark.parametrize("order", [1, 2, 3])
def test_distribution_sums_to_one(sentences, order):
    model = NgramModel(order=order).fit(sentences)
    histories = [[BOS], [BOS, "le"], ["mange", "le"], ["chat"], ["fromage", "dort"], ["inconnu"], []]
    for history in histories:
        total = sum(model.prob(w, history) for w in model.words)
        assert total == pytest.approx(1.0, rel=1e-9), history


# -------------------------------------------- suggestions are exact top-k
def brute_force_top_k(model, history, k):
    excluded = {BOS, EOS, "<unk>"}
    scored = [(-model.prob(w, history), w) for w in model.words if w not in excluded]
    scored = [(p, w) for p, w in scored if p < 0.0]
    scored.sort()
    return [w for _, w in scored[:k]]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_suggest_equals_brute_force_on_the_mini_corpus(sentences, order):
    model = NgramModel(order=order).fit(sentences)
    for history in [[BOS], [BOS, "le"], ["le", "chat"], ["mange", "le"], ["chat"], ["souris", "mange"], ["inconnu"]]:
        assert model.suggest(history, 3) == brute_force_top_k(model, history, 3), history


@pytest.mark.parametrize("order", [2, 3, 4])
def test_suggest_equals_brute_force_on_a_random_corpus(order):
    rng = random.Random(7)
    words = [f"w{i:02d}" for i in range(40)]
    weights = [1 / (i + 1) for i in range(40)]  # Zipf-like
    corpus = [rng.choices(words, weights, k=rng.randint(2, 9)) for _ in range(400)]
    model = NgramModel(order=order).fit(corpus)
    for _ in range(60):
        sentence = rng.choice(corpus)
        cut = rng.randint(0, len(sentence))
        history = [BOS] + sentence[:cut]
        assert model.suggest(history, 3) == brute_force_top_k(model, history, 3), history


def test_suggest_never_returns_special_tokens():
    corpus = [["a", "<unk>", "b"], ["a", "b", "<unk>"], ["b", "a"]]
    model = NgramModel(order=2).fit(corpus)
    for history in ([BOS], [BOS, "a"], ["b"]):
        out = model.suggest(history, 5)
        assert BOS not in out and EOS not in out and "<unk>" not in out


# ------------------------------------------------------------ validation
def test_invalid_parameters():
    with pytest.raises(ValueError):
        NgramModel(order=0)
    for d in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            NgramModel(order=2, discount=d)


def test_unknown_word_has_zero_probability_and_infinite_perplexity(bigram):
    assert bigram.prob("inconnu", ["le"]) == 0.0
    assert bigram.perplexity([["le", "inconnu"]]) == math.inf


# ------------------------------------------------------------------ pruning
def random_corpus(seed=7, sentences=400):
    rng = random.Random(seed)
    words = [f"w{i:02d}" for i in range(40)]
    weights = [1 / (i + 1) for i in range(40)]
    return [rng.choices(words, weights, k=rng.randint(2, 9)) for _ in range(sentences)]


def test_prune_with_thresholds_of_one_changes_nothing(sentences):
    reference = NgramModel(order=3).fit(sentences)
    pruned = NgramModel(order=3).fit(sentences).prune([1, 1])
    assert pruned.ngram_counts() == reference.ngram_counts()
    for history in (["mange", "le"], [BOS, "le"], ["chat"]):
        for w in reference.words:
            assert pruned.prob(w, history) == pytest.approx(reference.prob(w, history), **APPROX)


def test_prune_removes_entries_and_keeps_unigrams(sentences):
    full = NgramModel(order=3).fit(sentences)
    pruned = NgramModel(order=3).fit(sentences).prune([2, 2])
    before, after = full.ngram_counts(), pruned.ngram_counts()
    assert after[1] == before[1]
    assert after[2] < before[2] and after[3] < before[3]
    assert all(c >= 2 for c in pruned.num[2].values()) and all(c >= 2 for c in pruned.num[3].values())


@pytest.mark.parametrize("thresholds", [[2, 2], [1, 3], [3, 2]])
def test_pruned_distribution_still_sums_to_one(thresholds):
    model = NgramModel(order=3).fit(random_corpus()).prune(thresholds)
    for history in ([BOS, "w00"], ["w01", "w00"], ["w30", "w31"], ["w39"], ["inconnu"], []):
        total = sum(model.prob(w, history) for w in model.words)
        assert total == pytest.approx(1.0, rel=1e-9), history


@pytest.mark.parametrize("order,thresholds", [(3, [2, 2]), (4, [1, 2, 3]), (4, [2, 2, 2])])
def test_pruned_suggest_equals_brute_force(order, thresholds):
    corpus = random_corpus()
    model = NgramModel(order=order).fit(corpus).prune(thresholds)
    rng = random.Random(3)
    for _ in range(60):
        sentence = rng.choice(corpus)
        history = [BOS] + sentence[: rng.randint(0, len(sentence))]
        assert model.suggest(history, 3) == brute_force_top_k(model, history, 3), history


def test_pruned_history_falls_back_to_the_lower_order(sentences):
    model = NgramModel(order=3).fit(sentences).prune([1, 100])  # every trigram removed
    assert model.prob("fromage", ["mange", "le"]) == pytest.approx(
        model.prob("fromage", ["le"]), **APPROX
    )
    assert model.ngram_counts()[3] == 0


def test_prune_rejects_bad_thresholds(sentences):
    model = NgramModel(order=3).fit(sentences)
    with pytest.raises(ValueError):
        model.prune([2])
    with pytest.raises(ValueError):
        model.prune([0, 2])
