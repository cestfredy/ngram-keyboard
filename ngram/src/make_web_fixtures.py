"""Build the tiny model and the reference answers used by the JavaScript tests.

Writes keyboard/tests/fixtures/tiny.ngm (a few KB, safe to commit) and tiny.json: for many
(history, prefix) pairs the exact suggestions and probabilities given by the Python model
(`CompactNgramModel`). The JS reader must reproduce them.

Usage (from the ngram/ directory): python src/make_web_fixtures.py
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from ngramlm import BOS
from ngramlm.compact import CompactNgramModel, export, export_web
from ngramlm.ngram import NgramModel
from ngramlm.text import normalize, tokenize

OUT = Path(__file__).resolve().parents[2] / "keyboard" / "tests" / "fixtures"


def main() -> None:
    rng = random.Random(11)
    words = [f"mot{i:02d}" for i in range(30)] + ["é", "été", "l'", "a", "à", "<unk>", "<num>", "est-ce"]
    weights = [1 / (i + 1) for i in range(len(words))]
    corpus = [rng.choices(words, weights, k=rng.randint(2, 9)) for _ in range(500)]
    model = NgramModel(order=4).fit(corpus).prune([1, 2, 2])
    OUT.mkdir(parents=True, exist_ok=True)
    export_web(model, OUT / "tiny.ngm")

    with tempfile.TemporaryDirectory() as tmp:
        export(model, Path(tmp) / "m.npz")
        ref = CompactNgramModel(Path(tmp) / "m.npz")

    cases, probs = [], []
    prefixes = ["", "m", "mo", "mot0", "mot1", "é", "é", "a", "l'", "est-", "zz", "été"]
    for _ in range(150):
        s = rng.choice(corpus)
        history = [BOS] + s[: rng.randint(0, len(s))]
        if rng.random() < 0.15:
            history = history + ["inconnu"]
        for prefix in rng.sample(prefixes, 3):
            cases.append({"history": history, "prefix": prefix, "k": 3,
                          "expected": ref.suggest(history, 3, prefix=prefix)})
        w = rng.choice(model.words)
        probs.append({"history": history, "word": w, "p": ref.prob(w, history)})
    probs.append({"history": [BOS], "word": "pas-dans-le-modele", "p": 0.0})
    texts = [
        "L’ami d’aujourd’hui est-ce qu'il a 3,5 euros ?", "Peut-être qu'il viendra à 18h30.",
        "J'ai trouvé 1.234 pommes", "C'est l'été !", "Qu'est-ce que c'est ? Ça va !", "aujourd'hui",
        "a-t-il dit", "l' ami", "naïve café coopération", "Œuf À LA COQUE", "A  B   C",
        "x — y", "j’ai\u00a0faim\u202f!", "ZERO\u200bWIDTH sof\u00adt", "il y a 12 ou 345,6 chats",
        "qu'est-ce qu'on mange ce soir", "n'importe quoi... vraiment", "(entre parenthèses) « guillemets »",
    ]
    tokens = [{"text": t, "tokens": tokenize(normalize(t))} for t in texts]
    (OUT / "tiny.json").write_text(
        json.dumps({"discount": model.discount, "order": model.order, "suggest": cases, "prob": probs, "tokens": tokens},
                   ensure_ascii=False), encoding="utf-8")
    size = (OUT / "tiny.ngm").stat().st_size
    print(f"tiny.ngm {size:,} bytes | {len(cases)} suggestion cases | {len(probs)} probability cases")


if __name__ == "__main__":
    main()
