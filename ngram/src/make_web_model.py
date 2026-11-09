"""Train, prune and export the models shipped with the web keyboard, plus parity references.

Fits once on TRAIN (vocabulary 20,000), writes keyboard/models/ngram-standard.ngm (min counts
2,2,2) then ngram-light.ngm (2,3,4; pruning is monotone, so pruning the standard model further
gives exactly the light model). For each, a JSON file of exact Python answers on VALID positions
(`parity_<name>.json`) lets `keyboard/tests/parity.mjs` check the JavaScript reader on the real
model. keyboard/models/ is git-ignored. The test split is never read.

Usage (from the ngram/ directory): python src/make_web_model.py
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from ngramlm import BOS
from ngramlm.compact import CompactNgramModel, export, export_web
from ngramlm.data import count_tokens, map_to_vocab, top_vocabulary
from ngramlm.ngram import NgramModel

ROOT = Path(__file__).resolve().parents[2]


def read_split(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def parity_cases(ref: CompactNgramModel, valid_m: list[list[str]], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        s = rng.choice(valid_m)
        i = rng.randrange(len(s))
        history = [BOS] + s[:i]
        word = s[i]
        for prefix in ("", word[:1], word[:2]):
            if word == "<unk>" and prefix:
                continue
            cases.append({"history": history, "prefix": prefix, "expected": ref.suggest(history, 3, prefix=prefix)})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--data", default=ROOT / "ngram" / "data" / "processed", type=Path)
    ap.add_argument("--out", default=ROOT / "keyboard" / "models", type=Path)
    args = ap.parse_args()

    train = read_split(args.data / "train.txt")
    valid = read_split(args.data / "valid.txt")
    vocab = top_vocabulary(count_tokens(train), args.vocab)
    train_m, valid_m = map_to_vocab(train, vocab), map_to_vocab(valid, vocab)
    del train
    args.out.mkdir(parents=True, exist_ok=True)

    model = NgramModel(order=4).fit(train_m)
    for name, mins in (("standard", [2, 2, 2]), ("light", [2, 3, 4])):
        model.prune(mins)
        entries = sum(model.ngram_counts().values())
        path = args.out / f"ngram-{name}.ngm"
        export_web(model, path)
        with tempfile.TemporaryDirectory() as tmp:
            export(model, Path(tmp) / "m.npz")
            ref = CompactNgramModel(Path(tmp) / "m.npz")
            cases = parity_cases(ref, valid_m, args.cases, seed=42)
        (args.out / f"parity_{name}.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: min counts {mins} | {entries:,} entries | {path.stat().st_size / 1e6:.2f} MB | {len(cases)} parity cases")


if __name__ == "__main__":
    main()
