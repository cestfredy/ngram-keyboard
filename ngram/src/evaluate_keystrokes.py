"""Keystroke savings on VALID for exported models, plus a no-context unigram reference.

The test split is never read. See `ngramlm/keystrokes.py` for the exact definition.

Usage (from the ngram/ directory; run `export_model.py` first):
  python src/evaluate_keystrokes.py --models models/o4_v20000_m222.npz models/o4_v20000_m234.npz
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from ngramlm.compact import CompactNgramModel, export
from ngramlm.data import count_tokens, map_to_vocab, top_vocabulary
from ngramlm.keystrokes import keystroke_savings
from ngramlm.ngram import NgramModel


def read_split(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--unigram-vocab", type=int, default=20000,
                    help="vocabulary size of the no-context reference (0 = skip it)")
    ap.add_argument("--sample", type=int, default=5000, help="VALID sentences (0 = all)")
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--models-dir", default="models", type=Path)
    ap.add_argument("--out", default="results/ngram_kn_keystrokes.csv", type=Path)
    args = ap.parse_args()

    valid = read_split(args.data / "valid.txt")
    if args.sample and args.sample < len(valid):
        idx = sorted(random.Random(42).sample(range(len(valid)), args.sample))
        valid = [valid[i] for i in idx]

    paths = list(args.models)
    if args.unigram_vocab:
        train = read_split(args.data / "train.txt")
        vocab = top_vocabulary(count_tokens(train), args.unigram_vocab)
        uni = NgramModel(order=1).fit(map_to_vocab(train, vocab))
        uni_path = args.models_dir / f"o1_v{args.unigram_vocab}.npz"
        args.models_dir.mkdir(parents=True, exist_ok=True)
        export(uni, uni_path)
        paths.insert(0, uni_path)
        del train, uni

    header = ["model", "words", "keys_without", "keys_with", "savings_pct",
              "hit_without_typing_pct", "hit_before_end_pct", "avg_letters_before_hit", "ms_per_word"]
    rows = []
    print(f"{'model':<24} {'words':>8} {'keys w/o':>9} {'keys with':>9} {'savings %':>9} "
          f"{'hit @0 %':>8} {'hit %':>6} {'avg letters':>11} {'ms/word':>8}", flush=True)
    for path in paths:
        model = CompactNgramModel(path)
        vocab = set(model.tokens) - {"<s>", "</s>", "<unk>"}
        t0 = time.perf_counter()
        r = keystroke_savings(valid, lambda h, p: model.suggest(h, 3, prefix=p), vocab)
        ms = 1000 * (time.perf_counter() - t0) / r["words"]
        rows.append([path.stem, r["words"], r["keys_without"], r["keys_with"], f"{r['savings_pct']:.2f}",
                     f"{r['hit_without_typing_pct']:.2f}", f"{r['hit_before_end_pct']:.2f}",
                     f"{r['avg_letters_before_hit']:.2f}", f"{ms:.3f}"])
        print(f"{path.stem:<24} {r['words']:>8,} {r['keys_without']:>9,} {r['keys_with']:>9,} "
              f"{r['savings_pct']:>9.2f} {r['hit_without_typing_pct']:>8.2f} "
              f"{r['hit_before_end_pct']:>6.2f} {r['avg_letters_before_hit']:>11.2f} {ms:>8.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
