"""Train interpolated Kneser-Ney n-gram models on TRAIN and evaluate them on VALID.

The test split is never read here. Run `prepare_data.py` first.

Usage (from the ngram/ directory):
  python src/evaluate_ngrams.py --orders 2 3 4 --discounts 0.75 --vocab 20000 [--out results/ngram_kn_valid.csv]
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from ngramlm.data import count_tokens, map_to_vocab, top_vocabulary
from ngramlm.evaluate import evaluate_topk
from ngramlm.ngram import NgramModel


def read_split(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--discounts", type=float, nargs="+", default=[0.75])
    ap.add_argument("--vocab", type=int, nargs="+", default=[20000])
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--out", default="results/ngram_kn_valid.csv", type=Path)
    ap.add_argument("--sample", type=int, default=0,
                    help="top-k on a fixed random sample of this many VALID sentences (0 = all); perplexity always uses all of VALID")
    ap.add_argument("--append", action="store_true", help="append to the output file instead of overwriting")
    args = ap.parse_args()

    train = read_split(args.data / "train.txt")
    valid = read_split(args.data / "valid.txt")
    counts = count_tokens(train)
    sample_idx = None
    if args.sample and args.sample < len(valid):
        sample_idx = sorted(random.Random(42).sample(range(len(valid)), args.sample))

    header = ["order", "discount", "vocab_size", "ngram_types", "top1_pct", "top3_pct",
              "top1_in_vocab_pct", "top3_in_vocab_pct", "perplexity_valid", "fit_seconds",
              "ms_per_prediction"]
    rows = []
    print(f"{'n':>2} {'d':>5} {'vocab':>7} {'n-gram types':>13} {'top-1 %':>8} {'top-3 %':>8} "
          f"{'PPL':>8} {'fit s':>6} {'ms/pred':>8}", flush=True)
    for size in args.vocab:
        vocab = top_vocabulary(counts, size)
        train_m = map_to_vocab(train, vocab)
        valid_m = map_to_vocab(valid, vocab)
        for order in args.orders:
            for d in args.discounts:
                t0 = time.perf_counter()
                model = NgramModel(order=order, discount=d).fit(train_m)
                fit_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                topk_set = [valid_m[i] for i in sample_idx] if sample_idx else valid_m
                res = evaluate_topk(topk_set, lambda history: model.suggest(history, 3), ks=(1, 3))
                ms = 1000 * (time.perf_counter() - t0) / res["positions"]
                ppl = model.perplexity(valid_m)
                types = sum(model.ngram_counts().values())

                rows.append([order, d, size, types, f"{res['top1_pct']:.2f}", f"{res['top3_pct']:.2f}",
                             f"{res['top1_in_vocab_pct']:.2f}", f"{res['top3_in_vocab_pct']:.2f}",
                             f"{ppl:.2f}", f"{fit_s:.1f}", f"{ms:.3f}"])
                print(f"{order:>2} {d:>5} {size:>7,} {types:>13,} {res['top1_pct']:>8.2f} "
                      f"{res['top3_pct']:>8.2f} {ppl:>8.1f} {fit_s:>6.1f} {ms:>8.3f}", flush=True)
                del model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    with open(args.out, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
