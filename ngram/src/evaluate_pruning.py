"""Size / accuracy trade-off of count-threshold pruning on VALID (test is never read).

Each configuration is a comma-separated list of minimum counts for orders 2..n, e.g. `1,2,2`
keeps every bigram, trigrams and 4-grams seen at least twice (1 = keep all).

Usage (from the ngram/ directory):
  python src/evaluate_pruning.py --order 4 --vocab 20000 --sample 5000 --configs 1,1,1 1,2,2 2,2,2
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
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--discount", type=float, default=0.75)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--out", default="results/ngram_kn_pruning.csv", type=Path)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    train = read_split(args.data / "train.txt")
    valid = read_split(args.data / "valid.txt")
    vocab = top_vocabulary(count_tokens(train), args.vocab)
    train_m, valid_m = map_to_vocab(train, vocab), map_to_vocab(valid, vocab)
    del train
    if args.sample and args.sample < len(valid_m):
        idx = sorted(random.Random(42).sample(range(len(valid_m)), args.sample))
        topk_set = [valid_m[i] for i in idx]
    else:
        topk_set = valid_m

    header = ["order", "discount", "vocab_size", "min_counts", "entries", "est_raw_mb",
              "top1_pct", "top3_pct", "perplexity_valid", "ms_per_prediction"]
    rows = []
    print(f"{'min counts':>10} {'entries':>10} {'est MB':>7} {'top-1 %':>8} {'top-3 %':>8} "
          f"{'PPL':>7} {'ms/pred':>8}", flush=True)
    for cfg in args.configs:
        mins = [int(x) for x in cfg.split(",")]
        model = NgramModel(order=args.order, discount=args.discount).fit(train_m)
        model.prune(mins)
        counts = model.ngram_counts()
        entries = sum(counts.values())
        # naive raw size: 4 bytes per word id + 4 bytes for the count, per entry
        est_mb = sum(n * (4 * k + 4) for k, n in counts.items()) / 1e6

        t0 = time.perf_counter()
        res = evaluate_topk(topk_set, lambda h: model.suggest(h, 3), ks=(1, 3))
        ms = 1000 * (time.perf_counter() - t0) / res["positions"]
        ppl = model.perplexity(valid_m)
        rows.append([args.order, args.discount, args.vocab, cfg, entries, f"{est_mb:.1f}",
                     f"{res['top1_pct']:.2f}", f"{res['top3_pct']:.2f}", f"{ppl:.2f}", f"{ms:.3f}"])
        print(f"{cfg:>10} {entries:>10,} {est_mb:>7.1f} {res['top1_pct']:>8.2f} "
              f"{res['top3_pct']:>8.2f} {ppl:>7.1f} {ms:>8.3f}", flush=True)
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
