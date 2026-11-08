"""Unigram baseline: always suggest the 3 most frequent words. Evaluated on the VALIDATION split.

The test split is never read here. Run `prepare_data.py` first.

Usage (from the ngram/ directory): python src/evaluate_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path

from ngramlm.data import count_tokens, map_to_vocab, top_vocabulary
from ngramlm.evaluate import evaluate_topk, unigram_suggester


def read_split(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml", type=Path)
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--results", default="results", type=Path)
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    sizes = cfg["vocab"]["sizes_to_try"]

    train = read_split(args.data / "train.txt")
    valid = read_split(args.data / "valid.txt")
    counts = count_tokens(train)

    rows = []
    for size in sizes:
        vocab = top_vocabulary(counts, size)
        suggest = unigram_suggester(counts, vocab, k=3)
        res = evaluate_topk(map_to_vocab(valid, vocab), suggest, ks=(1, 3))
        rows.append(
            [size, int(res["positions"]), int(res["in_vocab_positions"]),
             f"{res['top1_pct']:.2f}", f"{res['top3_pct']:.2f}",
             f"{res['top1_in_vocab_pct']:.2f}", f"{res['top3_in_vocab_pct']:.2f}",
             " ".join(suggest(["<s>"]))]
        )

    args.results.mkdir(parents=True, exist_ok=True)
    header = ["vocab_size", "valid_positions", "valid_in_vocab_positions", "top1_pct", "top3_pct",
              "top1_in_vocab_pct", "top3_in_vocab_pct", "suggestions"]
    with open(args.results / "baseline_unigram.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("== Unigram baseline on the validation split ==")
    print(f"{'vocab':>8} {'positions':>10} {'top-1 %':>8} {'top-3 %':>8} {'top-1 known %':>14} {'top-3 known %':>14}  suggestions")
    for size, pos, _, t1, t3, t1k, t3k, sugg in rows:
        print(f"{size:>8,} {pos:>10,} {t1:>8} {t3:>8} {t1k:>14} {t3k:>14}  {sugg}")


if __name__ == "__main__":
    main()
