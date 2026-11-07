"""Prepare the Tatoeba French corpus: clean, tokenise, deduplicate, split, build the vocabulary.

Order (each step protects the evaluation):
  1. clean + tokenise every sentence            (rules in ngramlm/text.py)
  2. deduplicate on token sequences             (before the split, so no sentence is in two splits)
  3. shuffle with the configured seed, split 90/5/5
  4. count tokens and choose the vocabulary on TRAIN only
  5. report out-of-vocabulary rates on VALID only (the test split is written but never read here)

Usage (from the ngram/ directory):
  python src/prepare_data.py [--raw data/raw/fra_sentences.tsv.bz2] [--config config.toml]
"""

from __future__ import annotations

import argparse
import bz2
import csv
import tomllib
from pathlib import Path

from ngramlm.data import count_tokens, deduplicate, oov_stats, split, top_vocabulary
from ngramlm.text import clean


def read_texts(path: Path):
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 3:
                yield parts[2]


def write_split(path: Path, sentences) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/fra_sentences.tsv.bz2", type=Path)
    ap.add_argument("--config", default="config.toml", type=Path)
    ap.add_argument("--out", default="data/processed", type=Path)
    ap.add_argument("--results", default="results", type=Path)
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    seed = cfg["project"]["seed"]
    r_train, r_valid = cfg["split"]["train"], cfg["split"]["valid"]
    sizes = cfg["vocab"]["sizes_to_try"]

    # 1. clean + tokenise
    raw_count = 0
    kept: list[list[str]] = []
    for text in read_texts(args.raw):
        raw_count += 1
        tokens = clean(text)
        if tokens is not None:
            kept.append(tokens)
    # 2. deduplicate
    unique = deduplicate(kept)
    # 3. split
    train, valid, test = split(unique, seed=seed, train=r_train, valid=r_valid)

    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    write_split(args.out / "train.txt", train)
    write_split(args.out / "valid.txt", valid)
    write_split(args.out / "test.txt", test)

    # 4. vocabulary from TRAIN only
    counts = count_tokens(train)
    with open(args.out / "vocab_counts.tsv", "w", encoding="utf-8") as f:
        for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{tok}\t{c}\n")

    # pipeline summary
    stages = [
        ("raw sentences", raw_count),
        ("rejected by cleaning filters", raw_count - len(kept)),
        ("duplicates removed (token level)", len(kept) - len(unique)),
        ("kept after cleaning and dedup", len(unique)),
        ("train", len(train)),
        ("valid", len(valid)),
        ("test", len(test)),
        ("train tokens", sum(len(s) for s in train)),
        ("train distinct tokens", len(counts)),
    ]
    with open(args.results / "data_pipeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "count"])
        w.writerows(stages)
    print("== Pipeline ==")
    for name, value in stages:
        print(f"{name:<36}: {value:>10,}")

    # 5. OOV on VALID only
    rows = []
    for size in sizes:
        vocab = top_vocabulary(counts, size)
        st = oov_stats(valid, vocab)
        rows.append(
            [size, len(vocab), st["tokens"], st["oov_tokens"],
             f"{st['oov_rate_pct']:.3f}", f"{st['sentences_with_oov_pct']:.2f}"]
        )
    with open(args.results / "oov_by_vocab_size.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vocab_size_requested", "vocab_size_actual", "valid_tokens",
                    "valid_oov_tokens", "oov_rate_pct", "sentences_with_oov_pct"])
        w.writerows(rows)
    print("\n== OOV on the validation split ==")
    print(f"{'vocab':>8} {'actual':>8} {'OOV tokens':>11} {'OOV %':>8} {'sentences with OOV %':>22}")
    for size, actual, _, oov, rate, sent in rows:
        print(f"{size:>8,} {actual:>8,} {oov:>11,} {rate:>8} {sent:>22}")


if __name__ == "__main__":
    main()
