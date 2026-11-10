"""Final evaluation on the TEST split. Run once, at the very end.

Everything is frozen before this script runs: vocabulary (20,000 words, train only), order 4,
discount 0.75, the two shipped pruning thresholds (standard 2,2,2 and light 2,3,4). Nothing is tuned here.

The script takes a lock before it reads the test split and refuses to run again once the lock or the
results file exists, so the test set cannot be reused by accident. To check that the code works,
run it on the validation split first (`--split valid`, no lock, `--sample N` for a quick run).

Metrics (same definitions as on validation):
  * top-1 / top-3 with no letter typed, every word position of every sentence (`</s>` excluded),
    a target outside the vocabulary counts as a miss;
  * perplexity (`</s>` included);
  * keystroke savings of the typing simulation, 3 suggestions per prefix.

Usage (from the ngram/ directory, after export_model.py has written the models):
  python src/evaluate_test.py                       # the real, single test run
  python src/evaluate_test.py --split valid --sample 300 --out /tmp/check.csv    # dry run
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

from ngramlm.compact import CompactNgramModel
from ngramlm.data import count_tokens, map_to_vocab, top_vocabulary
from ngramlm.evaluate import evaluate_topk, unigram_suggester
from ngramlm.keystrokes import keystroke_savings
from ngramlm.ngram import NgramModel

ORDER, DISCOUNT, VOCAB = 4, 0.75, 20000
STANDARD, LIGHT = [2, 2, 2], [2, 3, 4]


def read_split(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "valid"], default="test")
    ap.add_argument("--sample", type=int, default=0, help="only for a dry run on valid (0 = all sentences)")
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--models-dir", default="models", type=Path)
    ap.add_argument("--out", default="results/test_final.csv", type=Path)
    args = ap.parse_args()

    lock = args.out.with_suffix(".lock")
    if args.split == "test":
        if lock.exists() or args.out.exists():
            sys.exit(f"The test split was already evaluated ({lock} or {args.out} exists). Not running again.")
        if args.sample:
            sys.exit("--sample is for dry runs on the validation split only.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(time.strftime("test split read at %Y-%m-%d %H:%M:%S\n"))

    train = read_split(args.data / "train.txt")
    vocab = top_vocabulary(count_tokens(train), VOCAB)
    train_m = map_to_vocab(train, vocab)
    del train
    raw = read_split(args.data / f"{args.split}.txt")
    if args.sample:
        raw = [raw[i] for i in sorted(random.Random(42).sample(range(len(raw)), args.sample))]
    eval_m = map_to_vocab(raw, vocab)
    print(f"{args.split}: {len(raw):,} sentences", flush=True)

    rows = []

    def record(name: str, entries: int | str, top: dict, ppl: float | str, ks: dict | None) -> None:
        row = [name, args.split, len(raw), top["positions"], entries,
               f"{top['top1_pct']:.2f}", f"{top['top3_pct']:.2f}",
               f"{ppl:.2f}" if isinstance(ppl, float) else ppl,
               f"{ks['savings_pct']:.2f}" if ks else "", f"{ks['hit_without_typing_pct']:.2f}" if ks else ""]
        rows.append(row)
        print("  ".join(str(x) for x in row), flush=True)

    # 1. unigram baseline: same three words everywhere; keystroke savings with prefix filtering
    counts = count_tokens(train_m)
    top = evaluate_topk(eval_m, unigram_suggester(counts, vocab, 3))
    uni = CompactNgramModel(args.models_dir / "o1_v20000.npz")
    words = set(uni.tokens) - {"<s>", "</s>", "<unk>"}
    ks = keystroke_savings(raw, lambda h, p: uni.suggest(h, 3, prefix=p), words)
    record("unigram baseline", "", top, "n/a", ks)

    # 2. 4-gram, perplexity of every variant (dict model), top-k of the unpruned one
    model = NgramModel(order=ORDER, discount=DISCOUNT).fit(train_m)
    entries_full = sum(model.ngram_counts().values())
    t0 = time.perf_counter()
    top_full = evaluate_topk(eval_m, lambda h: model.suggest(h, 3))
    ppl_full = model.perplexity(eval_m)
    print(f"(unpruned top-k + perplexity took {time.perf_counter() - t0:.0f} s)", flush=True)
    record("4-gram, no pruning", entries_full, top_full, ppl_full, None)

    ppl, entries = {}, {}
    for name, mins in (("standard", STANDARD), ("light", LIGHT)):  # pruning is monotone: light after standard
        model.prune(mins)
        entries[name] = sum(model.ngram_counts().values())
        ppl[name] = model.perplexity(eval_m)
    del model

    # 3. shipped models: top-k and keystroke savings from the exported compact models
    for name, file in (("4-gram, standard (2,2,2)", "o4_v20000_m222.npz"), ("4-gram, light (2,3,4)", "o4_v20000_m234.npz")):
        key = "standard" if "standard" in name else "light"
        cm = CompactNgramModel(args.models_dir / file)
        words = set(cm.tokens) - {"<s>", "</s>", "<unk>"}
        top = evaluate_topk(eval_m, lambda h: cm.suggest(h, 3))
        ks = keystroke_savings(raw, lambda h, p: cm.suggest(h, 3, prefix=p), words)
        record(name, entries[key], top, ppl[key], ks)

    header = ["model", "split", "sentences", "positions", "entries", "top1_pct", "top3_pct",
              "perplexity", "keystroke_savings_pct", "hit_without_typing_pct"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
