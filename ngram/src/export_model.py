"""Train, prune and export compact models, then measure their real size, loading time and accuracy.

Reads TRAIN (to fit) and VALID (to measure). The test split is never read.

Usage (from the ngram/ directory):
  python src/export_model.py --order 4 --vocab 20000 --configs 2,2,2 2,3,4
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import resource
import time
from pathlib import Path

from ngramlm import BOS
from ngramlm.compact import CompactNgramModel, export
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
    ap.add_argument("--models", default="models", type=Path)
    ap.add_argument("--out", default="results/ngram_kn_export.csv", type=Path)
    args = ap.parse_args()

    train = read_split(args.data / "train.txt")
    valid = read_split(args.data / "valid.txt")
    vocab = top_vocabulary(count_tokens(train), args.vocab)
    train_m, valid_m = map_to_vocab(train, vocab), map_to_vocab(valid, vocab)
    del train
    idx = sorted(random.Random(42).sample(range(len(valid_m)), args.sample))
    sample = [valid_m[i] for i in idx]
    args.models.mkdir(parents=True, exist_ok=True)

    header = ["order", "vocab_size", "min_counts", "entries", "file_mb", "file_mb_compressed",
              "load_seconds", "top1_pct", "top3_pct", "ms_per_prediction", "identical_to_dict_model"]
    rows = []
    for cfg in args.configs:
        mins = [int(x) for x in cfg.split(",")]
        model = NgramModel(order=args.order, discount=args.discount).fit(train_m).prune(mins)
        entries = sum(model.ngram_counts().values())
        tag = f"o{args.order}_v{args.vocab}_m{cfg.replace(',', '')}"
        raw_path, zip_path = args.models / f"{tag}.npz", args.models / f"{tag}_compressed.npz"
        export(model, raw_path)
        export(model, zip_path, compressed=True)

        t0 = time.perf_counter()
        compact = CompactNgramModel(raw_path)
        load_s = time.perf_counter() - t0

        # the compact model must give exactly the same suggestions as the dict model
        same, checked = True, 0
        for s in sample[:1000]:
            h = [BOS]
            for w in s:
                same &= compact.suggest(h, 3) == model.suggest(h, 3)
                checked += 1
                h.append(w)
        t0 = time.perf_counter()
        res = evaluate_topk(sample, lambda h: compact.suggest(h, 3), ks=(1, 3))
        ms = 1000 * (time.perf_counter() - t0) / res["positions"]
        rows.append([args.order, args.vocab, cfg, entries, f"{raw_path.stat().st_size / 1e6:.2f}",
                     f"{zip_path.stat().st_size / 1e6:.2f}", f"{load_s:.2f}",
                     f"{res['top1_pct']:.2f}", f"{res['top3_pct']:.2f}", f"{ms:.3f}", same])
        print(f"{cfg}: {entries:,} entries | file {rows[-1][4]} MB (zip {rows[-1][5]} MB) | "
              f"load {load_s:.2f}s | top-1 {rows[-1][7]} top-3 {rows[-1][8]} | {ms:.3f} ms/pred | "
              f"identical to dict model on {checked:,} positions: {same}", flush=True)
        del model, compact

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("peak RSS of this process:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "MB")


if __name__ == "__main__":
    main()
