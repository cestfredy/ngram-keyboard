"""Explore a raw Tatoeba sentences file (id <TAB> lang <TAB> text) before any cleaning.

Read-only: prints statistics and writes a random sample for manual reading.
Usage: python explore_data.py data/raw/fra_sentences.tsv.bz2 [--sample out.txt] [--seed 42]
"""

from __future__ import annotations

import argparse
import bz2
import random
import statistics
import unicodedata
from collections import Counter

# Characters worth knowing about before choosing normalisation rules.
WATCHED = {
    "’": "apostrophe courbe ’",
    "'": "apostrophe droite '",
    "‘": "guillemet simple ouvrant ‘",
    "«": "guillemet français «",
    "»": "guillemet français »",
    "“": "guillemet double “",
    "”": "guillemet double ”",
    '"': 'guillemet droit "',
    "–": "tiret demi-cadratin –",
    "—": "tiret cadratin —",
    "-": "trait d'union -",
    "…": "points de suspension …",
    " ": "espace insécable",
    " ": "espace fine insécable",
    "\t": "tabulation",
}


def read_rows(path: str):
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip("\n").rstrip("\r").split("\t")


def pct(n: int, total: int) -> str:
    return f"{n:>9,} ({100 * n / total:5.2f} %)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", help="write 100 random sentences to this file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(read_rows(args.path))
    total = len(rows)
    print(f"Lignes lues                 : {total:,}")

    malformed = [r for r in rows if len(r) != 3]
    print(f"Lignes mal formées (≠3 col.) : {len(malformed):,}")
    rows = [r for r in rows if len(r) == 3]
    langs = Counter(r[1] for r in rows)
    print(f"Langues                     : {dict(langs)}")
    ids = [r[0] for r in rows]
    print(f"Identifiants uniques        : {len(set(ids)):,} / {len(ids):,}")

    texts = [r[2] for r in rows]
    n = len(texts)

    # ---- lengths
    words = [len(t.split()) for t in texts]
    chars = [len(t) for t in texts]

    def summary(name: str, xs: list[int]) -> None:
        xs_sorted = sorted(xs)
        p = lambda q: xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]
        print(
            f"{name:<28}: min {xs_sorted[0]}, médiane {statistics.median(xs):.0f}, "
            f"moyenne {statistics.mean(xs):.2f}, p95 {p(0.95)}, p99 {p(0.99)}, max {xs_sorted[-1]}"
        )

    print("\n== Longueurs ==")
    summary("Mots par phrase", words)
    summary("Caractères par phrase", chars)
    print(f"Phrases vides               : {pct(sum(1 for t in texts if not t.strip()), n)}")
    print(f"Phrases d'1 mot             : {pct(sum(1 for w in words if w == 1), n)}")
    print(f"Phrases de plus de 40 mots  : {pct(sum(1 for w in words if w > 40), n)}")
    print(f"Phrases de plus de 25 mots  : {pct(sum(1 for w in words if w > 25), n)}")

    # ---- duplicates
    print("\n== Doublons ==")
    exact = Counter(texts)
    dup_exact = sum(c - 1 for c in exact.values() if c > 1)
    print(f"Doublons exacts             : {pct(dup_exact, n)}")
    lowered = Counter(t.lower().strip() for t in texts)
    dup_lower = sum(c - 1 for c in lowered.values() if c > 1)
    print(f"Doublons après minuscules   : {pct(dup_lower, n)}")
    print("Phrases les plus répétées   :", [(t, c) for t, c in exact.most_common(5) if c > 1])

    # ---- characters
    print("\n== Caractères à surveiller (nombre de phrases concernées) ==")
    for ch, label in WATCHED.items():
        k = sum(1 for t in texts if ch in t)
        if k:
            print(f"{label:<34}: {pct(k, n)}")

    def has_digit(t: str) -> bool:
        return any(c.isdigit() for c in t)

    print(f"{'contient un chiffre':<34}: {pct(sum(1 for t in texts if has_digit(t)), n)}")
    print(f"{'commence par un tiret':<34}: {pct(sum(1 for t in texts if t.lstrip().startswith(('-', '–', '—'))), n)}")
    print(f"{'tout en majuscules':<34}: {pct(sum(1 for t in texts if t.isupper()), n)}")
    print(f"{'ne finit pas par . ! ? … » \"':<34}: {pct(sum(1 for t in texts if t.rstrip()[-1:] not in '.!?…»\"”'), n)}")

    # unusual scripts / symbols
    cat_counter: Counter[str] = Counter()
    odd_chars: Counter[str] = Counter()
    for t in texts:
        for c in set(t):
            cat_counter[unicodedata.category(c)] += 1
            if not (c.isalpha() or c.isdigit() or c.isspace()) and c not in WATCHED and c not in ".,;:!?()[]%&*/+=$€£@#":
                odd_chars[c] += 1
    latin_odd = Counter()
    for t in texts:
        for c in set(t):
            if c.isalpha() and "LATIN" not in unicodedata.name(c, "LATIN"):
                latin_odd[unicodedata.name(c, "?").split()[0]] += 1
    print(f"\nPhrases avec lettres non latines (par écriture) : {dict(latin_odd.most_common(8))}")
    print(f"Symboles rares les plus fréquents : {odd_chars.most_common(25)}")
    print(f"Caractères de contrôle           : {cat_counter.get('Cc', 0):,} phrases")

    # ---- elisions and hyphens (tokenisation planning)
    import re

    elision = re.compile(r"\b[A-Za-zÀ-ÿ]{1,2}['’][A-Za-zÀ-ÿ]")
    hyph = re.compile(r"\b[A-Za-zÀ-ÿ]+-[A-Za-zÀ-ÿ]+\b")
    print("\n== Tokenisation ==")
    print(f"Phrases avec élision (l'ami, j'ai…) : {pct(sum(1 for t in texts if elision.search(t)), n)}")
    print(f"Phrases avec mot composé à trait d'union : {pct(sum(1 for t in texts if hyph.search(t)), n)}")
    hy = Counter(m for t in texts for m in hyph.findall(t.lower()))
    print("Mots à trait d'union les plus fréquents :", hy.most_common(15))
    el = Counter(m.lower() for t in texts for m in re.findall(r"\b[A-Za-zÀ-ÿ]{1,2}['’]", t))
    print("Débuts d'élision les plus fréquents    :", el.most_common(12))

    # ---- sample
    if args.sample:
        rng = random.Random(args.seed)
        sample = rng.sample(texts, min(100, n))
        with open(args.sample, "w", encoding="utf-8") as f:
            f.write("\n".join(sample) + "\n")
        print(f"\nÉchantillon de {len(sample)} phrases (graine {args.seed}) écrit dans {args.sample}")


if __name__ == "__main__":
    main()
