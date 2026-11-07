# Journal

One entry per decision, with the number that justified it.

## Rules

- The test set is used **once**, at the very end.
- Vocabulary and counts are built on the **training split only**.
- Every metric goes in the results table below as soon as it is measured.

## Decisions

| Date | Decision | Why / evidence |
|------|----------|----------------|
| 2026-11-07 | Seed fixed in `config.toml` | Reproducible splits |
| 2026-11-07 | Store counts, compute probabilities on demand | Swap smoothing without recounting |

## Results

| Model | Vocab | Smoothing | Perplexity (valid) | Top-1 | Top-3 | Keystroke savings | Size | Latency |
|-------|-------|-----------|--------------------|-------|-------|-------------------|------|---------|
|       |       |           |                    |       |       |                   |      |         |

## Data observations

### Tatoeba, French (`fra_sentences.tsv.bz2`)

- File: 9,798,519 bytes, downloaded 2026-11-07, sha256 `ac4206ab4f175af8f7e5d05bf7307679a121526599c97b6f6d8a666ab4435ef0`.
- Format: `id<TAB>lang<TAB>text`, UTF-8, 0 malformed rows, all rows `fra`, 730,079 sentences with unique ids.
- Length: median 6 words, mean 7.47, p95 15, p99 26, max 246. 0.20 % one-word sentences, 0.27 % over 40 words.
- Duplicates: 0 exact; 149 (0.02 %) after lowercasing. They must be removed **before** the split, otherwise a sentence can sit in both train and test.
- Apostrophes: 37.85 % of sentences use the straight `'`, 2.90 % the curly `’`. Without normalisation, `l'ami` and `l’ami` are different words.
- Spaces: non-breaking space (2.52 %) and narrow non-breaking space (2.11 %), typically before `? ! : ;`. Zero-width spaces (18) and soft hyphens (7) also appear.
- Hyphens: 15.08 % of sentences contain a hyphen, mostly French inversions (`est-ce` 13,334, `as-tu`, `avez-vous`, `a-t-il`) and compounds (`peut-être`, `là-bas`).
- Elisions: 39.75 % of sentences contain one (`l'`, `n'`, `d'`, `j'`, `qu'`, `c'`, `s'`, `m'`, `t'`).
- Digits: 0.69 % of sentences. Non-Latin scripts: about 140 sentences (Cyrillic, Greek, CJK, Hebrew...).
- Domain bias: the name "Tom" appears in 5.99 % of sentences (43,744), "Marie" in 2.2 %. The corpus is written, translated sentences, not text messages. Expect "Tom" among suggestions; this is a known limitation to compare against OpenSubtitles later.
- Manual reading of 100 random sentences (seed 42): clean, grammatical, almost no slang, spaces before `?` and `!` are inconsistent.

### Cleaning rules (proposed, to validate)

1. Normalise `’` to `'`; replace non-breaking spaces by a normal space; delete zero-width spaces and soft hyphens.
2. Lowercase.
3. Drop sentences with non-Latin letters, over 40 words, or with fewer than 2 words.
4. Remove duplicates after lowercasing, before the split.
5. Tokenisation: split punctuation off and drop it; split elisions after the apostrophe (`l'ami` becomes `l'` + `ami`); keep hyphenated words as one token (`est-ce`, `peut-être`); map digit sequences to `<num>`.
