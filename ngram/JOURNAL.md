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
| 2026-11-07 | Split elisions (`l'` + `ami`), keep hyphenated words whole | Smaller vocabulary; 39.75 % of sentences have an elision, `est-ce` alone appears 13,334 times |
| 2026-11-07 | Deduplicate on token sequences, before the split | 8,597 duplicates vs 149 on raw lowercased text: punctuation hid them |
| 2026-11-07 | Vocabulary from the train split only | The test set must not influence the model |

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
- Digits: 0.69 % of sentences. Non-Latin letters: 43 sentences (Greek, Cyrillic, CJK, Hebrew, math symbols...).
- Domain bias: the name "Tom" appears in 5.99 % of sentences (43,744), "Marie" in 2.2 %. The corpus is written, translated sentences, not text messages. Expect "Tom" among suggestions; this is a known limitation to compare against OpenSubtitles later.
- Manual reading of 100 random sentences (seed 42): clean, grammatical, almost no slang, spaces before `?` and `!` are inconsistent.

### Cleaning rules (applied, `src/ngramlm/text.py`)

1. Normalise `’` `‘` `ʼ` to `'`; replace non-breaking spaces by a normal space; delete zero-width spaces and soft hyphens.
2. Lowercase.
3. Drop sentences with non-Latin letters, over 40 words, or fewer than 2 tokens.
4. Tokenisation: drop punctuation; split elisions after the apostrophe (`l'ami` becomes `l'` + `ami`); keep hyphenated words whole (`est-ce`, `peut-être`); keep `aujourd'hui` whole; map digit sequences to `<num>`.
5. Remove duplicates **on token sequences** (so `Tom ?` and `Tom!` count as one), before the split.

### Pipeline result (`src/prepare_data.py`, seed 42, `results/data_pipeline.csv`)

| Stage | Count |
|-------|-------|
| Raw sentences | 730,079 |
| Rejected by cleaning filters | 5,212 (1,451 one word, 1,715 one token such as `dépêche-toi !`, 2,003 over 40 words, 43 non-Latin) |
| Duplicates removed (token level) | 8,597 |
| Kept | 716,270 |
| Train / valid / test | 644,643 / 35,813 / 35,814 |
| Train tokens | 4,956,646 |
| Train distinct tokens | 78,166 (29,470 seen only once) |

Checks: the three splits share no sentence; running the script twice gives identical files (same sha256).

### Out-of-vocabulary rate on the validation split (`results/oov_by_vocab_size.csv`)

Vocabulary = the N most frequent TRAIN tokens; everything else becomes `<unk>`.

| Vocabulary | OOV tokens (valid) | OOV rate | Sentences with at least one OOV |
|-----------:|-------------------:|---------:|--------------------------------:|
| 10,000 | 15,010 | 5.43 % | 31.41 % |
| 20,000 | 8,071 | 2.92 % | 18.37 % |
| 50,000 | 2,841 | 1.03 % | 6.96 % |

Reading: this is the ceiling on what the model can ever propose. With 20,000 words, 2.9 % of the words to predict are out of reach whatever the model does. Going from 20k to 50k halves the loss but multiplies the model size; the choice is made later with the size/accuracy trade-off.

Observation: `tom` is the 22nd most frequent train token (39,275 occurrences) and appears in 2,220 of 35,813 validation sentences (6.2 %).
