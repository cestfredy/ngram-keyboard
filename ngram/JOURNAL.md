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
| 2026-11-08 | Tune top-k on a 5,000-sentence validation sample | Full validation costs 15+ min per bigram run in pure Python; perplexity stays on full validation |
| 2026-11-07 | Vocabulary from the train split only | The test set must not influence the model |

## Results

| Model | Vocab | Smoothing | Perplexity (valid) | Top-1 | Top-3 | Keystroke savings | Size | Latency |
|-------|-------|-----------|--------------------|-------|-------|-------------------|------|---------|
| Unigram baseline (always `de je est`) | 20,000 | none | n/a | 3.11 % | 7.55 % | n/a | ~0 | ~0 |
| Interpolated Kneser-Ney, n=2 (d=0.75) | 20,000 | KN | 81.0 | 14.85 % | 28.45 % | n/a | | 3.4 ms |
| Interpolated Kneser-Ney, n=3 (d=0.75) | 20,000 | KN | 48.1 | 22.24 % | 37.78 % | n/a | | 5.1 ms |
| Interpolated Kneser-Ney, n=4 (d=0.75) | 20,000 | KN | 42.2 | 25.71 % | 40.88 % | n/a | | 7.0 ms |
| Interpolated Kneser-Ney, n=5 (d=0.75) | 20,000 | KN | 42.5 | 26.46 % | 41.48 % | n/a | | 8.9 ms |

Next-word prediction with no letter typed, on the validation split (276,255 positions, `</s>` not evaluated, an out-of-vocabulary target counts as a miss). The baseline suggests the same three words everywhere, so its overall top-1 and top-3 are identical for 10k, 20k and 50k (`results/baseline_unigram.csv`); only the accuracy restricted to known words moves (top-3: 7.98 % at 10k, 7.77 % at 20k, 7.63 % at 50k). Any n-gram model has to beat **7.55 % top-3** to justify itself.

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

### Kneser-Ney n-grams on validation (`results/ngram_kn_valid.csv`)

Top-1/top-3 measured on a fixed random sample of 5,000 validation sentences (seed 42, about 38k positions, roughly ±0.3 point); perplexity on the full validation split. Latency is pure Python, one core, with two runs sharing the machine, so treat it as an upper bound. Test split untouched.

- Order: top-3 goes 7.55 % (baseline) to 28.5 % (2), 37.8 % (3), 40.9 % (4), 41.5 % (5). Gains shrink fast: 4 to 5 gives +0.6 point for +63 % n-grams and +1.9 ms. Perplexity stops improving at 4 (42.2 vs 42.5).
- Discount d at order 4: top-3 is flat for d from 0.6 to 0.9 (40.88 / 40.88 / 40.73), falls at 0.95 (40.35); perplexity is best near 0.75 to 0.9 (42.2 / 42.7). d=0.75 kept.
- Next: vocabulary size (10k/50k), then full-validation run of the chosen configuration, then pruning for size.

### Vocabulary size at order 4, d=0.75 (same 5,000-sentence sample)

| Vocabulary | Top-1 | Top-3 | n-gram types | Latency | Perplexity |
|-----------:|------:|------:|-------------:|--------:|-----------:|
| 10,000 | 25.08 % | 40.34 % | 4.40 M | 3.6 ms | 35.5 |
| 20,000 | 25.71 % | 40.88 % | 4.90 M | 7.0 ms | 42.2 |
| 50,000 | 25.92 % | 41.13 % | 5.34 M | 8.5 ms | 49.5 |

Perplexities are **not comparable** across vocabulary sizes: a smaller vocabulary turns more words into `<unk>`, which are easy to predict. Compare top-k only. From 10k to 50k the gain is +0.8 point of top-3 for +21 % n-grams; latency is also affected by the candidate set growing with the vocabulary (the 7.0 ms of the 20k run was measured with two runs sharing the machine). 20,000 stays the working value until pruning and a clean latency measurement decide.

### Full validation run, order 4 (`results/ngram_kn_valid_full.csv`)

All 276,255 positions, vocabulary 20,000, d=0.75, machine otherwise idle: top-1 **25.52 %**, top-3 **40.80 %**, perplexity 42.2, fit 16.4 s, **5.0 ms** per prediction (pure Python, 4.9 M n-gram entries). The 5,000-sentence sample estimate (40.88 %) was within 0.08 point. Not yet pruned or exported: size on disk is still to measure.
