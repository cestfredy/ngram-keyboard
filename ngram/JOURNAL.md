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
| 2026-11-10 | Test split evaluated once, on the models frozen after validation (`src/evaluate_test.py`, locked after the run) | Vocabulary 20,000, order 4, d=0.75, pruning `2,2,2` (standard) and `2,3,4` (light), nothing tuned on test |

## Results

| Model | Vocab | Smoothing | Perplexity (valid) | Top-1 | Top-3 | Keystroke savings | Size | Latency |
|-------|-------|-----------|--------------------|-------|-------|-------------------|------|---------|
| Unigram baseline (always `de je est`) | 20,000 | none | 487.8 | 3.11 % | 7.55 % | n/a | ~0 | ~0 |
| Interpolated Kneser-Ney, n=2 (d=0.75) | 20,000 | KN | 81.0 | 14.85 % | 28.45 % | n/a | | 3.4 ms |
| Interpolated Kneser-Ney, n=3 (d=0.75) | 20,000 | KN | 48.1 | 22.24 % | 37.78 % | n/a | | 5.1 ms |
| Interpolated Kneser-Ney, n=4 (d=0.75) | 20,000 | KN | 42.2 | 25.71 % | 40.88 % | n/a | | 7.0 ms |
| Interpolated Kneser-Ney, n=5 (d=0.75) | 20,000 | KN | 42.5 | 26.46 % | 41.48 % | n/a | | 8.9 ms |

Next-word prediction with no letter typed, on the validation split (276,255 positions, `</s>` not evaluated, an out-of-vocabulary target counts as a miss). The baseline suggests the same three words everywhere, so its overall top-1 and top-3 are identical for 10k, 20k and 50k (`results/baseline_unigram.csv`); only the accuracy restricted to known words moves (top-3: 7.98 % at 10k, 7.77 % at 20k, 7.63 % at 50k). Any n-gram model has to beat **7.55 % top-3** to justify itself. The perplexity of the baseline (487.8, full validation split, vocabulary 20,000) is that of the unigram model behind it, which gives a probability to every word (the fixed list `de je est` alone does not); it was added on 2026-11-10 and shows how much the context brings: 488 without it, 42 with 3 words of context.

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

### Count-threshold pruning, order 4, vocabulary 20,000, d=0.75 (`results/ngram_kn_pruning.csv`)

`min counts` = minimum count kept for orders 2, 3, 4 (counts as used by Kneser-Ney: continuation counts below the top order, raw at the top). Top-k on the same 5,000-sentence validation sample, perplexity on the full validation split. "Est. MB" is a naive raw size (4 bytes per word id + 4 per count, no compression, no index): a ruler, not the final file size.

| min counts | Entries | Est. MB | Top-1 | Top-3 | Perplexity | Latency |
|-----------:|--------:|--------:|------:|------:|-----------:|--------:|
| 1,1,1 (none) | 4,900,470 | 86.5 | 25.71 % | 40.88 % | 42.2 | 6.3 ms |
| 1,1,2 | 2,782,181 | 44.1 | 24.69 % | 39.95 % | 47.9 | 5.9 ms |
| 1,2,2 | 1,413,574 | 22.2 | 24.00 % | 38.86 % | 57.3 | 5.5 ms |
| 2,2,2 | 1,054,301 | 17.9 | 23.98 % | 38.79 % | 62.9 | 3.6 ms |
| 1,2,3 | 1,140,086 | 16.8 | 23.41 % | 38.14 % | 63.1 | 5.4 ms |
| 2,3,3 | 606,195 | 9.7 | 22.91 % | 37.41 % | 78.8 | 3.3 ms |
| 3,3,3 | 523,753 | 8.7 | 22.86 % | 37.25 % | 88.2 | 2.5 ms |
| 2,3,4 | 519,615 | 7.9 | 22.41 % | 36.80 % | 85.0 | 3.2 ms |

Reading:
- Removing 78 % of the entries (2,2,2) costs 2.1 points of top-3 (40.88 to 38.79). Removing 89 % (2,3,4) costs 4.1 points.
- Pruning bigrams seen once is almost free for top-3: 1,2,2 and 2,2,2 are equal within noise (38.86 vs 38.79) while 2,2,2 is 25 % smaller.
- Perplexity degrades much faster than top-3 (42 to 63 at 2,2,2): rare n-grams carry probability mass that the model must spread elsewhere, and perplexity is sensitive to that, while the ranking of the top 3 words barely moves. For a keyboard top-k is the metric that matters, but it is a known weakness of plain count cutoffs; a smarter criterion (entropy-based pruning) could do better at equal size.
- Latency improves up to about 2x because fewer candidates are scored (timings are noisy: two runs shared the machine).

### Compact export (`src/ngramlm/compact.py`, `src/export_model.py`, `results/ngram_kn_export.csv`)

The pruned model is written as sorted numpy arrays (16-bit word ids, 32-bit counts) in one `.npz` file and read back by `CompactNgramModel`, which finds n-grams by binary search over packed 64-bit keys and scores all candidates at once. Order 4, vocabulary 20,000, d=0.75, same 5,000-sentence validation sample.

| min counts | Entries | File | File, compressed | Load | Top-1 | Top-3 | Latency |
|-----------:|--------:|-----:|-----------------:|-----:|------:|------:|--------:|
| 2,2,2 | 1,054,301 | **11.2 MB** | 4.1 MB | 0.05 s | 23.98 % | 38.79 % | **0.42 ms** |
| 2,3,4 | 519,615 | **5.2 MB** | 2.0 MB | 0.03 s | 22.41 % | 36.80 % | **0.40 ms** |

- Accuracy is identical to the dict model (same top-1/top-3 as in the pruning table) and the suggestions are identical on 7,547 checked positions; the tests compare both models on random corpora.
- Latency drops about 10x (3.6 ms to 0.4 ms) because candidates are scored in one vectorised step instead of a Python loop. The full model (4.9 M entries, 1.2 GB as Python dictionaries) is now 11 MB on disk after pruning, so the size goal is met with a large margin.
- The "estimated raw MB" of the pruning table was a ruler; actual files are within about 40 % of it (11.2 vs 17.9 MB for 2,2,2).
- Models are written to `ngram/models/` (git-ignored, never committed).

### Keystroke savings (`src/ngramlm/keystrokes.py`, `src/evaluate_keystrokes.py`, `results/ngram_kn_keystrokes.csv`)

Definition: a simulated user types each word letter by letter; before every letter the keyboard shows the 3 most probable words starting with what is typed; as soon as the word appears it is tapped (1 key press, inserts the word and the space). Without prediction a word costs `len(word) + 1`. Out-of-vocabulary words are typed in full with their real length; `<num>` tokens are skipped (their length is lost after cleaning). Savings = 1 - keys with / keys without. Whole validation split, 275,943 words, 1,446,648 key presses without prediction.

| Model | Savings | Word proposed before any letter | Word proposed before its last letter | Letters typed before a hit | Latency |
|-------|--------:|-------------------:|--------------------:|------:|--------:|
| Unigram, no context (20k) | 38.32 % | 7.56 % | 86.93 % | 2.00 | 0.11 ms/word |
| 4-gram, pruned `2,2,2` (11.2 MB) | **56.18 %** | 38.70 % | 93.04 % | 1.03 | 0.51 ms/word |
| 4-gram, pruned `2,3,4` (5.2 MB) | **55.12 %** | 36.70 % | 92.85 % | 1.08 | 0.50 ms/word |

- A prediction bar that knows no context already saves 38 % on this corpus, because the most frequent words starting with one or two letters are short function words. The context brings the model to 56 %, i.e. +18 points, and halves the letters typed before a hit (2.0 to 1.0).
- The lighter model (`2,3,4`, half the size) loses only 1.1 point of savings, much less than the 4 points it lost on top-3 with no letter typed: once a letter or two is typed, the candidate set is small and context matters less.
- Caveats: Tatoeba is short, simple, translated sentences (median 6 words), so these figures are optimistic compared with real messages; they assume the user always sees and taps a correct suggestion at no cost. They must be re-measured on another corpus (OpenSubtitles) before being quoted as a general result.

### Web demo (`keyboard/`, plain HTML/CSS/JS)

The pruned model is exported to a flat binary format (`.ngm`, `export_web` in `compact.py`) read directly by `keyboard/js/model.js`: no dependency, same formula and tie-break as the Python model. The model is downloaded once, stored with the browser's Cache API, and a service worker keeps the app itself, so after the first visit everything works with no network.

- **Parity:** on 3,000 real validation positions per model (with and without a typed prefix), the JavaScript reader gives exactly the same 3 suggestions as the Python model (0 mismatches, standard and light); 10 unit tests also compare it with Python on a tiny model (450 suggestion cases, 151 probabilities, 18 tokenisation cases).
- **Speed (Node):** mean 0.2 ms per suggestion without prefix, 0.05 to 0.07 ms with a prefix; reading the 5.2 MB light model takes about 45 ms. In headless Chromium, a whole keystroke (suggest + screen update) takes about 0.6 ms at the median, 1.3 ms at most.
- **Offline:** tested in Chromium with the network cut and the page reloaded: the app opens, reads the model from the cache and predicts.
- **Light vs standard:** light = 5.2 MB (min counts 2,3,4), standard = 11.3 MB (2,2,2); the page lets the user switch.
- Only the two web-demo models are committed (`keyboard/models/ngram-light.ngm`, `ngram-standard.ngm`, 16.5 MB together) so the page can serve them; every other model and the parity files stay git-ignored. `ngram/src/make_web_model.py` rebuilds them.

## Final evaluation on the test split (2026-11-10, `results/test_final.csv`)

The test split (35,814 sentences, 275,535 word positions) was read once, by `src/evaluate_test.py`, after every choice was frozen. The script was first checked on 300 validation sentences, takes a lock before reading the test file and refuses to run again. Top-1/top-3 use **all** test positions (no sampling), `</s>` is not evaluated, an out-of-vocabulary target counts as a miss; perplexity includes `</s>`; keystroke savings use the same simulation as on validation (3 suggestions).

| Model | Entries | Perplexity | Top-1 | Top-3 | Keystroke savings |
|-------|--------:|-----------:|------:|------:|------------------:|
| Unigram baseline (`de je est`, prefix-filtered in the simulation) | | n/a | 3.08 % | 7.55 % | 38.45 % |
| 4-gram, no pruning | 4,900,470 | 41.5 | 25.75 % | 41.11 % | |
| 4-gram, standard (`2,2,2`, 11.3 MB) | 1,054,301 | 62.1 | 24.28 % | 39.05 % | 56.41 % |
| 4-gram, light (`2,3,4`, 5.2 MB) | 519,615 | 83.4 | 22.77 % | 36.99 % | 55.36 % |

- Accuracy and keystroke savings on test are 0.2 to 0.4 point above validation (top-3 on validation: 40.80 / 38.79 / 36.80; keystroke savings: 56.18 / 55.12) and perplexity is 1 to 2 % lower (41.5 / 62.1 / 83.4 against 42.2 / 63.0 / 85.0). That is the size of the sampling noise measured on validation (about 0.3 point), and the direction is the same for every model, so there is no sign of overfitting to the validation split.
- The ranking of the models and the cost of pruning are unchanged: the light model loses 4.1 points of top-3 against the unpruned model but only 1.05 point of keystroke savings against the standard one.
- The perplexity of the unigram baseline was not measured on the test split (487.8 on validation): the test split is kept to a single evaluation of the final models.
- The test split is now spent: no further tuning may use it. Any new variant (another corpus, another smoothing) needs a new held-out set.
