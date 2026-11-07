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

_To fill after exploring the corpus: sentence count, length, duplicates, odd characters._
