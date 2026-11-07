# ngram-keyboard

Lightweight French n-gram language model and predictive keyboard app, built to run offline in milliseconds.

> Work in progress. No results are published yet; they will be added here once measured.

## Task

Given the words already typed, suggest the 3 most likely next words, fully offline and in under 10 ms per prediction.

*(FR : à partir des mots déjà tapés, proposer les 3 mots suivants les plus probables, hors ligne et en moins de 10 ms par prédiction.)*

## Goals

- Train French n-gram models (unigram baseline, then bigram, trigram, 4-gram) with smoothing (Kneser-Ney).
- Evaluate them with perplexity, top-1 / top-3 accuracy, keystroke savings, model size and latency.
- Integrate the best model into a small keyboard simulation app.

## Data and licenses

- **Code:** MIT (see `LICENSE`).
- **Tatoeba:** French sentences from [Tatoeba](https://tatoeba.org), licensed CC-BY 2.0 FR. Attribution is required.
- **Raw and processed corpora are not redistributed** and are excluded from the repository (`.gitignore`).
