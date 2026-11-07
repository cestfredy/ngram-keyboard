"""Normalisation, filtering and tokenisation of French sentences.

Rules (see JOURNAL.md):
1. Normalise apostrophes to ``'``, non-breaking spaces to a space, drop invisible characters.
2. Lowercase.
3. Reject sentences with non-Latin letters, fewer than 2 or more than 40 words.
4. Tokenise: drop punctuation, split elisions after the apostrophe (``l'ami`` -> ``l'`` ``ami``),
   keep hyphenated words whole (``est-ce``, ``peut-être``), keep ``aujourd'hui`` whole,
   map digit sequences to ``<num>``.
"""

from __future__ import annotations

import re
import unicodedata

NUM = "<num>"
MIN_WORDS = 2
MAX_WORDS = 40

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SPACES = str.maketrans({" ": " ", " ": " "})
_INVISIBLE = {"​", "­"}  # zero-width space, soft hyphen

_L = r"[^\W\d_]"  # any Unicode letter
_TOKEN = re.compile(
    r"aujourd'hui"  # kept whole: it is one word, not an elision
    rf"|{_L}+'(?={_L})"  # elision: l', j', qu', c', ...
    rf"|{_L}+(?:-{_L}+)*"  # word, possibly hyphenated: est-ce, a-t-il
    r"|\d+(?:[.,]\d+)*"  # number
)


def normalize(text: str) -> str:
    """Unicode NFC, apostrophes and spaces unified, invisible characters removed, lowercased."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_APOSTROPHES).translate(_SPACES)
    text = "".join(c for c in text if c not in _INVISIBLE)
    return " ".join(text.lower().split())


def has_non_latin_letters(text: str) -> bool:
    return any(c.isalpha() and not unicodedata.name(c, "").startswith("LATIN") for c in text)


def is_acceptable(text: str) -> bool:
    """Length and script filter, applied to normalised text."""
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS and not has_non_latin_letters(text)


def tokenize(text: str) -> list[str]:
    """Tokenise normalised text."""
    return [NUM if tok[0].isdigit() else tok for tok in _TOKEN.findall(text)]


def clean(raw: str) -> list[str] | None:
    """Full pipeline for one raw sentence. Returns tokens, or None if the sentence is rejected."""
    text = normalize(raw)
    if not is_acceptable(text):
        return None
    tokens = tokenize(text)
    return tokens if len(tokens) >= 2 else None
