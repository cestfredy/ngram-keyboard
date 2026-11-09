"""Compact, read-only version of a trained `NgramModel`: sorted numpy arrays instead of dicts.

File format (``.npz``), everything needed to predict and nothing else:
  meta        int64  [order, bos_id, eos_id, unk_id (-1 if none)]
  discount    float64 [d]
  tokens      uint8   UTF-8 bytes of the token list joined by "\\n" (index = id)
  ids_k       uint16  shape (n_k, k): the n-grams of order k >= 2, sorted lexicographically
  counts_k    uint32  shape (n_k,): the count used by the smoothing for each n-gram
  unigrams    uint32  shape (V,): count of every id at order 1
Ids use 16 bits, so a model holds at most 65,535 distinct tokens.

At load time the n-grams are packed into one uint64 key each (16 bits per id), so that finding
a history or an n-gram is a binary search. Probabilities and suggestions are identical to
`NgramModel` (same formula, same tie-break by id).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .model import BOS
from .ngram import NgramModel

_MAX_TOKENS = 1 << 16


def export(model: NgramModel, path: str | Path, compressed: bool = False) -> None:
    """Write ``model`` (already pruned if wanted) to ``path`` in the format above."""
    if len(model.tokens) >= _MAX_TOKENS:
        raise ValueError(f"at most {_MAX_TOKENS - 1} tokens can be exported")
    unk = model.ids.get("<unk>", -1)
    arrays: dict[str, np.ndarray] = {
        "meta": np.array([model.order, model._bos, model.ids["</s>"], unk], dtype=np.int64),
        "discount": np.array([model.discount], dtype=np.float64),
        "tokens": np.frombuffer("\n".join(model.tokens).encode("utf-8"), dtype=np.uint8),
    }
    uni = np.zeros(len(model.tokens), dtype=np.uint32)
    for (w,), c in model.num[1].items():
        uni[w] = c
    arrays["unigrams"] = uni
    for k in range(2, model.order + 1):
        grams = sorted(model.num[k])
        arrays[f"ids_{k}"] = np.array(grams, dtype=np.uint16).reshape(len(grams), k)
        arrays[f"counts_{k}"] = np.array([model.num[k][g] for g in grams], dtype=np.uint32)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)


def _pack(ids: np.ndarray) -> np.ndarray:
    """(n, m) uint16 ids -> (n,) uint64 keys, first id in the highest bits."""
    key = np.zeros(ids.shape[0], dtype=np.uint64)
    for j in range(ids.shape[1]):
        key = (key << np.uint64(16)) | ids[:, j].astype(np.uint64)
    return key


class CompactNgramModel:
    def __init__(self, path: str | Path) -> None:
        with np.load(path) as z:
            meta = z["meta"]
            self.order, self._bos, self._eos, unk = (int(x) for x in meta)
            self.discount = float(z["discount"][0])
            self.tokens = bytes(z["tokens"]).decode("utf-8").split("\n")
            self._uni = z["unigrams"].astype(np.float64)
            levels = {k: (z[f"ids_{k}"], z[f"counts_{k}"]) for k in range(2, self.order + 1)}
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self._excluded = {self._bos, self._eos} | ({unk} if unk >= 0 else set())
        self._unigram_total = float(self._uni.sum())

        # per level: sorted n-gram keys, their counts, and a table of histories
        self._keys: dict[int, np.ndarray] = {}
        self._cnt: dict[int, np.ndarray] = {}
        self._last: dict[int, np.ndarray] = {}
        self._hist_keys: dict[int, np.ndarray] = {}
        self._start: dict[int, np.ndarray] = {}
        self._den: dict[int, np.ndarray] = {}
        self._fol: dict[int, np.ndarray] = {}
        for k, (ids, counts) in levels.items():
            keys = _pack(ids)
            hk = keys >> np.uint64(16)
            starts = np.flatnonzero(np.r_[True, hk[1:] != hk[:-1]]) if len(keys) else np.array([], dtype=np.int64)
            self._keys[k], self._cnt[k] = keys, counts.astype(np.float64)
            self._last[k] = ids[:, -1].astype(np.int64)
            self._hist_keys[k] = hk[starts]
            self._start[k] = np.r_[starts, len(keys)]
            self._den[k] = np.add.reduceat(self._cnt[k], starts) if len(keys) else np.array([])
            self._fol[k] = np.diff(self._start[k]).astype(np.float64)

        allowed = np.array([i for i in range(len(self.tokens)) if i not in self._excluded], dtype=np.int64)
        order = np.lexsort((allowed, -self._uni[allowed]))
        ranked = allowed[order]
        self._rank1 = ranked[self._uni[ranked] > 0]

    # ------------------------------------------------------------------ #
    def _history_ids(self, history: Sequence[str]) -> list[int]:
        want = self.order - 1
        if want == 0:
            return []
        ids = [self.ids.get(t, -1) for t in history[-want:]]
        return [self._bos] * (want - len(ids)) + ids

    def _row(self, level: int, hist: list[int]) -> int:
        """Row of the history in the level's table, or -1 if unseen (or it has an unknown word)."""
        h = hist[len(hist) - (level - 1):]
        if min(h) < 0:
            return -1
        key = 0
        for i in h:
            key = (key << 16) | i
        hk = self._hist_keys[level]
        r = int(np.searchsorted(hk, np.uint64(key)))
        return r if r < len(hk) and int(hk[r]) == key else -1

    def _scores(self, cands: np.ndarray, hist: list[int]) -> np.ndarray:
        d = self.discount
        p = self._uni[cands] / self._unigram_total
        for k in range(2, self.order + 1):
            r = self._row(k, hist)
            if r < 0:
                continue
            total, fol = self._den[k][r], self._fol[k][r]
            hkey = int(self._hist_keys[k][r])
            q = (np.uint64(hkey) << np.uint64(16)) | cands.astype(np.uint64)
            keys = self._keys[k]
            idx = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
            c = np.where(keys[idx] == q, self._cnt[k][idx], 0.0)
            p = np.maximum(c - d, 0.0) / total + d * fol / total * p
        return p

    def prob(self, word: str, history: Sequence[str]) -> float:
        w = self.ids.get(word, -1)
        if w < 0:
            return 0.0
        return float(self._scores(np.array([w], dtype=np.int64), self._history_ids(history))[0])

    def suggest(self, history: Sequence[str], k: int = 3) -> list[str]:
        hist = self._history_ids(history)
        parts = []
        for level in range(2, self.order + 1):
            r = self._row(level, hist)
            if r >= 0:
                s = self._start[level]
                parts.append(self._last[level][s[r]:s[r + 1]])
        cands = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
        if self._excluded and len(cands):
            cands = cands[~np.isin(cands, list(self._excluded))]
        # complete with the first k words of the unigram ranking that are not candidates yet
        head = self._rank1[: k + len(cands)]
        extra = head[~np.isin(head, cands)][:k]
        cands = np.concatenate([cands, extra]).astype(np.int64)
        if not len(cands):
            return []
        p = self._scores(cands, hist)
        order = np.lexsort((cands, -p))[:k]
        return [self.tokens[int(cands[i])] for i in order if p[i] > 0.0]
