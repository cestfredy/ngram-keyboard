import json
import random
import struct

import numpy as np

from ngramlm.compact import export_web
from ngramlm.ngram import NgramModel


def test_web_format_layout_round_trips(tmp_path):
    rng = random.Random(7)
    words = [f"w{i:02d}" for i in range(30)] + ["<unk>", "é"]
    corpus = [rng.choices(words, [1 / (i + 1) for i in range(32)], k=rng.randint(2, 8)) for _ in range(300)]
    model = NgramModel(order=3).fit(corpus).prune([1, 2])
    path = tmp_path / "m.ngm"
    export_web(model, path)
    data = path.read_bytes()

    assert data[:4] == b"NGM1"
    (hlen,) = struct.unpack("<I", data[4:8])
    assert (8 + hlen) % 4 == 0
    header = json.loads(data[8 : 8 + hlen])
    assert header["tokens"] == model.tokens and header["order"] == 3
    pos = 8 + hlen
    V = len(model.tokens)
    uni = np.frombuffer(data, dtype="<u4", count=V, offset=pos)
    assert {(i,): int(c) for i, c in enumerate(uni) if c} == model.num[1]
    pos += 4 * V
    for level in header["levels"]:
        k, n = level["k"], level["n"]
        ids = np.frombuffer(data, dtype="<u2", count=n * k, offset=pos).reshape(n, k)
        pos += 2 * n * k + (-(2 * n * k) % 4)
        counts = np.frombuffer(data, dtype="<u4", count=n, offset=pos)
        pos += 4 * n
        assert {tuple(int(x) for x in r): int(c) for r, c in zip(ids, counts)} == model.num[k]
    assert pos == len(data)
