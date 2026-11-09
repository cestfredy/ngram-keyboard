// Read-only n-gram model with interpolated Kneser-Ney smoothing, reading the .ngm file written by
// ngram/src/ngramlm/compact.py (export_web). Same formula and same tie-break (by token id) as the
// Python model, so both give the same suggestions. No dependency.

export const BOS = "<s>";
export const EOS = "</s>";
export const UNK = "<unk>";
export const NUM = "<num>";

export class NgramModel {
  /** @param {ArrayBuffer} buffer contents of a .ngm file */
  constructor(buffer) {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== "NGM1") throw new Error("Not an NGM1 model file");
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, hlen)));
    if (header.version !== 1) throw new Error(`Unsupported model version ${header.version}`);

    this.order = header.order;
    this.discount = header.discount;
    this.tokens = header.tokens;
    this.bos = header.bos;
    this.eos = header.eos;
    this.unk = header.unk;
    this.ids = new Map(this.tokens.map((t, i) => [t, i]));
    this.excluded = new Set([this.bos, this.eos, ...(this.unk >= 0 ? [this.unk] : [])]);
    const V = this.tokens.length;

    let pos = 8 + hlen;
    this.uni = Float64Array.from(new Uint32Array(buffer, pos, V));
    pos += 4 * V;
    this.unigramTotal = this.uni.reduce((a, b) => a + b, 0);

    this.levels = new Array(this.order + 1).fill(null);
    this.levelSizes = { 1: V }; // entries per order (unigrams = vocabulary size)
    for (const { k, n } of header.levels) {
      const ids = new Uint16Array(buffer, pos, n * k);
      pos += 2 * n * k + ((4 - ((2 * n * k) % 4)) % 4);
      const cnt = new Uint32Array(buffer, pos, n);
      pos += 4 * n;
      this.levels[k] = this.#indexLevel(k, n, ids, cnt);
      this.levelSizes[k] = n;
    }

    // words ranked by unigram-level count (ties: smaller id first), used to complete candidates
    const allowed = [];
    for (let i = 0; i < V; i++) if (!this.excluded.has(i) && this.uni[i] > 0) allowed.push(i);
    allowed.sort((a, b) => this.uni[b] - this.uni[a] || a - b);
    this.rank1 = Int32Array.from(allowed);

    this.stamp = new Uint32Array(V); // scratch: "already a candidate" marks
    this.gen = 0;
  }

  /** Group the n-grams of one level by history (their first k-1 ids). */
  #indexLevel(k, n, ids, cnt) {
    const histKey = (i) => {
      let key = 0;
      for (let j = 0; j < k - 1; j++) key = key * 65536 + ids[i * k + j]; // < 2^48: exact in a double
      return key;
    };
    let H = 0;
    let prev = -1;
    for (let i = 0; i < n; i++) {
      const key = histKey(i);
      if (key !== prev) { H++; prev = key; }
    }
    const histKeys = new Float64Array(H);
    const start = new Uint32Array(H + 1);
    const den = new Float64Array(H);
    const last = new Uint16Array(n);
    let h = -1;
    prev = -1;
    for (let i = 0; i < n; i++) {
      const key = histKey(i);
      if (key !== prev) { h++; histKeys[h] = key; start[h] = i; prev = key; }
      den[h] += cnt[i];
      last[i] = ids[i * k + k - 1];
    }
    start[H] = n;
    return { k, n, histKeys, start, den, last, cnt };
  }

  /** Last order-1 words as ids, padded on the left with <s>; unknown words get -1. */
  historyIds(history) {
    const want = this.order - 1;
    const ids = history.slice(-want).map((t) => (this.ids.has(t) ? this.ids.get(t) : -1));
    while (ids.length < want) ids.unshift(this.bos);
    return ids;
  }

  /** Row of the history in level k's table, or -1 (unseen, or contains an unknown word). */
  #row(level, hist) {
    let key = 0;
    for (let j = hist.length - (level.k - 1); j < hist.length; j++) {
      if (hist[j] < 0) return -1;
      key = key * 65536 + hist[j];
    }
    const hk = level.histKeys;
    let lo = 0, hi = hk.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (hk[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo < hk.length && hk[lo] === key ? lo : -1;
  }

  /** Probabilities of the candidate ids (Int32Array/array), given history ids. */
  #scores(cands, hist) {
    const d = this.discount;
    const p = new Float64Array(cands.length);
    for (let i = 0; i < cands.length; i++) p[i] = this.uni[cands[i]] / this.unigramTotal;
    for (let k = 2; k <= this.order; k++) {
      const level = this.levels[k];
      const r = this.#row(level, hist);
      if (r < 0) continue;
      const total = level.den[r];
      const s = level.start[r], e = level.start[r + 1];
      const fol = e - s;
      for (let i = 0; i < cands.length; i++) {
        let lo = s, hi = e;
        const w = cands[i];
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (level.last[mid] < w) lo = mid + 1; else hi = mid;
        }
        const c = lo < e && level.last[lo] === w ? level.cnt[lo] : 0;
        p[i] = Math.max(c - d, 0) / total + (d * fol) / total * p[i];
      }
    }
    return p;
  }

  /** P(word | history); history is an array of tokens (may start with "<s>", any length). */
  prob(word, history) {
    if (!this.ids.has(word)) return 0;
    return this.#scores([this.ids.get(word)], this.historyIds(history))[0];
  }

  /**
   * The k most probable next words (never <s>, </s>, <unk>), optionally restricted to words that
   * start with `prefix`. Exact: with a prefix every matching word is scored; without one, the
   * candidates are the words seen after the history plus the k best unigram-level words.
   */
  suggest(history, k = 3, prefix = "") {
    const hist = this.historyIds(history);
    let cands;
    if (prefix) {
      let lo = 0, hi = this.tokens.length; // first token >= prefix (tokens are sorted)
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (this.tokens[mid] < prefix) lo = mid + 1; else hi = mid;
      }
      const list = [];
      for (let i = lo; i < this.tokens.length && this.tokens[i].startsWith(prefix); i++) {
        if (!this.excluded.has(i)) list.push(i);
      }
      cands = list;
    } else {
      const gen = ++this.gen;
      const list = [];
      for (let k2 = 2; k2 <= this.order; k2++) {
        const level = this.levels[k2];
        const r = this.#row(level, hist);
        if (r < 0) continue;
        for (let i = level.start[r]; i < level.start[r + 1]; i++) {
          const w = level.last[i];
          if (this.stamp[w] !== gen && !this.excluded.has(w)) { this.stamp[w] = gen; list.push(w); }
        }
      }
      let extra = 0;
      for (let i = 0; i < this.rank1.length && extra < k; i++) {
        const w = this.rank1[i];
        if (this.stamp[w] !== gen) { this.stamp[w] = gen; list.push(w); extra++; }
      }
      cands = list;
    }
    if (cands.length === 0) return [];
    const p = this.#scores(cands, hist);

    // top-k by (probability desc, id asc), single pass
    const bestI = [];
    for (let i = 0; i < cands.length; i++) {
      if (!(p[i] > 0)) continue;
      let pos = bestI.length;
      while (pos > 0) {
        const j = bestI[pos - 1];
        if (p[i] > p[j] || (p[i] === p[j] && cands[i] < cands[j])) pos--; else break;
      }
      if (pos < k) {
        bestI.splice(pos, 0, i);
        if (bestI.length > k) bestI.pop();
      }
    }
    return bestI.map((i) => this.tokens[cands[i]]);
  }
}
