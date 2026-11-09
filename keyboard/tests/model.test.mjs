// Run with: node --test keyboard/tests/   (Node 20+, no dependency)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { NgramModel, BOS } from "../js/model.js";
import { contextFromText, normalize, tokenize } from "../js/context.js";

const dir = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
const bytes = readFileSync(join(dir, "tiny.ngm"));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const model = new NgramModel(buffer);
const ref = JSON.parse(readFileSync(join(dir, "tiny.json"), "utf8"));

test("header is read", () => {
  assert.equal(model.order, ref.order);
  assert.equal(model.discount, ref.discount);
  assert.ok(model.tokens.includes("été"));
});

test("suggestions are identical to the Python model", () => {
  for (const c of ref.suggest) {
    assert.deepEqual(model.suggest(c.history, c.k, c.prefix), c.expected, JSON.stringify(c));
  }
});

test("probabilities are identical to the Python model", () => {
  for (const c of ref.prob) {
    const got = model.prob(c.word, c.history);
    assert.ok(Math.abs(got - c.p) <= 1e-12 * Math.max(c.p, 1e-300), `${c.word}: ${got} vs ${c.p}`);
  }
});

test("distribution sums to one", () => {
  for (const history of [[BOS], [BOS, "mot00"], ["mot01", "mot00", "mot02"], ["inconnu"]]) {
    const total = model.tokens.reduce((s, w) => s + model.prob(w, history), 0);
    assert.ok(Math.abs(total - 1) < 1e-9, `${history}: ${total}`);
  }
});

test("special tokens are never suggested", () => {
  for (const prefix of ["", "<", "<s", "</"]) {
    for (const w of model.suggest([BOS], 5, prefix)) assert.ok(!["<s>", "</s>", "<unk>"].includes(w), w);
  }
});

test("tokenisation is identical to the Python one", () => {
  for (const c of ref.tokens) assert.deepEqual(tokenize(normalize(c.text)), c.tokens, c.text);
});

const knows = (t) => model.ids.has(t);
const known = new Set(["je", "l'", "ami", "est-ce", "<num>", "aujourd'hui", "été"]);
const ctx = (text) => contextFromText(text, (t) => known.has(t));

test("context: partial word and finished words", () => {
  assert.deepEqual(ctx(""), { history: [], prefix: "" });
  assert.deepEqual(ctx("je "), { history: ["je"], prefix: "" });
  assert.deepEqual(ctx("Je ét"), { history: ["je"], prefix: "ét" });
  assert.deepEqual(ctx("je zzz "), { history: ["je", "<unk>"], prefix: "" });
});

test("context: elisions, hyphens, aujourd'hui, numbers", () => {
  assert.deepEqual(ctx("l'"), { history: ["l'"], prefix: "" });
  assert.deepEqual(ctx("l'am"), { history: ["l'"], prefix: "am" });
  assert.deepEqual(ctx("est-"), { history: [], prefix: "est-" });
  assert.deepEqual(ctx("est-ce "), { history: ["est-ce"], prefix: "" });
  assert.deepEqual(ctx("aujourd'"), { history: [], prefix: "aujourd'" });
  assert.deepEqual(ctx("aujourd'hu"), { history: [], prefix: "aujourd'hu" });
  assert.deepEqual(ctx("12 "), { history: ["<num>"], prefix: "" });
});

test("context: only the current sentence counts", () => {
  assert.deepEqual(ctx("je suis. l'ami"), { history: ["l'"], prefix: "ami" });
  assert.deepEqual(ctx("ça va ? je "), { history: ["je"], prefix: "" });
  assert.deepEqual(ctx("il a 3.5 ami "), { history: ["<unk>", "<unk>", "<num>", "ami"], prefix: "" });
  assert.deepEqual(ctx("fini.\n"), { history: [], prefix: "" });
});

test("typed text feeds the model end to end", () => {
  const { history, prefix } = contextFromText("mot00 mo", knows);
  const out = model.suggest(history, 3, prefix);
  assert.ok(out.length > 0 && out.every((w) => w.startsWith("mo")), out.join());
});
