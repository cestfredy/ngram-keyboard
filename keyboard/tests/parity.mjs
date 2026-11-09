// Check the JS reader against exact Python answers on the real model, and time it.
// Usage: node keyboard/tests/parity.mjs keyboard/models/ngram-light.ngm keyboard/models/parity_light.json
import { readFileSync } from "node:fs";
import { NgramModel } from "../js/model.js";

const [modelPath, casesPath] = process.argv.slice(2);
const bytes = readFileSync(modelPath);
const t0 = performance.now();
const model = new NgramModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const loadMs = performance.now() - t0;
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

let bad = 0;
const times = { "": [], prefixed: [] };
for (const c of cases) {
  const t = performance.now();
  const got = model.suggest(c.history, 3, c.prefix);
  (c.prefix ? times.prefixed : times[""]).push(performance.now() - t);
  if (JSON.stringify(got) !== JSON.stringify(c.expected)) {
    if (bad++ < 5) console.log("MISMATCH", JSON.stringify(c.history.slice(-3)), JSON.stringify(c.prefix), got, c.expected);
  }
}
const stat = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return `mean ${(a.reduce((x, y) => x + y, 0) / a.length).toFixed(3)} ms, p95 ${s[Math.floor(s.length * 0.95)].toFixed(3)} ms, max ${s[s.length - 1].toFixed(3)} ms`;
};
console.log(`${modelPath}: loaded in ${loadMs.toFixed(0)} ms, ${cases.length} cases, ${bad} mismatches`);
console.log("no prefix  :", stat(times[""]));
console.log("with prefix:", stat(times.prefixed));
process.exit(bad ? 1 : 0);
