// Turns the text typed so far into what the model needs: the words of the current sentence and the
// letters of the word being typed. Mirrors ngram/src/ngramlm/text.py (same cleaning, same tokens).

import { NUM, UNK } from "./model.js";

const TOKEN = /aujourd'hui|\p{L}+'(?=\p{L})|\p{L}+(?:-\p{L}+)*|\d+(?:[.,]\d+)*/gu;

/** Unicode NFC, apostrophes and spaces unified, invisible characters removed, lowercased. */
export function normalizeKeepSpaces(text) {
  return text
    .normalize("NFC")
    .replace(/[’‘ʼ]/g, "'")
    .replace(/[  ]/g, " ")
    .replace(/[​­]/g, "")
    .toLowerCase();
}

/** Same as the Python `normalize`: also collapses runs of whitespace and trims. */
export function normalize(text) {
  return normalizeKeepSpaces(text).split(/\s+/).filter(Boolean).join(" ");
}

/** Tokenise normalised text (digits become <num>). */
export function tokenize(text) {
  return (text.match(TOKEN) || []).map((t) => (/^\d/.test(t) ? NUM : t));
}

/**
 * @param {string} text everything typed so far
 * @param {(token: string) => boolean} knows whether a token is in the model's vocabulary
 * @returns {{history: string[], prefix: string}} words of the current sentence (unknown ones
 *   replaced by <unk>) and the partial word being typed (lowercase, "" after a space).
 */
export function contextFromText(text, knows) {
  let t = normalizeKeepSpaces(text);

  // only the current sentence counts
  let cut = 0;
  for (const m of t.matchAll(/[.!?…]+(?=\s|$)|\n/gu)) cut = m.index + m[0].length;
  t = t.slice(cut);

  // the word being typed (it may contain an apostrophe only inside "aujourd'hui")
  let prefix = "";
  let head = t;
  let m = t.match(/aujourd'\p{L}*$/u) || t.match(/\p{L}+(?:-\p{L}*)*$/u);
  if (m) {
    prefix = m[0];
    head = t.slice(0, m.index);
  }

  // words before it; a trailing "l'" is an elision token even though nothing follows it yet
  let tokens;
  const elision = head.match(/(\p{L}+)'$/u);
  if (elision) {
    tokens = tokenize(head.slice(0, head.length - elision[0].length));
    tokens.push(elision[1] + "'");
  } else {
    tokens = tokenize(head);
  }
  return { history: tokens.map((tok) => (knows(tok) ? tok : UNK)), prefix };
}
