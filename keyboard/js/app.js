import { NgramModel } from "./model.js";
import { contextFromText } from "./context.js";
import { loadModelBytes } from "./store.js";
import { playKey, setSoundEnabled, soundKind, unlockAudio } from "./sound.js";

// Figures measured on the validation split (see ngram/JOURNAL.md); shown in the settings panel.
const MODELS = {
  light: { url: "models/ngram-light.ngm?v=1", top3: "≈ 36,8 %", savings: "55,1 %" },
  standard: { url: "models/ngram-standard.ngm?v=1", top3: "≈ 38,8 %", savings: "56,2 %" },
};
const $ = (sel) => document.querySelector(sel);
const SENTENCE_END = /(^|[.!?…]\s*|\n\s*)$/u;
const VARIANTS = { a: "àâæä", e: "éèêë", i: "îï", o: "ôœ", u: "ùûü", c: "ç", y: "ÿ" };
const LONG_PRESS_MS = 380;

const LAYERS = {
  letters: [
    [..."azertyuiop"],
    [..."qsdfghjklm"],
    ["shift", ..."wxcvbn", "back"],
    ["layer", "space", "enter"],
  ],
  symbols: [
    [..."1234567890"],
    [..."-/:;()€&@\""],
    ["spacer", ".", ",", "?", "!", "'", "back"],
    ["layer", "space", "enter"],
  ],
};
const LABELS = {
  space: "espace", enter: "retour",
  shift: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3 2.5 13.5H8V20h8v-6.5h5.5z"/></svg>',
  back: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 5H9l-6.5 7L9 19h12a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1z"/><path d="m17 9-6 6M11 9l6 6"/></svg>',
};

const state = {
  text: "", presses: 0, taps: 0, shift: false, layer: "letters",
  sent: [], committedChars: 0,
  model: null, modelName: "light", source: null, loadMs: 0, bytes: 0, swReady: false,
  suggestions: [], ctx: { history: [], prefix: "" }, times: [],
  themePref: "auto",
};

// ---------------------------------------------------------------- model loading
async function loadModel(name) {
  const overlay = $("#loading");
  overlay.hidden = false;
  $("#loading-text").textContent = "Chargement du modèle…";
  $("#bar").style.width = "0%";
  state.model = null;
  try {
    const { buffer, source } = await loadModelBytes(MODELS[name].url, (f) => {
      $("#loading-text").textContent = `Téléchargement du modèle… ${Math.round(f * 100)} %`;
      $("#bar").style.width = `${Math.round(f * 100)}%`;
    });
    const t0 = performance.now();
    const model = new NgramModel(buffer);
    state.loadMs = performance.now() - t0;
    // warm-up: the first calls run before the JS engine has optimised the code and are not
    // representative of the speed the user will see
    for (let i = 0; i < 30; i++) { model.suggest([], 4); model.suggest(["je"], 4, "p"); }
    Object.assign(state, { model, modelName: name, source, bytes: buffer.byteLength, times: [] });
    overlay.hidden = true;
  } catch (err) {
    $("#loading-text").textContent = `Erreur : ${err.message}`;
    $("#bar").style.width = "0%";
    throw err;
  }
  refresh();
  render();
}

// ---------------------------------------------------------------- typing
const atSentenceStart = (text) => SENTENCE_END.test(text);
const isLetter = (ch) => /\p{L}/u.test(ch);

function typeLetter(ch) {
  const upper = state.shift || (isLetter(ch) && atSentenceStart(state.text));
  setText(state.text + (upper && isLetter(ch) ? ch.toUpperCase() : ch));
  state.presses++;
  state.shift = false;
  if (state.layer === "symbols" && ch === "'") { state.layer = "letters"; buildKeys(); } // elision: back to letters
  render();
}

function press(key) {
  if (!state.model) return;
  switch (key) {
    case "shift":
      state.shift = !state.shift;
      state.presses++;
      break;
    case "layer":
      state.layer = state.layer === "letters" ? "symbols" : "letters";
      state.presses++;
      buildKeys();
      break;
    case "back":
      setText([...state.text].slice(0, -1).join(""));
      state.presses++;
      break;
    case "space":
      setText(state.text + " ");
      state.presses++;
      if (state.layer === "symbols") { state.layer = "letters"; buildKeys(); }
      break;
    case "enter":
      send();
      break;
    case "spacer":
      return;
    default:
      typeLetter(key);
      return;
  }
  render();
}

function send() {
  state.presses++;
  const text = state.text.trim();
  if (text) {
    state.sent.push(text);
    state.committedChars += [...state.text].length + 1; // the text and the return key
  }
  state.shift = false;
  setText("");
}

function applySuggestion(i) {
  const word = state.suggestions[i];
  if (!word || !state.model) return;
  const { prefix } = state.ctx;
  const head = state.text.slice(0, state.text.length - prefix.length);
  const rawPrefix = state.text.slice(head.length);
  const capital = (rawPrefix && rawPrefix[0] !== rawPrefix[0].toLowerCase()) || (!prefix && atSentenceStart(head));
  const shown = capital ? word[0].toUpperCase() + word.slice(1) : word;
  setText(head + shown + (word.endsWith("'") ? "" : " "));
  state.presses++; // one tap inserts the word and its space
  state.taps++;
  state.shift = false;
  if (state.layer === "symbols") { state.layer = "letters"; buildKeys(); }
  render();
}

function setText(text) {
  state.text = text;
  refresh();
}

const TIMING_REPS = 8;

function refresh() {
  if (!state.model) return;
  const m = state.model;
  state.ctx = contextFromText(state.text, (t) => m.ids.has(t));
  // Browsers coarsen performance.now() (iOS Safari: 1 ms), so one call would read 0 ms.
  // We time a short batch of identical calls and divide; the result is the last call's.
  const t0 = performance.now();
  let words;
  for (let i = 0; i < TIMING_REPS; i++) words = m.suggest(state.ctx.history, 4, state.ctx.prefix);
  state.times.push((performance.now() - t0) / TIMING_REPS);
  // ask for one more word: a bare number token is useless as a suggestion
  state.suggestions = words.filter((w) => w !== "<num>").slice(0, 3);
  if (state.times.length > 300) state.times.shift();
}

function reset() {
  Object.assign(state, { text: "", presses: 0, taps: 0, shift: false, layer: "letters", sent: [], committedChars: 0, times: [] });
  refresh();
  buildKeys();
  render();
}

// ---------------------------------------------------------------- keyboard rendering
function buildKeys() {
  const keys = $("#keys");
  keys.textContent = "";
  LAYERS[state.layer].forEach((row) => {
    const div = document.createElement("div");
    div.className = "row";
    for (const key of row) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.key = key;
      const special = ["shift", "back", "layer", "enter"].includes(key);
      b.className = "key" + (special ? " special" : "") + (key === "space" ? " space" : "") + (key === "spacer" ? " spacer" : "");
      if (key === "layer") b.textContent = state.layer === "letters" ? "123" : "ABC";
      else if (LABELS[key] && key !== "space" && key !== "enter") b.innerHTML = LABELS[key];
      else b.textContent = LABELS[key] ?? key;
      b.setAttribute("aria-label", { back: "Effacer", shift: "Majuscule", enter: "Retour", space: "Espace", layer: "Chiffres et symboles" }[key] ?? key);
      if (key === "spacer") b.disabled = true;
      div.appendChild(b);
    }
    keys.appendChild(div);
  });
  paintCase();
}

function paintCase() {
  const up = state.shift || atSentenceStart(state.text);
  for (const b of document.querySelectorAll(".key")) {
    const k = b.dataset.key;
    if (k.length === 1 && isLetter(k)) b.textContent = up ? k.toUpperCase() : k;
    if (k === "shift") b.classList.toggle("on", state.shift);
  }
}

// pointer handling: letters act on touch-up (so a long press can open the accents), other keys
// act on touch-down; the click event only fires for keyboard activation (detail 0)
let down = null;
function wireKeys() {
  const keys = $("#keys");
  keys.addEventListener("pointerdown", (e) => {
    const b = e.target.closest(".key");
    if (!b || b.disabled) return;
    e.preventDefault();
    const key = b.dataset.key;
    down = { b, key, accents: false, timer: null, repeat: null };
    b.classList.add("down");
    unlockAudio();
    playKey(soundKind(key));
    if (key.length === 1) showPop(b); // character keys only (shift, space, return... have longer names)
    if (state.layer === "letters" && VARIANTS[key]) {
      down.timer = setTimeout(() => openAccents(b, key), LONG_PRESS_MS);
    } else if (key === "back") {
      press("back");
      down.timer = setTimeout(() => { down.repeat = setInterval(() => { playKey("delete"); press("back"); }, 70); }, 450);
    } else {
      press(key);
    }
  });
  const end = (e) => {
    if (!down) return;
    const d = down;
    down = null;
    clearTimeout(d.timer);
    clearInterval(d.repeat);
    d.b.classList.remove("down");
    hidePop();
    if (d.accents) {
      const sel = $("#accents .acc.sel");
      closeAccents();
      if (sel && e.type === "pointerup") typeLetter(sel.dataset.ch);
    } else if (state.layer === "letters" && VARIANTS[d.key] && e.type === "pointerup") {
      press(d.key);
    }
  };
  document.addEventListener("pointerup", end);
  // iOS only lets audio start from touchend / click (not pointerdown), so unlock on all of them
  for (const type of ["pointerup", "touchend", "click"]) document.addEventListener(type, unlockAudio, { capture: true });
  document.addEventListener("pointercancel", end);
  document.addEventListener("pointermove", (e) => {
    if (!down?.accents) return;
    const el = document.elementFromPoint(e.clientX, e.clientY)?.closest(".acc");
    if (el) for (const a of document.querySelectorAll("#accents .acc")) a.classList.toggle("sel", a === el);
  });
  keys.addEventListener("click", (e) => {
    const b = e.target.closest(".key");
    if (b && !b.disabled && e.detail === 0) press(b.dataset.key);
  });
  keys.addEventListener("contextmenu", (e) => e.preventDefault());

  const bar = $("#suggestions");
  bar.addEventListener("pointerdown", (e) => {
    const b = e.target.closest(".sug");
    if (!b) return;
    e.preventDefault();
    applySuggestion(Number(b.dataset.i));
  });
  bar.addEventListener("click", (e) => {
    const b = e.target.closest(".sug");
    if (b && e.detail === 0) applySuggestion(Number(b.dataset.i));
  });
}

function showPop(keyEl) {
  const pop = $("#pop");
  pop.textContent = keyEl.textContent;
  pop.hidden = false;
  const kb = $(".kbd").getBoundingClientRect();
  const r = keyEl.getBoundingClientRect();
  const w = r.width + 18;
  pop.style.width = `${w}px`;
  pop.style.left = `${Math.max(2, Math.min(kb.width - w - 2, r.left - kb.left + r.width / 2 - w / 2))}px`;
  pop.style.top = `${r.top - kb.top - pop.offsetHeight - 4}px`;
}

function hidePop() {
  $("#pop").hidden = true;
}

function openAccents(keyEl, key) {
  hidePop(); // the accent strip takes over from the bubble
  const up = state.shift || atSentenceStart(state.text);
  const box = $("#accents");
  box.textContent = "";
  [...VARIANTS[key]].forEach((ch, i) => {
    const d = document.createElement("div");
    const shown = up ? ch.toUpperCase() : ch;
    d.className = "acc" + (i === 0 ? " sel" : "");
    d.dataset.ch = shown;
    d.textContent = shown;
    box.appendChild(d);
  });
  box.hidden = false;
  const kb = $(".kbd").getBoundingClientRect();
  const r = keyEl.getBoundingClientRect();
  const w = box.offsetWidth;
  box.style.left = `${Math.max(4, Math.min(kb.width - w - 4, r.left - kb.left + r.width / 2 - w / 2))}px`;
  box.style.top = `${r.top - kb.top - box.offsetHeight - 6}px`;
  if (down) down.accents = true;
}

function closeAccents() {
  $("#accents").hidden = true;
}

// ---------------------------------------------------------------- rendering
function render() {
  const field = $("#field");
  $("#text").textContent = state.text;
  field.classList.toggle("filled", state.text.length > 0);
  $(".compose-box").classList.toggle("filled", state.text.length > 0);
  $("#btn-send").disabled = state.text.trim().length === 0;
  field.scrollTop = field.scrollHeight;

  const box = $("#messages");
  if (box.childElementCount !== state.sent.length) {
    box.textContent = "";
    for (const m of state.sent) {
      const d = document.createElement("div");
      d.className = "msg";
      d.textContent = m;
      box.appendChild(d);
    }
    box.scrollTop = box.scrollHeight;
  }

  const bar = $("#suggestions").children;
  for (let i = 0; i < 3; i++) bar[i].textContent = state.suggestions[i] ?? "";
  paintCase();
  renderSheet();
}

const fmt = (n) => n.toLocaleString("fr-FR");
function median(a) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

// Moves each segmented control's blue pill under its active button (no animation on the first placement).
function placeThumbs(animate) {
  for (const seg of document.querySelectorAll(".seg")) {
    const on = seg.querySelector("button.on"), thumb = seg.querySelector(".thumb");
    if (!on || !thumb || !on.offsetWidth) continue; // sheet hidden: measured when it opens
    if (!animate) thumb.classList.add("snap");
    thumb.style.setProperty("--x", `${on.offsetLeft}px`);
    thumb.style.setProperty("--w", `${on.offsetWidth}px`);
    if (!animate) { void thumb.offsetWidth; thumb.classList.remove("snap"); }
  }
}

// Milliseconds in, readable text out: microseconds below 0.1 ms, so a fast model never shows "0,00 ms".
function fmtDuration(ms) {
  if (ms < 0.1) return `${Math.max(1, Math.round(ms * 1000))} µs`;
  return `${ms.toFixed(ms < 10 ? 2 : 1).replace(".", ",")} ms`;
}

function renderSheet() {
  const chars = state.committedChars + [...state.text].length;
  $("#s-presses").textContent = state.presses;
  $("#s-chars").textContent = chars;
  $("#s-saved").textContent = chars ? `${Math.round(100 * (1 - state.presses / chars))} %` : "–";
  $("#s-taps").textContent = state.taps;
  const med = median(state.times);
  $("#s-ms").textContent = med == null ? "–" : fmtDuration(med);

  for (const b of document.querySelectorAll("[data-model]")) b.classList.toggle("on", b.dataset.model === state.modelName);
  for (const b of document.querySelectorAll("[data-theme-pref]")) b.classList.toggle("on", b.dataset.themePref === state.themePref);
  for (const b of document.querySelectorAll("[data-sound]")) b.classList.toggle("on", b.dataset.sound === state.sound);

  placeThumbs(sheetIsOpen());

  const m = state.model;
  const facts = $("#facts");
  facts.textContent = "";
  if (!m) return;
  const kept = Object.entries(m.levelSizes).filter(([k]) => k > 1).reduce((s, [, n]) => s + n, 0);
  const offline = state.swReady;
  const rows = [
    ["Type", `${m.order}-grammes · Kneser-Ney (d = ${String(m.discount).replace(".", ",")})`],
    ["Vocabulaire", `${fmt(m.tokens.length - 3)} mots`],
    ["N-grammes (ordres 2 à 4)", fmt(kept)],
    ["Fichier", `${(state.bytes / 1e6).toFixed(1).replace(".", ",")} Mo · lu en ${state.loadMs.toFixed(0)} ms`],
    ["Origine", state.source === "cache" ? "cache du navigateur" : "téléchargé (puis mis en cache)"],
    ["Hors-ligne", offline ? "✓ prêt" : "en préparation…", offline ? "ok" : ""],
    ["Top-3 (validation)", MODELS[state.modelName].top3],
    ["Touches économisées", MODELS[state.modelName].savings],
  ];
  for (const [k, v, cls] of rows) {
    const row = document.createElement("div"); // one flex row per fact: the value gets all the room the label leaves
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    if (cls) dd.className = cls;
    row.append(dt, dd);
    facts.append(row);
  }
}

// ---------------------------------------------------------------- theme, settings, wiring
function applyTheme() {
  const dark = state.themePref === "dark" || (state.themePref === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.querySelector('meta[name="theme-color"]').content = dark ? "#4a4a4a" : "#ffffff";
}

// ---------------------------------------------------------------- install (PWA)
// Chromium (Android, desktop) fires `beforeinstallprompt`; iOS has no install prompt, so we explain the Share menu.
const SHARE_ICON = '<svg class="share-i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 15V3M8 7l4-4 4 4M6 11H5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-1"/></svg>';
const install = { prompt: null, done: false, showSteps: false };

const isStandalone = () => matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
const isIos = () => /iphone|ipad|ipod/i.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const isFirefoxIos = () => /fxios/i.test(navigator.userAgent);

function renderInstall() {
  const box = $("#install"), btn = $("#btn-install"), note = $("#install-note");
  const ios = isIos() && !isFirefoxIos();
  const available = !isStandalone() && (install.done || install.prompt || ios);
  box.hidden = !available;
  if (!available) return;
  btn.hidden = install.done;
  if (install.done) {
    note.textContent = "✓ Installée. Ouvre-la depuis l'icône de ton écran d'accueil. Au premier lancement, garde la connexion : le modèle est copié une fois dans l'application.";
  } else if (install.prompt) {
    note.textContent = "S'ouvre dans sa propre fenêtre, avec son icône, et marche sans connexion.";
  } else if (install.showSteps) {
    note.innerHTML = `1. Touche <b>Partager</b> ${SHARE_ICON}.<br>2. Choisis <b>Sur l'écran d'accueil</b>, puis <b>Ajouter</b>.<br>Au premier lancement, garde la connexion : l'application garde sa propre copie du modèle.`;
  } else {
    note.textContent = "S'ouvre dans sa propre fenêtre, avec son icône, et marche sans connexion.";
  }
}

function wireInstall() {
  addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); install.prompt = e; renderInstall(); });
  addEventListener("appinstalled", () => { install.prompt = null; install.done = true; renderInstall(); });
  $("#btn-install").addEventListener("click", async () => {
    if (install.prompt) {
      const ev = install.prompt;
      install.prompt = null; // a prompt event can be used once
      ev.prompt();
      try { await ev.userChoice; } catch { /* dismissed */ }
    } else {
      install.showSteps = !install.showSteps;
    }
    renderInstall();
  });
  renderInstall();
}

function setThemePref(pref) {
  state.themePref = pref;
  try { localStorage.setItem("ngram-theme", pref); } catch { /* optional */ }
  applyTheme();
  renderSheet();
}

// The sheet slides in and out (CSS transitions on .open); `hidden` is only set once the exit is over.
let sheetTimer = 0;
const sheetIsOpen = () => $("#sheet").classList.contains("open");

function openSheet() {
  const sheet = $("#sheet");
  clearTimeout(sheetTimer);
  renderSheet();
  sheet.hidden = false;
  placeThumbs(false);
  void sheet.offsetWidth; // commit the closed state so the transition has a starting point
  sheet.classList.add("open");
}

function closeSheet() {
  const sheet = $("#sheet");
  if (!sheetIsOpen()) return;
  sheet.classList.remove("open");
  clearTimeout(sheetTimer);
  sheetTimer = setTimeout(() => { sheet.hidden = true; }, 260);
}

// Pull the sheet down by its handle: it follows the finger, then closes if dragged far or flicked, else springs back.
function wireSheetDrag() {
  const grip = $(".panel-grip"), panel = $(".panel"), backdrop = $(".backdrop");
  let drag = null;
  grip.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button") || !sheetIsOpen()) return;
    drag = { id: e.pointerId, y0: e.clientY, t0: performance.now(), dy: 0 };
    grip.setPointerCapture(e.pointerId);
    panel.style.transition = "none";
    backdrop.style.transition = "none";
  });
  grip.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    drag.dy = Math.max(0, e.clientY - drag.y0);
    panel.style.transform = `translateY(${drag.dy}px)`;
    backdrop.style.opacity = String(Math.max(0, 1 - drag.dy / panel.offsetHeight));
  });
  const end = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const { dy, t0 } = drag;
    drag = null;
    const speed = dy / Math.max(1, performance.now() - t0); // px per ms
    const dismiss = e.type === "pointerup" && (dy > panel.offsetHeight * 0.3 || (speed > 0.5 && dy > 24));
    // back to the stylesheet: from the dragged position, it either springs open again or slides out
    panel.style.transition = panel.style.transform = backdrop.style.transition = backdrop.style.opacity = "";
    if (dismiss) closeSheet();
  };
  grip.addEventListener("pointerup", end);
  grip.addEventListener("pointercancel", end);
}

function wireDesktopKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || !state.model || sheetIsOpen()) return;
    if (e.target.closest?.("button")) { if (e.key === " " || e.key === "Enter") return; }
    let handled = true;
    unlockAudio();
    if (e.key === "Backspace") { playKey("delete"); press("back"); }
    else if (e.key === "Enter") { playKey("modifier"); press("enter"); }
    else if (e.key === " ") { playKey("modifier"); press("space"); }
    else if ([...e.key].length === 1) { playKey("letter"); typeLetter(e.key === "’" ? "'" : e.key); }
    else handled = false;
    if (handled) e.preventDefault();
  });
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  try {
    await navigator.serviceWorker.register("sw.js");
    await navigator.serviceWorker.ready;
    state.swReady = true;
    renderSheet();
  } catch { /* file:// or blocked: the app still works online */ }
}

function savedPref(key, valid, fallback) {
  try {
    const v = localStorage.getItem(key);
    return valid(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

async function init() {
  state.themePref = savedPref("ngram-theme", (v) => ["auto", "light", "dark"].includes(v), "auto");
  state.modelName = savedPref("ngram-model", (v) => v in MODELS, "light");
  state.sound = savedPref("ngram-sound", (v) => ["on", "off"].includes(v), "on");
  setSoundEnabled(state.sound === "on");
  applyTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

  buildKeys();
  wireKeys();
  wireDesktopKeyboard();
  wireInstall();
  // send button: acts on touch-down like the return key; click is only for keyboard activation (detail 0)
  $("#btn-send").addEventListener("pointerdown", (e) => { e.preventDefault(); press("enter"); });
  $("#btn-send").addEventListener("click", (e) => { if (e.detail === 0) press("enter"); });
  wireSheetDrag();

  $("#btn-theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark";
    setThemePref(dark ? "light" : "dark");
  });
  $("#btn-settings").addEventListener("click", openSheet);
  addEventListener("resize", () => { if (sheetIsOpen()) placeThumbs(false); });
  $("#sheet").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeSheet(); });
  $("#seg-theme").addEventListener("click", (e) => {
    const b = e.target.closest("[data-theme-pref]");
    if (b) setThemePref(b.dataset.themePref);
  });
  $("#seg-sound").addEventListener("click", (e) => {
    const b = e.target.closest("[data-sound]");
    if (!b) return;
    state.sound = b.dataset.sound;
    try { localStorage.setItem("ngram-sound", state.sound); } catch { /* optional */ }
    setSoundEnabled(state.sound === "on");
    unlockAudio();
    playKey("letter"); // a sample click when switching on (nothing happens when switching off)
    renderSheet();
  });
  $("#seg-model").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-model]");
    if (!b || b.dataset.model === state.modelName) return;
    try { localStorage.setItem("ngram-model", b.dataset.model); } catch { /* optional */ }
    try { await loadModel(b.dataset.model); } catch { /* message already shown */ }
  });
  $("#reset").addEventListener("click", () => { reset(); closeSheet(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && sheetIsOpen()) closeSheet(); });

  window.__ngramDemo = { state, press, applySuggestion, typeLetter }; // handy for automated tests
  registerServiceWorker();
  try {
    await loadModel(state.modelName);
  } catch {
    return;
  }
}

init();
