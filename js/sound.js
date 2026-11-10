// Keyboard click sounds, synthesized with the Web Audio API: no audio file, so it works offline and weighs nothing.
// Like on iOS there are three sounds: letters, delete, and modifier keys (shift, space, return, 123).
// Browsers only start audio after a user gesture, and iOS also mutes Web Audio when the silent switch is on.

const KINDS = {
  letter: { band: 3400, q: 1.1, tone: 1500, dur: 0.03, noise: 0.55, tonal: 0.18 },
  delete: { band: 2300, q: 1.0, tone: 900, dur: 0.04, noise: 0.6, tonal: 0.22 },
  modifier: { band: 1500, q: 1.0, tone: 700, dur: 0.04, noise: 0.55, tonal: 0.22 },
};

let ctx = null;
let master = null;
let noise = null;
let enabled = true;

export function setSoundEnabled(on) {
  enabled = Boolean(on);
}

export function soundKind(key) {
  if (key === "back") return "delete";
  return key.length === 1 ? "letter" : "modifier";
}

// Call from a user gesture (pointer, key): creates the audio context once and wakes it up if it was suspended.
export function unlockAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  if (!ctx) {
    try {
      ctx = new AC();
    } catch {
      return;
    }
    master = ctx.createGain();
    master.gain.value = 0.5;
    master.connect(ctx.destination);
    noise = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * 0.06), ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  if (ctx.state === "suspended") ctx.resume().catch(() => {});
}

export function playKey(kind = "letter") {
  if (!enabled || !ctx || ctx.state !== "running") return;
  const k = KINDS[kind] ?? KINDS.letter;
  const t = ctx.currentTime;

  // a short burst of filtered noise (the "tick") ...
  const src = ctx.createBufferSource();
  src.buffer = noise;
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = k.band;
  band.Q.value = k.q;
  const ng = ctx.createGain();
  ng.gain.setValueAtTime(k.noise, t);
  ng.gain.exponentialRampToValueAtTime(0.0001, t + k.dur);
  src.connect(band).connect(ng).connect(master);
  src.start(t);
  src.stop(t + k.dur + 0.01);

  // ... plus a tiny falling tone, which gives it a little body
  const osc = ctx.createOscillator();
  osc.frequency.setValueAtTime(k.tone, t);
  osc.frequency.exponentialRampToValueAtTime(k.tone * 0.6, t + k.dur);
  const og = ctx.createGain();
  og.gain.setValueAtTime(k.tonal, t);
  og.gain.exponentialRampToValueAtTime(0.0001, t + k.dur);
  osc.connect(og).connect(master);
  osc.start(t);
  osc.stop(t + k.dur + 0.01);
}
