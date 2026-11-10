// Keyboard click sounds, synthesized with the Web Audio API: no audio file, so it works offline and weighs nothing.
// Like on iOS there are three sounds: letters, delete, and modifier keys (shift, space, return, 123).
// Browsers only start audio after a user gesture, and iOS also mutes Web Audio when the silent switch is on.

// Each click = a short pitched "tock" (a sine that falls a little) + a very short high-passed noise "tick".
// Levels give a peak of about 0.55 to 0.7 of full scale, around -19 dBFS on average over the click
// (checked in tests/e2e.py; the first version peaked at 0.15 and was inaudible on a phone speaker).
export const KINDS = {
  letter: { body: 1800, bodyGain: 0.55, decay: 0.045, noiseHP: 2500, noiseGain: 0.35, noiseDecay: 0.012 },
  delete: { body: 1100, bodyGain: 0.6, decay: 0.06, noiseHP: 2000, noiseGain: 0.3, noiseDecay: 0.014 },
  modifier: { body: 1400, bodyGain: 0.58, decay: 0.055, noiseHP: 2200, noiseGain: 0.3, noiseDecay: 0.013 },
};

export function makeNoise(actx) {
  const buf = actx.createBuffer(1, Math.ceil(actx.sampleRate * 0.05), actx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  return buf;
}

// Schedules one click on any audio context (real or offline) starting at time `t`.
export function scheduleClick(actx, dest, noise, kind, t) {
  const k = KINDS[kind] ?? KINDS.letter;

  const osc = actx.createOscillator();
  osc.frequency.setValueAtTime(k.body, t);
  osc.frequency.exponentialRampToValueAtTime(k.body * 0.7, t + k.decay);
  const og = actx.createGain();
  og.gain.setValueAtTime(0.0001, t);
  og.gain.linearRampToValueAtTime(k.bodyGain, t + 0.001); // 1 ms attack: no pop
  og.gain.exponentialRampToValueAtTime(0.0001, t + k.decay);
  osc.connect(og).connect(dest);
  osc.start(t);
  osc.stop(t + k.decay + 0.01);

  const src = actx.createBufferSource();
  src.buffer = noise;
  const hp = actx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = k.noiseHP;
  const ng = actx.createGain();
  ng.gain.setValueAtTime(k.noiseGain, t);
  ng.gain.exponentialRampToValueAtTime(0.0001, t + k.noiseDecay);
  src.connect(hp).connect(ng).connect(dest);
  src.start(t);
  src.stop(t + k.noiseDecay + 0.01);
}

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

// Call from a user gesture (touch, click, key): creates the audio context once and wakes it up if it is
// not running (iOS starts it suspended, and sets it to "interrupted" after a call or a tab switch).
// iOS plays Web Audio as "ambient" sound: it follows the ringer volume and the silent switch.
// Declaring a "playback" session (or playing a silent <audio> element, for older iOS) makes the
// clicks follow the media volume instead, like any other sound.
let keepAlive = null;
function useMediaVolume() {
  try {
    if (navigator.audioSession) navigator.audioSession.type = "playback";
  } catch {
    /* unsupported */
  }
  if (keepAlive) {
    if (keepAlive.paused) keepAlive.play().catch(() => {});
    return;
  }
  try {
    const n = 800; // 0.1 s of silence, 8 kHz mono 8-bit
    const b = new Uint8Array(44 + n);
    const v = new DataView(b.buffer);
    const s = (o, x) => [...x].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    s(0, "RIFF"); v.setUint32(4, 36 + n, true); s(8, "WAVEfmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, 8000, true); v.setUint32(28, 8000, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true);
    s(36, "data"); v.setUint32(40, n, true); b.fill(128, 44);
    keepAlive = new Audio(URL.createObjectURL(new Blob([b], { type: "audio/wav" })));
    keepAlive.loop = true;
    keepAlive.setAttribute("playsinline", "");
    keepAlive.play().catch(() => {});
  } catch {
    keepAlive = null;
  }
}

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
    master.gain.value = 1;
    master.connect(ctx.destination);
    noise = makeNoise(ctx);
  }
  if (enabled) useMediaVolume();
  if (ctx.state !== "running") ctx.resume().catch(() => {});
}

export function playKey(kind = "letter") {
  if (!enabled || !ctx) return;
  if (ctx.state !== "running") {
    ctx.resume().catch(() => {});
    return;
  }
  scheduleClick(ctx, master, noise, kind, ctx.currentTime);
}
