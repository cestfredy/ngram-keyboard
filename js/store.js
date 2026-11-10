// Download a model once, keep it in the browser's Cache API, and read it from there afterwards
// (works with no network). Recommended by Chrome for large model files.

const CACHE_NAME = "ngram-models-v1";

async function openCache() {
  try {
    return "caches" in globalThis ? await caches.open(CACHE_NAME) : null;
  } catch {
    return null; // private mode, blocked storage...
  }
}

/**
 * @param {string} url model URL; change it (e.g. ?v=2) to publish a new model version
 * @param {(fraction: number) => void} [onProgress] called while downloading
 * @returns {Promise<{buffer: ArrayBuffer, source: "cache" | "network", persisted: boolean}>}
 */
export async function loadModelBytes(url, onProgress) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return { buffer: await hit.arrayBuffer(), source: "cache", persisted: await isPersisted() };
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Modèle introuvable (${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total && onProgress) onProgress(Math.min(received / total, 1));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }

  let persisted = false;
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
      // ask the browser not to evict it under storage pressure (it may refuse)
      if (navigator.storage?.persist) persisted = await navigator.storage.persist();
    } catch { /* quota or blocked: the model still works for this session */ }
  }
  return { buffer: bytes.buffer, source: "network", persisted };
}

async function isPersisted() {
  try {
    return (await navigator.storage?.persisted?.()) ?? false;
  } catch {
    return false;
  }
}
