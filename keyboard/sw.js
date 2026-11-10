// Service worker: keeps the app shell (HTML, CSS, JS) so the page opens with no network.
// Models are not handled here: the page stores them with the Cache API (see js/store.js).
const SHELL = "ngram-shell-v8";
const FILES = [
  "./", "index.html", "css/style.css", "js/app.js", "js/model.js", "js/context.js", "js/store.js", "js/sound.js",
  "manifest.webmanifest", "icons/icon.svg", "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png", "icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("ngram-shell-") && k !== SHELL).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== location.origin || url.pathname.includes("/models/")) return;
  event.respondWith(
    caches.match(event.request, { ignoreSearch: false }).then((hit) => hit || fetch(event.request)),
  );
});
