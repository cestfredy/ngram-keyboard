"""End-to-end check in a real (headless) Chromium: typing, suggestions, accents, offline, themes.

Needs the built models (ngram/src/make_web_model.py) and Playwright for Python.
Usage: python keyboard/tests/e2e.py [--shots DIR]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = set("0123456789-/:;()€&@\".,?!'")


def serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *a, **k: None  # type: ignore[method-assign]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/index.html"


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def key(page, k: str):
    return page.locator(f'.key[data-key="{k}"]')


def type_text(page, text: str) -> None:
    """Tap the on-screen keys, switching to the symbols layer when needed."""
    for ch in text:
        k = "space" if ch == " " else ch
        if ch in SYMBOLS and page.locator('.key[data-key="layer"]').inner_text() == "123":
            key(page, "layer").click()
        key(page, k).click()


def sugg(page) -> list[str]:
    return [b.inner_text() for b in page.locator(".sug").all()]


def field_text(page) -> str:
    return page.inner_text("#text")


def open_sheet(page) -> None:
    page.locator("#btn-settings").click()
    page.wait_for_selector("#sheet", state="visible")


def close_sheet(page) -> None:
    page.locator('.panel [data-close]').click()
    page.wait_for_selector("#sheet", state="hidden")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path)
    args = ap.parse_args()
    shots = args.shots
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    srv, url = serve()
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                  has_touch=True, is_mobile=True, color_scheme="light")
        page = ctx.new_page()
        # count the sounds the page starts (the audio itself cannot be heard in a headless test)
        page.add_init_script("window.__clicks = 0; const Orig = window.AudioContext;"
                             "window.AudioContext = class extends Orig { createBufferSource() { window.__clicks++; return super.createBufferSource(); } };")
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(url)
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        check(page.locator("html").get_attribute("data-theme") == "light", "starts in the light theme")
        check(sugg(page)[0] != "", f"suggestions at sentence start: {sugg(page)}")
        check(key(page, "a").inner_text() == "A", "first letter of a sentence is capitalised")
        if shots:
            page.screenshot(path=str(shots / "01-light-empty.png"))

        # typing, suggestions, taps
        type_text(page, "je ne sais ")
        s = sugg(page)
        check(s[0] == "pas", f"after 'je ne sais ' -> {s}")
        page.locator(".sug").nth(0).click()
        check(field_text(page) == "Je ne sais pas ", f"tap inserts the word: {field_text(page)!r}")
        type_text(page, "p")
        s = sugg(page)
        check(all(w.startswith("p") for w in s if w), f"prefix filters: {s}")

        # symbols layer + elision: l' then a word
        page.locator("#btn-settings").click()
        page.locator("#reset").click()
        type_text(page, "je voudrais l'")
        check(key(page, "a").is_visible() and page.locator('.key[data-key="layer"]').inner_text() == "123",
              "apostrophe returns to the letters layer")
        check(field_text(page) == "Je voudrais l'", f"elision typed: {field_text(page)!r}")
        check(len([w for w in sugg(page) if w]) == 3, f"suggestions after an elision: {sugg(page)}")

        # long press on e -> accent popup -> pick the 2nd accent (è)
        page.locator("#btn-settings").click()
        page.locator("#reset").click()
        type_text(page, "caf")
        box = key(page, "e").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.wait_for_timeout(550)
        check(page.locator("#accents").is_visible(), "long press opens the accent popup")
        if shots:
            page.screenshot(path=str(shots / "02-accents.png"))
        accs = page.locator("#accents .acc").all()
        check([a.inner_text() for a in accs] == list("éèêë"), "popup shows é è ê ë")
        page.mouse.up()
        check(field_text(page) == "Café", f"release picks the first accent: {field_text(page)!r}")
        page.locator("#btn-settings").click()
        page.locator("#reset").click()
        type_text(page, "p")
        box = key(page, "e").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.wait_for_timeout(550)
        second = page.locator("#accents .acc").nth(1).bounding_box()
        page.mouse.move(second["x"] + second["width"] / 2, second["y"] + second["height"] / 2, steps=4)
        page.mouse.up()
        check(field_text(page) == "Pè", f"sliding to the 2nd accent picks è: {field_text(page)!r}")

        # return sends the message
        page.locator("#btn-settings").click()
        page.locator("#reset").click()
        type_text(page, "je ne sais pas")
        key(page, "enter").click()
        check(page.locator(".msg").count() == 1 and page.inner_text(".msg") == "Je ne sais pas"
              and field_text(page) == "", "return sends the message and empties the field")
        if shots:
            page.screenshot(path=str(shots / "03-sent.png"))

        # theme button
        page.locator("#btn-theme").click()
        check(page.locator("html").get_attribute("data-theme") == "dark", "moon button switches to dark")
        if shots:
            page.screenshot(path=str(shots / "04-dark.png"))
        page.locator("#btn-theme").click()
        check(page.locator("html").get_attribute("data-theme") == "light", "sun button switches back")
        page.locator("#btn-theme").click()  # dark again; remembered after reload
        page.reload()
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        check(page.locator("html").get_attribute("data-theme") == "dark", "theme choice is remembered")

        # settings sheet
        open_sheet(page)
        facts = page.inner_text("#facts")
        check("4-grammes" in facts and "Kneser-Ney" in facts and "Top-3" in facts, "settings show the model facts")
        if shots:
            page.screenshot(path=str(shots / "05-settings-dark.png"))
        credit = page.locator(".credit a")
        check(credit.inner_text() == "@cestfredy" and credit.get_attribute("href") == "https://github.com/cestfredy"
              and "with love by" in page.inner_text(".credit"), "settings credit links to the GitHub profile")
        page.locator('[data-theme-pref="light"]').click()
        check(page.locator("html").get_attribute("data-theme") == "light", "theme segment works")
        page.locator('[data-theme-pref="auto"]').click()
        if shots:
            page.screenshot(path=str(shots / "06-settings-light.png"))
        page.locator('[data-model="standard"]').click()
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        check("standard" in page.evaluate("window.__ngramDemo.state.modelName") and "11,3" in page.inner_text("#facts"),
              "standard model loads from the settings")
        page.locator('[data-model="light"]').click()
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        close_sheet(page)

        # key preview bubble: shown while a character key is held, gone on release, absent for special keys
        box = key(page, "e").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        pop = page.locator("#pop").bounding_box()
        check(page.is_visible("#pop") and page.inner_text("#pop") in ("e", "E") and pop["y"] + pop["height"] <= box["y"],
              "key bubble appears above the pressed key")
        page.mouse.up()
        check(not page.is_visible("#pop"), "key bubble disappears on release")
        sb = key(page, "space").bounding_box()
        page.mouse.move(sb["x"] + 10, sb["y"] + 10)
        page.mouse.down()
        check(not page.is_visible("#pop"), "no bubble on the space key")
        page.mouse.up()

        # key sounds: on by default, one click per key press, silent once switched off in the settings
        for _ in range(3):  # the first touch unlocks the audio context, the next ones play
            box = key(page, "e").bounding_box()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        n_on = page.evaluate("window.__clicks")
        check(n_on >= 1, f"key presses play a click ({n_on} sounds)")
        open_sheet(page)
        page.locator('[data-sound="off"]').click()
        page.wait_for_timeout(100)
        n_before = page.evaluate("window.__clicks")
        close_sheet(page)
        box = key(page, "e").bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        check(page.evaluate("window.__clicks") == n_before, "no sound once switched off")
        open_sheet(page)
        page.locator('[data-sound="on"]').click()
        close_sheet(page)

        # click loudness, measured by rendering each sound offline: audible but never clipping
        levels = page.evaluate("""async () => {
            const m = await import('./js/sound.js'); const out = {};
            for (const name of Object.keys(m.KINDS)) {
              const sr = 44100, c = new OfflineAudioContext(1, Math.ceil(sr * 0.15), sr);
              const master = c.createGain(); master.connect(c.destination);
              m.scheduleClick(c, master, m.makeNoise(c), name, 0.01);
              const x = (await c.startRendering()).getChannelData(0);
              out[name] = Math.max(...x.map(Math.abs));
            } return out; }""")
        check(all(0.4 <= v <= 0.95 for v in levels.values()), f"click levels audible and not clipping {levels}")

        # pull the settings sheet down by its handle: a short pull springs back, a long one closes it
        open_sheet(page)
        page.wait_for_timeout(600)
        g = page.locator(".grabber").bounding_box()
        x, y = g["x"] + g["width"] / 2, g["y"] + 2
        page.mouse.move(x, y)
        page.mouse.down()
        for i in range(1, 5):  # slow pull (10 px every 100 ms), so it is not read as a flick
            page.mouse.move(x, y + 10 * i)
            page.wait_for_timeout(100)
        page.mouse.up()
        page.wait_for_timeout(700)
        check(page.evaluate("document.querySelector('#sheet').classList.contains('open')"), "short pull: the sheet springs back")
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 300, steps=6)
        page.mouse.up()
        page.wait_for_selector("#sheet", state="hidden", timeout=2000)
        check(True, "long pull closes the sheet")
        open_sheet(page)
        close_sheet(page)

        # send button: disabled while empty, sends the message like the return key
        page.evaluate("window.__ngramDemo.press('enter')")  # flush whatever is left in the field
        check(page.locator("#btn-send").is_disabled(), "send button disabled while the field is empty")
        n0 = page.evaluate("window.__ngramDemo.state.sent.length")
        page.evaluate("() => { const d = window.__ngramDemo; for (const ch of 'salut') d.typeLetter(ch); }")
        check(page.locator("#btn-send").is_enabled(), "send button enabled once there is text")
        page.locator("#btn-send").click()
        check(page.evaluate("window.__ngramDemo.state.sent.length") == n0 + 1 and field_text(page) == "",
              "send button sends the message and clears the field")
        check(page.locator("#btn-send").is_disabled(), "send button disabled again when the field is empty")

        # per-keystroke speed
        ms = page.evaluate("""() => {
            const d = window.__ngramDemo; const t = [];
            for (const ch of "il fait très beau aujourd'hui et je voudrais ") {
              const t0 = performance.now(); d.typeLetter(ch); t.push(performance.now() - t0);
            } return t; }""")
        ms.sort()
        print(f"INFO keystroke update (suggest + render): median {ms[len(ms)//2]:.2f} ms, max {ms[-1]:.2f} ms")
        check(ms[len(ms) // 2] < 10, "median keystroke update under 10 ms")

        # offline
        page.evaluate("navigator.serviceWorker.ready.then(() => true)")
        page.reload()
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        open_sheet(page)
        check("cache du navigateur" in page.inner_text("#facts") and "✓ prêt" in page.inner_text("#facts"),
              "second visit: model read from the cache, offline ready")
        close_sheet(page)
        ctx.set_offline(True)
        page.reload()
        page.wait_for_selector("#loading", state="hidden", timeout=20000)
        # "ù" is only reachable by long press: feed it through the page's own typing function
        page.evaluate("""() => { const d = window.__ngramDemo;
            for (const ch of "où") d.typeLetter(ch); d.press("space");
            for (const ch of "est") d.typeLetter(ch); d.press("space");
            for (const ch of "la") d.typeLetter(ch); d.press("space"); }""")
        s = sugg(page)
        check(s[0] == "gare", f"OFFLINE reload works, 'où est la ' -> {s}")
        if shots:
            page.screenshot(path=str(shots / "07-offline.png"))
        ctx.set_offline(False)
        browser.close()
    srv.shutdown()
    check(not errors, f"no console/page errors {errors}")
    print("ALL OK")


if __name__ == "__main__":
    main()
