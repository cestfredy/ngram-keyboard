"""Copy what the web demo needs (and nothing else) into a folder ready to publish.

Usage: python keyboard/build_site.py OUT_DIR
Includes the app, the manifest, the icons and the two models; leaves out tests and fixtures.
Adds `.nojekyll` so GitHub Pages serves the files as they are.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["index.html", "sw.js", "manifest.webmanifest"]
DIRS = ["css", "js", "icons"]
MODELS = ["ngram-light.ngm", "ngram-standard.ngm"]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(HERE / name, out / name)
    for name in DIRS:
        shutil.copytree(HERE / name, out / name, dirs_exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    for name in MODELS:
        shutil.copy2(HERE / "models" / name, out / "models" / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"site written to {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
