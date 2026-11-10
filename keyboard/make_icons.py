"""Regenerate the maskable and iOS PNG icons from icons/icon.svg (needs rsvg-convert).
icon-192.png and icon-512.png (purpose "any") are kept as they are.

  icon-maskable-512.png              same drawing scaled to 72 % inside the safe zone, so Android masks never crop it
  apple-touch-icon.png (180 x 180)   home-screen icon on iOS (full bleed: iOS rounds the corners itself)

Usage: python keyboard/make_icons.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ICONS = Path(__file__).resolve().parent / "icons"
SVG = (ICONS / "icon.svg").read_text()


def render(svg: str, size: int, out: str) -> None:
    subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(ICONS / out)], input=svg.encode(), check=True)


def maskable(svg: str) -> str:
    """Keep the background full-bleed and shrink everything else around the centre."""
    body = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S).group(1)
    bg = re.search(r"<rect width=\"512\" height=\"512\"[^>]*/>", body).group(0)
    rest = body.replace(bg, "")
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">' + bg +
            '<g transform="translate(256 256) scale(.72) translate(-256 -258)">' + rest + "</g></svg>")


if __name__ == "__main__":
    render(maskable(SVG), 512, "icon-maskable-512.png")
    render(SVG, 180, "apple-touch-icon.png")
    print("icons written to", ICONS)
