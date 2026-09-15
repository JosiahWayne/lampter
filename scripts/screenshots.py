#!/usr/bin/env python3
"""Regenerate the README screenshots from the bundled demo dataset.

The images in ``docs/screenshots/`` are generated rather than captured by hand, for the
same reason ``--demo`` exists at all: a hand-captured terminal image is a snapshot of a
UI that no longer exists the moment a column moves, and producing a fresh one meant
pointing the tool at a live cluster to answer a question about *rendering*.

This drives the real application through Textual's test harness at a fixed size, so what
it writes is exactly what the dashboard draws -- same widgets, same layouts, same
colours -- with :class:`~lampter.demo.DemoTransport` supplying the data.

    python scripts/screenshots.py

Writes ``docs/screenshots/<view>.svg``, and ``<view>.png`` alongside it when
``rsvg-convert`` is on PATH. Both are committed: GitHub renders SVG in a README, but
PyPI does not, and the README is also the package's long description.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Both of these are read by Textual *at import time* and both decide whether the
# colours survive at all, so they have to be set before `lampter.ui` pulls Textual in:
#
# * `NO_COLOR` (set by many shells, CI runners and editor terminals) makes Textual
#   install its monochrome filter and drop every foreground colour while keeping bold.
#   The first version of this script produced convincingly grey screenshots for exactly
#   that reason -- including for the GPU column, whose whole meaning is its colour.
# * `TERM`/`COLORTERM` drive Rich's `auto` colour-system detection, and a `dumb`
#   terminal resolves to "no colour".
#
# Overriding a user's `NO_COLOR` is the wrong thing for the dashboard to do and the
# right thing for a script whose output is a documentation image.
os.environ.pop("NO_COLOR", None)
os.environ["TEXTUAL_COLOR_SYSTEM"] = "truecolor"
os.environ["TERM"] = "xterm-256color"
os.environ["COLORTERM"] = "truecolor"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lampter.demo import DemoTransport  # noqa: E402
from lampter.ui import SlurmMonitorApp  # noqa: E402

OUT = ROOT / "docs" / "screenshots"

#: 240 columns so that the summary line stays on one line as well as the table: at 215
#: the wide jobs layout fits but the summary wraps, which reads as a rendering bug. The
#: height is a little more than the table needs, so the image does not look cropped.
SIZE = (240, 22)

#: (view name, filename stem). The jobs view is the headline image.
VIEWS = (
    ("jobs", "jobs"),
    ("partitions", "capacity"),
    ("accounts", "accounts"),
)

#: `rsvg-convert` renders Textual's SVG faithfully and is a single small binary. Without
#: it the SVGs are still written, so a missing rasteriser degrades rather than fails.
RASTERISER = "rsvg-convert"


async def capture(view: str, path: Path, size: tuple[int, int]) -> str:
    app = SlurmMonitorApp(DemoTransport(), interval=60, store=None)
    async with app.run_test(size=size) as pilot:
        for _ in range(50):
            if app.snapshot is not None:
                break
            await pilot.pause()
        app.action_show_view(view)
        # Two pauses: one for the view switch to be laid out, one for the table to be
        # populated from the snapshot.
        await pilot.pause()
        await pilot.pause()
        svg = app.export_screenshot()
    path.write_text(svg, encoding="utf-8")
    return svg


def rasterise(svg_path: Path, png_path: Path) -> bool:
    binary = shutil.which(RASTERISER)
    if binary is None:
        return False
    # `-z 1` keeps the file at the terminal's own pixel grid; upscaling would only
    # inflate the repository.
    result = subprocess.run(
        [binary, "-z", "1", "-o", str(png_path), str(svg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  {RASTERISER} failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for view, stem in VIEWS:
        svg_path = OUT / f"{stem}.svg"
        await capture(view, svg_path, SIZE)
        png_path = OUT / f"{stem}.png"
        rasterised = rasterise(svg_path, png_path)
        size_kb = svg_path.stat().st_size // 1024
        made = f"{size_kb}K svg"
        if rasterised:
            made += f" + {png_path.stat().st_size // 1024}K png"
        print(f"{stem:<10} {made}")
    if shutil.which(RASTERISER) is None:
        print(f"note: {RASTERISER} not found, so only SVG was written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
