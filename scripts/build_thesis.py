#!/usr/bin/env python3
"""Assemble the numbered thesis chapters into one THESIS.md.

    python3 scripts/build_thesis.py

The chapters in ``thesis/`` stay the editable source; this produces the single
document. Regenerate after editing a chapter — the two drift otherwise, and a
stale merged copy is worse than no merged copy.

Three things have to be rewritten on the way in, and each is easy to get wrong:

* **Heading depth.** Every chapter opens at ``#``. In one document exactly one
  ``#`` may exist, so chapter titles become ``##`` and everything below shifts
  down. Fenced code blocks are skipped: a ``# comment`` inside bash is not a
  heading, and demoting it corrupts the code.
* **Cross-references.** ``[x](06-metrics.md)`` has to become an anchor into this
  file. Anchors are generated with GitHub's slug rules from the *rewritten*
  heading, so they survive the renumbering.
* **Relative paths.** Chapters sit in ``thesis/`` and refer to ``../figures/``;
  the merged file sits at the repository root, where that is one level too high.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "thesis"
OUTPUT = ROOT / "THESIS.md"

TITLE = "Clinical Information Loss in German↔English Medical Machine Translation"
SUBTITLE = (
    "*Does German↔English medical translation need a specialised model?* — a "
    "benchmark of thirteen systems on radiology reports, and what it takes to "
    "measure the answer."
)

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,5})\s+(.*)$")
_CHAPTER_LINK = re.compile(r"\]\((0\d-[a-z0-9-]+)\.md(#[^)]*)?\)")
_PARENT_PATH = re.compile(r"\]\(\.\./")
_LABELLED_LINK = re.compile(r"\[`?(0\d-[a-z0-9-]+)\.md`?\]\(#([^)]+)\)")


def slug(text: str) -> str:
    """GitHub's heading-anchor rules: lowercase, drop punctuation, spaces to -."""
    text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code keeps its text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their label
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text.strip())


def main() -> int:
    files = sorted(CHAPTERS.glob("[0-9][0-9]-*.md"))
    if not files:
        raise SystemExit(f"no numbered chapters in {CHAPTERS}")

    # Pass 1: chapter stem -> the anchor its title will have in the merged file.
    anchors: dict[str, str] = {}
    display: dict[str, str] = {}
    titles: list[tuple[str, str]] = []
    for index, path in enumerate(files, start=1):
        first = next(
            (l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("# ")),
            f"# {path.stem}",
        )
        heading = f"{index}. {first[2:].strip()}"
        anchors[path.stem] = slug(heading)
        display[path.stem] = heading
        titles.append((heading, anchors[path.stem]))

    out: list[str] = [f"# {TITLE}", "", SUBTITLE, "", "---", "", "## Contents", ""]
    out += [f"{i}. [{h.split('. ', 1)[1]}](#{a})" for i, (h, a) in enumerate(titles, 1)]
    out += ["", "---", ""]

    for index, path in enumerate(files, start=1):
        in_fence = False
        seen_title = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
                out.append(line)
                continue
            if not in_fence:
                heading = _HEADING.match(line)
                if heading:
                    hashes, text = heading.groups()
                    if len(hashes) == 1 and not seen_title:
                        line = f"## {index}. {text}"
                        seen_title = True
                    else:
                        line = f"{'#' * min(6, len(hashes) + 1)} {text}"
                # chapter cross-reference -> in-document anchor
                line = _CHAPTER_LINK.sub(
                    lambda m: f"](#{anchors.get(m.group(1), m.group(1))})", line
                )
                # A link whose label is the chapter *filename* reads as a
                # dangling artefact once the target is an in-document anchor, so
                # the label becomes the chapter title too.
                line = _LABELLED_LINK.sub(
                    lambda m: f"[{display.get(m.group(1), m.group(1))}](#{m.group(2)})", line
                )
                # thesis/ -> repository root
                line = _PARENT_PATH.sub("](", line)
            out.append(line)
        out.append("")

    OUTPUT.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    words = len(OUTPUT.read_text(encoding="utf-8").split())
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"  {len(files)} chapters · {len(out)} lines · ~{words:,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
