#!/usr/bin/env python3
"""Render the MSML642 full proposal Markdown as an exact three-page PDF source."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import textwrap


PROPOSAL_DIR = Path(__file__).resolve().parent
SOURCE = PROPOSAL_DIR / "full-proposal.md"
OUTPUT = PROPOSAL_DIR / "MSML642_NandanPrince_ChrisPedretti.ps"

spec = spec_from_file_location("proposal_renderer", PROPOSAL_DIR / "render-proposal-pdf.py")
renderer = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(renderer)

# Compact academic layout: US Letter, single-spaced, with readable 8.2-point body text.
renderer.BODY_SIZE = 8.2
renderer.BODY_LEADING = 9.25
renderer.BODY_CHARS = 113
renderer.TOP = 758
renderer.BOTTOM = 34


class ThreePageWriter(renderer.PSWriter):
    def finish_page(self) -> None:
        self.page_ends.append(self.y)
        self.rule(28)
        self.line(
            "MSML642 Project Proposal - Nandan Prince and Chris Pedretti",
            renderer.LEFT,
            17,
            "Helvetica",
            7.2,
            (0.25, 0.25, 0.25),
        )
        self.line(
            f"Page {self.page} of 3",
            520,
            17,
            "Helvetica",
            7.2,
            (0.25, 0.25, 0.25),
        )
        self.commands.append("showpage")

    def title(self, value: str) -> None:
        for line in textwrap.wrap(renderer.strip_markdown(value), width=63):
            self.line(line, renderer.LEFT, self.y, "Helvetica-Bold", 16.0, (0.07, 0.25, 0.38))
            self.y -= 18
        self.rule(self.y + 4)
        self.y -= 7

    def heading(self, value: str) -> None:
        self.y -= 2
        self.line(renderer.strip_markdown(value), renderer.LEFT, self.y, "Helvetica-Bold", 10.3, (0.07, 0.30, 0.47))
        self.y -= 12

    def subheading(self, value: str) -> None:
        self.line(renderer.strip_markdown(value), renderer.LEFT, self.y, "Helvetica-Bold", 8.8, (0.12, 0.12, 0.12))
        self.y -= 10.5


def render() -> None:
    writer = ThreePageWriter()
    writer.start_page()
    paragraph: list[str] = []
    table_rows: list[list[str]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            writer.paragraph(" ".join(paragraph), space_after=2.2)
            paragraph = []

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            writer.table(table_rows)
            table_rows = []

    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "<!-- PAGE BREAK -->":
            flush_paragraph()
            flush_table()
            writer.start_page()
            continue
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if raw.startswith("    "):
            flush_paragraph()
            flush_table()
            writer.line(renderer.ascii_text(line), renderer.LEFT + 22, writer.y, "Courier", 7.7, (0.15, 0.15, 0.15))
            writer.y -= 8.5
        elif line.startswith("# "):
            flush_paragraph()
            writer.title(line[2:])
        elif line.startswith("## "):
            flush_paragraph()
            flush_table()
            writer.heading(line[3:])
        elif line.startswith("### "):
            flush_paragraph()
            writer.subheading(line[4:])
        elif line.startswith("**") and line.endswith("**"):
            flush_paragraph()
            writer.subheading(line[2:-2])
        elif line.startswith("**") and "**" in line[2:]:
            flush_paragraph()
            writer.metadata(line)
        elif re.match(r"^\d+\. ", line):
            flush_paragraph()
            match = re.match(r"^(\d+)\.\s+(.*)$", line)
            assert match
            writer.bullet(match.group(2), match.group(1))
        elif line.startswith("- "):
            flush_paragraph()
            writer.bullet(line[2:])
        elif line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
                table_rows.append(cells)
        else:
            paragraph.append(line)

    flush_paragraph()
    flush_table()
    writer.finish_page()

    if writer.page != 3:
        raise SystemExit(f"Expected exactly 3 pages, generated {writer.page}")
    for page_number, end_y in enumerate(writer.page_ends, start=1):
        if end_y < renderer.BOTTOM:
            raise SystemExit(f"Page {page_number} overflowed by {renderer.BOTTOM - end_y:.1f} points")

    header = [
        "%!PS-Adobe-3.0",
        "%%Creator: MSML642 proposal renderer",
        "%%Pages: 3",
        f"%%BoundingBox: 0 0 {renderer.PAGE_WIDTH} {renderer.PAGE_HEIGHT}",
        "%%EndComments",
    ]
    OUTPUT.write_text("\n".join(header + writer.commands + ["%%EOF", ""]), encoding="ascii")
    cursors = ", ".join(f"p{i}={y:.1f}" for i, y in enumerate(writer.page_ends, start=1))
    print(f"Generated {OUTPUT} with 3 pages; final cursors: {cursors}")


if __name__ == "__main__":
    render()
