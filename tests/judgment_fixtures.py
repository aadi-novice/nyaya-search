"""ParsedDocs for zone/chunk/link tests, built from short real excerpts.

The excerpts in tests/fixtures/judgments/ were extracted read-only with PyMuPDF
from the 2016, 2018 and 2025 tars, with running heads, page numbers and margin
letters removed (the parser's job). Pages in the position map are synthetic:
one printed page per 40 lines, from the SCR start page in the path.
"""

from __future__ import annotations

from pathlib import Path

from nyaya.contracts import Era, ParsedDoc

JUDGMENTS = Path(__file__).resolve().parent / "fixtures" / "judgments"

ERA_C = "era_c_2018_5_28_35"
ERA_C_TWO_OPINIONS = "era_c_2016_8_36_72"
ERA_D = "era_d_2025_4_1350_1372"


def position_map(text: str, first_page: int, lines_per_page: int = 40) -> list[tuple[int, int]]:
    out = []
    off = 0
    for i, line in enumerate(text.split("\n")):
        out.append((off, first_page + i // lines_per_page))
        off += len(line) + 1
    return out


def parsed_from_text(path: str, era: Era, text: str, lines_per_page: int = 40) -> ParsedDoc:
    first = int(path.split("_")[-2]) if path.split("_")[-2].isdigit() else 1
    pm = position_map(text, first, lines_per_page)
    return ParsedDoc(
        path=path,
        era=era,
        clean_text=text,
        position_map=pm,
        pages=max(p for _, p in pm) - first + 1,
        clean_text_version="ct1",
    )


def load(name: str) -> ParsedDoc:
    text = (JUDGMENTS / f"{name}.txt").read_text(encoding="utf-8")
    era_s, path = name.split("_", 2)[1], name.split("_", 2)[2]
    return parsed_from_text(path, Era(era_s), text)
