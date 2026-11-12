"""Citation patterns, canonical citation strings and alias rows (safety-critical).

Canonical forms (design, Technical Plan step 6 and file 24):

    scr:2025:1:230        [2025] 1 S.C.R. 230   or   2025 (1) SCR 230
    scr:2006:s6:72        [2006] SUPP. 6 S.C.R. 72   (Supplement volume)
    insc:2025:31          2025 INSC 31   or   2025INSC31
    scc:2009:8:751        (2009) 8 SCC 751
    air:1999:3750         AIR 1999 SC 3750
    cnr:escr010004242025  ESCR010004242025

The Supplement year is the year printed in the citation (the design's example
"scr:2005:s6:72 for [2006] SUPP. 6 ..." pairs two different years; the printed
year is the one a reader types, and it matches the year in `S_` paths).

Only SCR, neutral and CNR aliases are built: the metadata has no SCC or AIR
fields. Aliases are built only for loaded paths, and an alias that would
resolve to more than one path is dropped, so a lookup never resolves to a
wrong or unloaded judgment.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

# Each pattern: (kind, regex, function from match to canonical string).
_NB = r"(?<![\dA-Za-z])"  # no digit or letter just before
_NA = r"(?![\dA-Za-z])"  # no digit or letter just after

_SCR_BRACKET = re.compile(
    _NB + r"\[(\d{4})\]\s*(SUPP\.?\s*)?(\d+)\s*S\.?\s*C\.?\s*R\.?\s*(\d+)" + _NA, re.IGNORECASE
)
_SCR_PAREN = re.compile(_NB + r"(\d{4})\s*\((\d+)\)\s*SCR\s*(\d+)" + _NA, re.IGNORECASE)
_NEUTRAL = re.compile(_NB + r"(\d{4})\s*INSC\s*(\d+)" + _NA, re.IGNORECASE)
_SCC = re.compile(_NB + r"\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)" + _NA, re.IGNORECASE)
_AIR = re.compile(_NB + r"AIR\s*(\d{4})\s*SC\s*(\d+)" + _NA, re.IGNORECASE)
_CNR = re.compile(r"\b[A-Z]{4}\d{12}\b", re.IGNORECASE)


def _scr_bracket(m: re.Match[str]) -> str:
    year, supp, vol, page = m.group(1), m.group(2), m.group(3), m.group(4)
    vol_s = f"s{int(vol)}" if supp else str(int(vol))
    return f"scr:{int(year)}:{vol_s}:{int(page)}"


def _scr_paren(m: re.Match[str]) -> str:
    return f"scr:{int(m.group(1))}:{int(m.group(2))}:{int(m.group(3))}"


def _neutral(m: re.Match[str]) -> str:
    return f"insc:{int(m.group(1))}:{int(m.group(2))}"


def _scc(m: re.Match[str]) -> str:
    return f"scc:{int(m.group(1))}:{int(m.group(2))}:{int(m.group(3))}"


def _air(m: re.Match[str]) -> str:
    return f"air:{int(m.group(1))}:{int(m.group(2))}"


def _cnr(m: re.Match[str]) -> str:
    return f"cnr:{m.group(0).lower()}"


PATTERNS: tuple[tuple[str, re.Pattern[str], Any], ...] = (
    ("scr", _SCR_BRACKET, _scr_bracket),
    ("scr", _SCR_PAREN, _scr_paren),
    ("neutral", _NEUTRAL, _neutral),
    ("scc", _SCC, _scc),
    ("air", _AIR, _air),
    ("cnr", _CNR, _cnr),
)

Found = tuple[tuple[int, int], str, str]


def find_citations(text: str) -> list[Found]:
    """Every citation in `text` as ((start, end), kind, canonical), in text order.

    Overlapping matches keep the one that starts first (the longer one on a tie).
    """
    cands: list[Found] = []
    for kind, rx, canon in PATTERNS:
        for m in rx.finditer(text):
            cands.append(((m.start(), m.end()), kind, canon(m)))
    cands.sort(key=lambda f: (f[0][0], -f[0][1]))
    out: list[Found] = []
    last_end = -1
    for f in cands:
        if f[0][0] >= last_end:
            out.append(f)
            last_end = f[0][1]
    return out


def canonical_citation(s: str) -> str | None:
    """The canonical string for a text that is exactly one citation, else None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    for _, rx, canon in PATTERNS:
        m = rx.fullmatch(s)
        if m:
            return canon(m)
    return None


_PATH_RE = re.compile(r"(S_)?(\d{4})_(\d+)_(\d+)_(\d+)")


def path_citation(path: str) -> str | None:
    """The SCR citation encoded in a judgment path: `2025_1_230_238` -> scr:2025:1:230."""
    m = _PATH_RE.fullmatch(path)
    if not m:
        return None
    supp, year, vol, page = m.group(1), m.group(2), m.group(3), m.group(4)
    vol_s = f"s{int(vol)}" if supp else str(int(vol))
    return f"scr:{int(year)}:{vol_s}:{int(page)}"


def _value(row: Mapping[str, Any], *keys: str) -> str | None:
    """The first non-empty string among `keys` (schema name first, then raw name)."""
    for k in keys:
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        v = str(v).strip()
        if v:
            return v
    return None


def _rows(manifest: Any) -> Iterable[Mapping[str, Any]]:
    if hasattr(manifest, "to_dict") and hasattr(manifest, "columns"):
        return manifest.to_dict("records")
    return manifest


def _candidate_rows(manifest: Any, loaded_paths: Iterable[str]) -> list[dict[str, str]]:
    loaded = set(loaded_paths)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(alias: str | None, kind: str, path: str) -> None:
        if alias is None:
            return
        key = (alias, kind, path)
        if key not in seen:
            seen.add(key)
            out.append({"alias_norm": alias, "kind": kind, "path": path})

    for row in _rows(manifest):
        path = _value(row, "path")
        if path is None or path not in loaded:
            continue
        scr_text = _value(row, "scr_citation", "citation")
        scr = canonical_citation(scr_text) if scr_text else None
        if scr is not None and not scr.startswith("scr:"):
            scr = None
        add(scr, "scr", path)
        add(path_citation(path), "scr", path)
        nc_text = _value(row, "neutral_citation", "nc_display")
        nc = canonical_citation(nc_text) if nc_text else None
        add(nc if nc is not None and nc.startswith("insc:") else None, "neutral", path)
        cnr_text = _value(row, "cnr")
        cnr = canonical_citation(cnr_text) if cnr_text else None
        add(cnr if cnr is not None and cnr.startswith("cnr:") else None, "cnr", path)
    return out


def alias_collisions(
    manifest: Any, loaded_paths: Iterable[str]
) -> dict[tuple[str, str], list[str]]:
    """Aliases that point to more than one loaded path: {(alias, kind): sorted paths}."""
    paths: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in _candidate_rows(manifest, loaded_paths):
        paths[(r["alias_norm"], r["kind"])].add(r["path"])
    return {k: sorted(v) for k, v in paths.items() if len(v) > 1}


def build_alias_rows(manifest: Any, loaded_paths: Iterable[str]) -> list[dict[str, str]]:
    """Rows {alias_norm, kind, path} for the `aliases` table, for loaded paths only.

    `manifest` is a DataFrame or an iterable of mappings with the schema columns
    `path`, `scr_citation`, `neutral_citation`, `cnr` (the raw metadata names
    `citation` and `nc_display` are accepted too). SCR aliases come from the
    citation and from the path (the file name is the SCR citation). Aliases
    that would resolve to more than one path are left out; see alias_collisions.
    """
    loaded = set(loaded_paths)
    rows = _candidate_rows(manifest, loaded)
    bad = set(alias_collisions(manifest, loaded))
    return [r for r in rows if (r["alias_norm"], r["kind"]) not in bad]
