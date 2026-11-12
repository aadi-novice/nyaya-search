"""Split the judgment zone into opinions (design, Technical Plan step 4; file 21).

`split_opinions(parsed, zones, judges) -> [Opinion]`.

*   An opinion starts at a line "NAME, J." (or "J", "CJI", "C.J."), optionally
    followed on the same line by the first paragraph ("DEEPAK GUPTA, J. 1. ...");
    in Era B by any first sentence ("JASWANT SINGH, J.-These appeals ..."),
    where NAME matches a judge on the bench line or in `judges` (metadata).
    When no names are known at all, any such line starts an opinion.
*   The type comes from the ordered phrase list in `opinion.phrases`, matched
    in the first `opinion.phrase_window_chars` characters of the opinion.
*   `majority` is never inferred from position: a sole opinion is `majority`;
    otherwise an opinion without a phrase is `majority` only if its author is
    the Era D starred ("*") author of the main opinion, and `unknown` otherwise.
    A sole opinion is `majority` even if a phrase matches (it has nothing to
    dissent from, and "dissent" often refers to a lower court).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from nyaya.config import Settings
from nyaya.contracts import Era, Opinion, OpinionType, ParsedDoc, Zones
from nyaya.text.normalize import normalize_judge
from nyaya.zone.zoner import _C_BENCH, default_settings

_START = re.compile(
    r"^[ \t]*([A-Z][A-Za-z. \t]{2,60}?),[ \t]*(CJI|C\.J|J)(\.?)[ \t]*:?"
    # then the line end, or the first paragraph, after an optional dash
    # ("A.K. SIKRI, J. - 1. ..."); "I." is OCR for "1." in scanned years
    r"(?=[ \t]*$|[ \t]+(?:[-\u2013\u2014][ \t]+)?(?:\d{1,3}|I)\.\s)",
    re.MULTILINE,
)
# Era B: the first sentence follows the author on the same line ("JASWANT
# SINGH, J.-These appeals", "Shah, J. One Sashi", "ARIJIT PASAYAT, J.: Leave
# granted."); OCR may drop the comma ("Hegcle J. This"), put a speck first,
# a quote before the J ("SHARMA, 'J."), a space or a comma after it ("PATHAK,
# J, This", "KH.NNA, J .. A decree"), or read "C.J." as "CJ".
_START_B = re.compile(
    r"^[ \t]*[^\w\n]{0,3}[ \t]*([A-Z][A-Za-z. \t]{2,60}?),?[ \t]*['\"]?(?<![A-Za-z])"
    r"(CJI|C\.J|CJ|J)(?![A-Za-z])[ \t]?([.,]?)[ \t]*[:.,]?[ \t]*"
    r"(?:[-\N{EN DASH}\N{EM DASH}]+|\.{2,})?[ \t]*(?=$|[A-Z0-9(\"'])",
    re.MULTILINE,
)
_NUMBERED_LINE = re.compile(r"^[ \t]*\d{1,3}(?:\.\d{1,2})?\.\s", re.MULTILINE)


def _bench_entries(text: str) -> list[tuple[str, bool]]:
    """(printed name, starred) for each judge on the first bench line in `text`."""
    m = _C_BENCH.search(text)
    if not m:
        return []
    inner = m.group(0).strip().lstrip("[").rstrip("]").strip()
    inner = re.sub(r",\s*\*", "*,", inner)
    inner = re.sub(r"[,\s]*\b(?:JJ|J)\.?\s*$", "", inner)
    out = []
    for part in re.split(r",|\band\b", inner, flags=re.IGNORECASE):
        starred = "*" in part
        part = " ".join(part.replace("*", " ").split())
        if normalize_judge(part):
            out.append((part, starred))
    return out


def bench_names(text: str) -> tuple[list[str], str | None]:
    """Normalised bench names from the first bench line in `text`, and the starred one."""
    entries = _bench_entries(text)
    names = [normalize_judge(p) for p, _ in entries]
    starred = next((normalize_judge(p) for p, s in entries if s), None)
    return names, starred


def names_match(a: str, b: str) -> bool:
    """Same judge after normalize_judge(), tolerating initials and OCR spacing.

    "J. CHELAMESWAR" = "JASTI CHELAMESWAR"; "D Y CHANDRACHUD" = "DR. D.Y.
    CHANDRACHUD"; "MARKANDEY KAT JU" = "MARKANDEY KATJU".
    """
    na, nb = normalize_judge(a), normalize_judge(b)
    if not na or not nb:
        return False
    if na == nb or na.replace(" ", "") == nb.replace(" ", ""):
        return True
    ta, tb = na.split(), nb.split()
    if ta[-1] != tb[-1]:
        return False
    fa, fb = ta[:-1], tb[:-1]
    if not fa or not fb:
        return True
    if len(fa) != len(fb):
        return False
    return all(
        x == y or (min(len(x), len(y)) == 1 and x[0] == y[0]) for x, y in zip(fa, fb, strict=True)
    )


OCR_NAME_MIN = 0.75  # SequenceMatcher ratio for an Era B author name read with OCR errors
OCR_NAME_MIN_LETTERS = 4


def ocr_name_match(name: str, known: str) -> bool:
    """Era B: the author's printed name against a bench or metadata name, with
    OCR errors ("BHAGWATI" / "P. N. BHAGWAT!", "NANAVATl" / "G.T. NANA VAT!",
    "FAzAL ALI" / "S. MURTAZA FAZAL ALI").

    The letters of `name` are compared with the same number of letters at the
    end of `known` (the surname comes last); names of fewer than
    OCR_NAME_MIN_LETTERS letters must match exactly (names_match).
    """
    a = re.sub(r"[^A-Z]", "", name.upper())
    b = re.sub(r"[^A-Z]", "", known.upper())
    if len(a) < OCR_NAME_MIN_LETTERS or not b:
        return False
    return SequenceMatcher(None, a, b[-len(a) :], autojunk=False).ratio() >= OCR_NAME_MIN


def _phrase_type(text: str, settings: Settings) -> OpinionType | None:
    window = text[: settings.opinion.phrase_window_chars].lower()
    window = " ".join(window.split())
    for type_name, phrases in settings.opinion.phrases.items():
        if any(p.lower() in window for p in phrases):
            return OpinionType(type_name)
    return None


def split_opinions(
    parsed: ParsedDoc,
    zones: Zones,
    judges: Iterable[str] = (),
    settings: Settings | None = None,
) -> list[Opinion]:
    """Opinions inside zones.judgment_span, in text order."""
    settings = settings or default_settings()
    text = parsed.clean_text
    j_start, j_end = zones.judgment_span

    entries = _bench_entries(text[zones.header_span[0] : zones.header_span[1]] or text)
    known = [p for p, _ in entries] + [j for j in judges if j and j.strip()]
    starred = [p for p, s in entries if s]

    starts: list[tuple[int, str, str]] = []  # (offset, display author, name)
    start_rx = _START_B if parsed.era is Era.B else _START
    for m in start_rx.finditer(text, j_start, j_end):
        name = " ".join(m.group(1).split())
        era_b = parsed.era is Era.B
        if known and not any(
            names_match(name, k) or (era_b and ocr_name_match(name, k)) for k in known
        ):
            continue
        title = {"J": "J.", "C.J": "C.J.", "CJ": "C.J.", "CJI": "CJI"}[m.group(2)]
        starts.append((m.start(), f"{name}, {title}", name))

    spans: list[tuple[int, int, str | None, str | None]] = []
    if not starts:
        author = starred[0] if parsed.era is Era.D and starred else None
        spans.append((j_start, j_end, author, author))
    else:
        first = starts[0][0]
        if _NUMBERED_LINE.search(text, j_start, first):
            spans.append((j_start, first, None, None))
        for i, (s, display, name) in enumerate(starts):
            e = starts[i + 1][0] if i + 1 < len(starts) else j_end
            spans.append((s, e, display, name))

    sole = len(spans) == 1
    out: list[Opinion] = []
    for s, e, display, name in spans:
        if sole:
            kind = OpinionType.MAJORITY
        else:
            kind = _phrase_type(text[s:e], settings)
            if kind is None:
                is_main = (
                    parsed.era is Era.D
                    and name is not None
                    and any(names_match(name, st) for st in starred)
                )
                kind = OpinionType.MAJORITY if is_main else OpinionType.UNKNOWN
        out.append(Opinion(author=display, type=kind, char_start=s, char_end=e))
    return out
