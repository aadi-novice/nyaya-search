"""Quote snapping: replace a near-verbatim quote with the exact span it copies (safety-critical).

Design, "Citation check", decision 2026-11-18. When a citation's quote is not
found in its cited chunk, `snap_quote` looks for a span of that same chunk's
`text` that the quote copies with only small edits. If one is found, the
citation is shown with the span, cut from the chunk's raw text, so every quote
shown to a user is still an exact substring of the cited source. The LLM's
quote is never shown.

Text is split into word pieces (runs of letters, digits and marks, so
"No.2-Society" is "No", "2", "Society"), compared by their key:
`normalize_text()`, then letters and digits only. Punctuation and spacing
between pieces is ignored when matching but kept inside a span, which runs from
the first matched piece to the last.

Two tiers, tried in order:

1. **spacing**: the quote's keys, joined, equal the joined keys of a run of
   whole word pieces of the chunk. Only spacing and punctuation differ ("ﬁ led" for
   "filed", "no.2- society" for "No.2-Society", "this· Court"). Similarity 1.0.
2. **fuzzy**: the word-level `SequenceMatcher` ratio between the quote's keys
   and a window of chunk keys is at least `min_similarity`.

Both tiers are refused when:

* the quote contains an ellipsis (it stitches separate places together);
* the digit runs of quote and span differ (numbers, sections, dates);
* (fuzzy) a word inserted or deleted is a negation, or a replaced word differs
  from its partner only by a prefix or suffix ("lawful"/"unlawful",
  "arbitrable"/"nonarbitrable");
* (fuzzy) more than `max_edits` quote words are changed or missing.

Only the cited chunk is searched; a snap never moves a quote to another passage.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import regex

from nyaya.text.normalize import normalize_text

_TOKEN_RE = regex.compile(r"[\p{L}\p{N}\p{M}]+")  # a word piece: "No.2-Society" is 3
_NON_KEY_RE = regex.compile(r"[^\p{L}\p{N}]+")
_DIGITS_RE = regex.compile(r"\p{Nd}+")
_ELLIPSIS_RE = regex.compile(r"\.\s*\.\s*\.|\u2026")
# Closing punctuation right after the last piece is kept, so a span ends "Court."
_TRAILING_RE = regex.compile(r"[.,;:!?)\]\"'\u2019\u201d]*")

NEGATIONS = frozenset(
    {
        "not", "no", "never", "nor", "neither", "none", "nothing", "nobody",
        "cannot", "without", "unless", "except", "cant", "dont", "doesnt",
        "didnt", "isnt", "wasnt", "arent", "werent", "wont", "wouldnt",
        "shouldnt", "couldnt", "hasnt", "havent", "hadnt", "mustnt", "neednt",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Snap:
    """A span of the cited chunk shown in place of the LLM's quote."""

    span: str  # exact substring of the chunk's text
    similarity: float  # 1.0 for the spacing tier
    method: str  # "spacing" | "fuzzy"

    def to_dict(self) -> dict[str, object]:
        return {"span": self.span, "similarity": round(self.similarity, 4), "method": self.method}


def word_key(token: str) -> str:
    """`normalize_text()`, then letters and digits only."""
    return _NON_KEY_RE.sub("", normalize_text(token))


@dataclass(frozen=True)
class _Tokens:
    keys: list[str]  # non-empty keys only
    starts: list[int]  # char offset in the source text, per key
    ends: list[int]


def _tokens(text: str) -> _Tokens:
    keys: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for m in _TOKEN_RE.finditer(text):
        key = word_key(m.group())
        if key:
            keys.append(key)
            starts.append(m.start())
            ends.append(m.end())
    return _Tokens(keys, starts, ends)


def _digits(text: str) -> list[str]:
    return _DIGITS_RE.findall(normalize_text(text))


def _affix_pair(a: str, b: str) -> bool:
    """One word is the other with something added at the front or back."""
    if a == b:
        return False
    short, long_ = sorted((a, b), key=len)
    return bool(short) and (long_.startswith(short) or long_.endswith(short))


def _edits_safe(q: list[str], w: list[str], max_edits: int) -> bool:
    edits = 0
    sm = difflib.SequenceMatcher(None, q, w, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or "".join(q[i1:i2]) == "".join(w[j1:j2]):
            continue  # same letters, split differently ("No.2-Society" / "No.2- society")
        edits += max(i2 - i1, j2 - j1)
        changed = q[i1:i2] + w[j1:j2]
        if any(t in NEGATIONS for t in changed):
            return False
        if tag == "replace" and any(_affix_pair(a, b) for a in q[i1:i2] for b in w[j1:j2]):
            return False
    return edits <= max_edits


def _spacing_match(qkeys: list[str], ct: _Tokens) -> tuple[int, int] | None:
    """(first, last) chunk token indices whose joined keys equal the quote's."""
    target = "".join(qkeys)
    joined = "".join(ct.keys)
    offsets: dict[int, int] = {}  # char offset in `joined` -> token index starting there
    pos = 0
    for i, key in enumerate(ct.keys):
        offsets[pos] = i
        pos += len(key)
    ends = {}  # char offset just past a token -> token index
    pos = 0
    for i, key in enumerate(ct.keys):
        pos += len(key)
        ends[pos] = i
    start = joined.find(target)
    while start != -1:
        stop = start + len(target)
        if start in offsets and stop in ends:
            return offsets[start], ends[stop]
        start = joined.find(target, start + 1)
    return None


def _fuzzy_match(qkeys: list[str], ct: _Tokens, min_similarity: float) -> tuple[int, int, float]:
    """Best (first, last, ratio) window; ratio is 0.0 when none reaches `min_similarity`."""
    n = len(qkeys)
    slack = max(2, n // 5)
    best = (0, -1, 0.0)
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(qkeys)  # seq2 is cached by SequenceMatcher
    for length in range(max(1, n - slack), n + slack + 1):
        for i in range(0, max(1, len(ct.keys) - length + 1)):
            window = ct.keys[i : i + length]
            sm.set_seq1(window)
            floor = max(best[2], min_similarity)
            if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
                continue
            ratio = sm.ratio()
            if ratio > best[2]:
                blocks = [b for b in sm.get_matching_blocks() if b.size]
                first = i + blocks[0].a
                last = i + blocks[-1].a + blocks[-1].size - 1
                best = (first, last, ratio)
    if best[2] < min_similarity:
        return (0, -1, 0.0)
    # trim to matched tokens and re-score the trimmed window
    first, last, _ = best
    ratio = difflib.SequenceMatcher(None, ct.keys[first : last + 1], qkeys, autojunk=False).ratio()
    return (first, last, ratio) if ratio >= min_similarity else (0, -1, 0.0)


def best_window(quote: str, text: str) -> tuple[float, str]:
    """(ratio, span) of the window of `text` closest to `quote`, with no guards.

    A diagnostic for evaluation (how close is a quote to its passage?); it
    never decides what is shown.
    """
    qkeys = _tokens(quote).keys
    ct = _tokens(text)
    if not qkeys or not ct.keys:
        return 0.0, ""
    first, last, ratio = _fuzzy_match(qkeys, ct, 1e-9)
    if last < first:
        return 0.0, ""
    return ratio, text[ct.starts[first] : ct.ends[last]]


def snap_quote(
    quote: str,
    text: str,
    *,
    min_similarity: float,
    max_edits: int,
) -> Snap | None:
    """The exact span of `text` that `quote` copies with small edits, or None."""
    if _ELLIPSIS_RE.search(quote):
        return None
    qkeys = _tokens(quote).keys
    if not qkeys:
        return None
    ct = _tokens(text)
    if not ct.keys:
        return None

    method, similarity = "spacing", 1.0
    found = _spacing_match(qkeys, ct)
    if found is None:
        first, last, similarity = _fuzzy_match(qkeys, ct, min_similarity)
        if last < first or not _edits_safe(qkeys, ct.keys[first : last + 1], max_edits):
            return None
        method = "fuzzy"
    else:
        first, last = found

    end = _TRAILING_RE.match(text, ct.ends[last]).end()
    span = text[ct.starts[first] : end]
    if _digits(span) != _digits(quote):
        return None
    return Snap(span=span, similarity=similarity, method=method)
