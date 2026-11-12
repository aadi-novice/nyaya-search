"""Split a ParsedDoc into zones (design, Technical Plan step 4; file 20).

`zone(parsed) -> Zones` finds the header, issue, editorial, references and
judgment zones with line-anchored patterns for each era (Era B: the Era C
patterns with OCR tolerance, see `_zone_era_b`), splits the editorial
zone into Held points with their "[Para N]" links, and collects Acts and
keywords:

*   Era D: the "List of Acts" and "List of Keywords" entries.
*   Eras B and C (author decision E1): every Act named in the catchwords, the
    editorial text before "HELD:".

`Zones.acts` holds Act names as printed, one per canonical name; use
`canonical_act()` (or `act_refs()`) for the normalised `refs.value_norm`.
"""

from __future__ import annotations

import itertools
import re
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache

from nyaya.config import REPO_ROOT, Settings, load_settings
from nyaya.contracts import Era, HeldPoint, ParsedDoc, Zone, Zones
from nyaya.text.normalize import normalize_text

_F = re.IGNORECASE | re.MULTILINE


@lru_cache(maxsize=1)
def default_settings() -> Settings:
    return load_settings(env_file=None)


# --- zone markers ----------------------------------------------------------------

# Era D
_D_ISSUE = re.compile(r"^[ \t]*Issue for Consideration[ \t]*$", _F)
_D_HEADNOTES = re.compile(r"^[ \t]*Headnotes\b", _F)
_D_CASE_LAW = re.compile(r"^[ \t]*Case Law Cited[ \t]*$", _F)
_D_ACTS = re.compile(r"^[ \t]*List of Acts[ \t]*$", _F)
_D_KEYWORDS = re.compile(r"^[ \t]*List of Keywords[ \t]*$", _F)
_D_ARISING = re.compile(r"^[ \t]*Case Arising From[ \t]*$", _F)
_D_JUDGMENT = re.compile(r"^[ \t]*Judgment\s*/\s*Order of the Supreme Court[ \t]*$", _F)
_D_HELD = re.compile(r"^[ \t]*Held\s*:", _F)

# Era C. The bench line may wrap; it is the first "[...J.]" / "[...JJ.]" block.
# Searched only before the first HELD (and in the first BENCH_WINDOW characters),
# so a running head "[DEEPAK GUPTA, J.]" deep in the judgment is never taken.
# "(" and ")" are OCR forms of the brackets in scanned years.
_C_BENCH = re.compile(
    r"[\[(][^\[\]()]{0,300}?(?:J\.|JJ\.|CJI\.?)\s*[\])J]?[ \t]*$", _F
)  # "JJ.J": OCR "]"
BENCH_WINDOW = 4000
# Fallback start of the judgment zone when no jurisdiction line is found.
_C_DELIVERED = re.compile(r"^[ \t]*The Judgments? of the Court (?:was|were) delivered by", _F)
# Upper case only: catchwords say "On appeal, held:" in lower case.
_C_HELD = re.compile(r"HELD\s*:")
# Single-point headnotes print "<catchwords> - Held: <holding>" with no upper-case
# HELD anywhere (about 10% of a 2010-2023 sample). Used only when no "HELD:"
# exists, so the inline "- Held:" inside multi-point catchwords is never a start.
_C_HELD_TITLE = re.compile(r"\bHeld\s*:")
_C_CASE_LAW = re.compile(r"^[ \t]*Case Law Reference[ \t]*$", _F)
_C_JURISDICTION = re.compile(
    # Upper case only (a Held line may start "criminal court has no jurisdiction"),
    # no \b after the first word (OCR drops spaces: "CIVILAPPELLATEJURISDICTION"),
    # and "l" or "1" for "I" (OCR: "ClVILAPPELLATE JURISDICTION").
    r"^[ \t]*(?:C[Il1]V[Il1]L|CR[Il1]M[Il1]NAL|OR[Il1]G[Il1]NAL|WR[Il1]T|APPELLATE)"
    r"[A-Z \t()/&,.\-Il1]*JUR[Il1]SD[A-Z]*",  # "JURISDICATION" (sic) is seen
    re.MULTILINE,
)
# Era C Held point numbers ("1.", "1.1", "2.3"); case-sensitive lookahead.
# Only at a line start (or straight after "HELD:", checked in _era_c_points): a
# number inside a line is a citation volume or a witness ("(2018) 2 SCC 357").
_C_POINT = re.compile(r"(?:^|(?<=\s))(\d{1,2}(?:\.\d{1,2})?)\.?\s+(?=[A-Z])")
_LINE_START = re.compile(r"(?:^|\n)[ \t]*$")
_PER = re.compile(r"^[ \t]*Per\b", re.IGNORECASE | re.MULTILINE)
FIRST_POINT_WINDOW = 300

# --- paragraph links -------------------------------------------------------------

# Design regex, plus an optional stray margin letter before the numbers
# ("[Para C 10)"), margin letters inside the list, "to" ranges ("[Paras 11 to
# 15)") and "(" for "[" (OCR in scanned years: "(Paras 5, 6]").
_PARA_LINK = re.compile(
    r"[\[(]\s*Paras?\s+(?:[A-H]\s+)?([0-9][0-9.,&\s\-\u2013andtoA-H]*?)\s*[\])]",
    re.IGNORECASE,
)
# Page-and-line pinpoints after a link: "[34-C-D]", "(67-G-H; 68-\nA-B]".
_PINPOINT_RUN = re.compile(r"(?:\s*[\[(]\s*\d[^\[\]()]{0,40}[\])])+")
_LABEL = re.compile(r"\d{1,3}(?:\.\d{1,2})?")
_MAX_RANGE = 50


def parse_para_list(inner: str) -> list[str]:
    """Paragraph labels from the text inside a "[Para ...]" link."""
    inner = re.sub(r"(?<![A-Za-z])[A-H](?![A-Za-z])", " ", inner)
    labels: list[str] = []
    inner = re.sub(r"\bto\b", "-", inner)
    for part in re.split(r",|&|\band\b", inner):
        part = part.strip()
        if not part:
            continue
        rng = re.fullmatch(r"(\S+?)\.?\s*[-\u2013]\s*(\S+?)\.?", part)
        if rng:
            a, b = rng.group(1), rng.group(2)
            if a.isdigit() and b.isdigit() and int(a) <= int(b) <= int(a) + _MAX_RANGE:
                labels.extend(str(n) for n in range(int(a), int(b) + 1))
            else:
                labels.extend([a, b])
            continue
        labels.extend(part.split())
    out: list[str] = []
    for lab in labels:
        lab = lab.rstrip(".")
        if _LABEL.fullmatch(lab) and lab not in out:
            out.append(lab)
    return out


def find_para_links(text: str) -> list[tuple[tuple[int, int], list[str]]]:
    """Every "[Para ...]" link as ((start, end), labels). Line breaks count as spaces."""
    joined = text.replace("\n", " ")
    return [
        ((m.start(), m.end()), parse_para_list(m.group(1))) for m in _PARA_LINK.finditer(joined)
    ]


def para_refs(text: str) -> list[str]:
    """All paragraph labels named by links in `text`, in order, without repeats."""
    out: list[str] = []
    for _, labels in find_para_links(text):
        out.extend(lab for lab in labels if lab not in out)
    return out


# --- Act names -------------------------------------------------------------------

# The design regex, plus "Code of Criminal/Civil Procedure" (whose name does not
# end in a keyword) and word boundaries around the keyword ("contract, 1990").
_NAME_CHARS = r"[A-Za-z().&'\u2019\- ]"
_NAME = rf"[A-Z]{_NAME_CHARS}{{2,120}}?"
_ACT_RE = re.compile(
    r"(?:The\s+)?("
    rf"(?:{_NAME}\s)?Code\s+of\s+(?:Criminal|Civil)\s+Procedure"
    rf"|{_NAME}\b(?:Act|Code|Rules|Regulations|Ordinance|Order|Adhiniyam|Sanhita)"
    r")\s*[,;.]\s*(\d{4})\b",  # "." and ";" are OCR forms of the comma
    re.IGNORECASE,
)
# En and em dashes always; an ASCII hyphen only next to a space ("1988 - s.3",
# "(appeal)- Tribunal"), so "Co-operative" stays one word.
_DASHES = re.compile(r"\s*[\u2013\u2014]\s*|\s+-\s*|\s*-\s+")
_LONE_MARGIN = re.compile(r"(?<=\s)[A-H](?=\s)")


@lru_cache(maxsize=8)
def _alias_map_from_file(path: str) -> dict[str, str]:
    raw = tomllib.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for canon, variants in raw.get("aliases", {}).items():
        c = _prepare_act(canon)
        out[c] = c
        for v in variants:
            out[_prepare_act(v)] = c
    return out


def load_act_aliases(settings: Settings | None = None) -> dict[str, str]:
    """{prepared variant: canonical name} from `acts.alias_file`."""
    settings = settings or default_settings()
    return dict(_alias_map_from_file(settings.acts.alias_file))


def _prepare_act(name: str) -> str:
    s = normalize_text(name)
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s*[,;]?\s*(\d{4})\.?$", r", \1", s)
    s = re.sub(r"\s+,", ",", s)
    return s.strip()


def canonical_act(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """normalize_text(), then the alias file; an unmapped name keeps its prepared form.

    Also used for an Act filter value before it is compared with refs.value_norm.
    """
    if aliases is None:
        aliases = load_act_aliases()
    prepared = _prepare_act(name)
    return aliases.get(prepared, prepared)


def act_normalizer(settings: Settings | None = None) -> Callable[[str], str]:
    """`canonical_act` bound to the alias file: the Act filter's normaliser (file 20)."""
    aliases = load_act_aliases(settings)
    return lambda name: canonical_act(name, aliases)


def act_refs(
    names: Iterable[str], aliases: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """(value, value_norm) pairs for `refs` rows of kind 'act'."""
    if aliases is None:
        aliases = load_act_aliases()
    return [(n, canonical_act(n, aliases)) for n in names]


def unmapped_act_names(
    names: Iterable[str], aliases: Mapping[str, str] | None = None
) -> Counter[str]:
    """Counts of prepared names the alias file does not map (for act_names_unmapped.csv)."""
    if aliases is None:
        aliases = load_act_aliases()
    return Counter(p for p in (_prepare_act(n) for n in names) if p not in aliases)


def _prepare_catchwords(text: str) -> str:
    text = " ".join(text.split())
    text = _LONE_MARGIN.sub(" ", text)
    text = _DASHES.sub(" \u2013 ", text)
    return " ".join(text.split())


def catchword_groups(text: str, settings: Settings | None = None) -> list[str]:
    """Catchword groups: split after a full stop followed by a capital letter,
    unless the full stop ends one of `chunk.abbreviations`."""
    settings = settings or default_settings()
    abbrevs = {a.lower() for a in settings.chunk.abbreviations}
    text = _prepare_catchwords(text)
    groups: list[str] = []
    start = 0
    for m in re.finditer(r"\.\s+(?=[A-Z])", text):
        word = text[start : m.start() + 1].split()[-1].lstrip("([").lower()
        if word in abbrevs:
            continue
        groups.append(text[start : m.start() + 1].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        groups.append(tail)
    return groups


def _trim_leading_lower(name: str) -> str:
    words = name.split()
    while words and not (words[0][0].isupper() or words[0][0] in "(&"):
        words.pop(0)
    return " ".join(words)


def _no_year_patterns(settings: Settings) -> list[tuple[re.Pattern[str], str]]:
    out = []
    for name in settings.acts.no_year_allowlist:
        first = re.escape(name.split()[0])
        rx = re.compile(
            rf"\b{re.escape(name)}\b|\b{first}\b(?=\s*[,:\u2013-]?\s*Arts?\.)",
            re.IGNORECASE,
        )
        out.append((rx, name))
    return out


def _with_bracket_prefix(group: str, start: int, name: str) -> str:
    """Extend a name that closes a bracket opened before it, across commas:
    "Hazardous Wastes (Management, Handling ...) Rules"."""
    if name.count(")") <= name.count("("):
        return name.strip()
    prefix = group[:start]
    idx = prefix.rfind("(")
    if idx < 0 or ")" in prefix[idx:]:
        return name.strip()
    lead = re.search(_NAME_CHARS + r"*$", prefix[:idx])
    return (prefix[lead.start() if lead else idx :] + name).strip()


def _group_acts(group: str, settings: Settings) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for m in _ACT_RE.finditer(group):
        name = _trim_leading_lower(_with_bracket_prefix(group, m.start(1), m.group(1)))
        if name and name.lower() not in {"act", "code", "rules", "order"}:
            found.append((m.start(1), f"{name}, {m.group(2)}"))
    taken = [(m.start(), m.end()) for m in _ACT_RE.finditer(group)]
    for rx, canon in _no_year_patterns(settings):
        for m in rx.finditer(group):
            if not any(s <= m.start() < e for s, e in taken):
                found.append((m.start(), canon))
    found.sort()
    return found


def extract_era_c_acts(
    text: str,
    settings: Settings | None = None,
    aliases: Mapping[str, str] | None = None,
    unmatched: list[str] | None = None,
    clean: Callable[[str], str] | None = None,
) -> list[str]:
    """Every Act named in Era C catchword text, as printed, one per canonical name.

    Groups with no Act are appended to `unmatched` when it is given. `clean`
    (Era B: `title_run`) rewrites each name before it is deduplicated; names
    it empties are dropped.
    """
    settings = settings or default_settings()
    if aliases is None:
        aliases = load_act_aliases(settings)
    out: list[str] = []
    seen: set[str] = set()
    for group in catchword_groups(text, settings):
        acts = _group_acts(group, settings)
        if not acts and unmatched is not None:
            unmatched.append(group)
        for _, name in acts:
            if clean is not None:
                name = clean(name)
                if not name:
                    continue
            key = canonical_act(name, aliases)
            if key not in seen:
                seen.add(key)
                out.append(name)
    return out


_NAME_JOINERS = frozenset({"of", "and", "the", "for", "in", "on", "to", "&"})


def title_run(name: str) -> str:
    """Era B: the Act name as the run of capitalised words before its year.

    Era B prints a summary of facts between the catchwords and "HELD:", so a
    name can start mid-sentence ("The appellant is a registered dealer under
    the Tamil Nadu General Sales Tax Act, 1959"). The name kept is the
    longest run of words ending at the keyword in which every word starts
    with a capital (or a bracket and a capital, "(Companies)") or is a joiner
    such as "of" or "and", and whose first word is capitalised: "Tamil Nadu
    General Sales Tax Act, 1959". "" when only the keyword is left.
    """
    head, sep, year = name.rpartition(", ")
    if not sep:
        head, year = name, ""
    words = head.split()
    keep: list[str] = []
    for w in reversed(words):
        bare = w.lstrip("(")
        if (bare[:1].isupper()) or w.lower() in _NAME_JOINERS:
            keep.append(w)
        else:
            break
    # a leading joiner, or a lone letter (a margin letter OCR left: "I I Customs Act")
    while keep and (not keep[-1].lstrip("(")[:1].isupper() or len(keep[-1]) == 1):
        keep.pop()
    if len(keep) < 2:  # the keyword alone ("Act, 1959")
        return ""
    out = " ".join(reversed(keep))
    return f"{out}, {year}" if year else out


def _split_list(text: str) -> list[str]:
    """Entries of an Era D "List of Acts" / "List of Keywords" block."""
    joined = " ".join(text.split())
    out: list[str] = []
    for part in joined.split(";"):
        part = part.strip().rstrip(".").strip()
        if part and part not in out:
            out.append(part)
    return out


# --- zoning ----------------------------------------------------------------------


def _first(rx: re.Pattern[str], text: str, pos: int = 0) -> re.Match[str] | None:
    return rx.search(text, pos)


def _line_end(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl < 0 else nl + 1


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _held_point(text: str, start: int, end: int) -> HeldPoint | None:
    """A Held point over text[start:end], cut after its last paragraph link."""
    start, end = _strip_span(text, start, end)
    if start >= end:
        return None
    links = find_para_links(text[start:end])
    if links:
        cut = start + links[-1][0][1]
        run = _PINPOINT_RUN.match(text, cut)
        if run:
            cut = run.end()
        end = min(end, cut)
        start, end = _strip_span(text, start, end)
    body = text[start:end]
    return HeldPoint(text=body, char_start=start, char_end=end, para_refs=para_refs(body))


def _era_c_points(text: str, start: int, end: int) -> list[HeldPoint]:
    """Held points of one Era C HELD block over text[start:end].

    Numbers must run in order. The first follows "HELD:" directly or starts a
    line within FIRST_POINT_WINDOW characters (after "PER X, J. (for himself
    ...)"); numbering restarts at 1 after a line starting "Per" (a new opinion's
    Held points under the same HELD).
    """
    cuts: list[int] = []
    major, minor = 0, 0
    for m in _C_POINT.finditer(text, start, end):
        label = m.group(1)
        a, _, b = label.partition(".")
        ma, mi = int(a), int(b) if b else 0
        ok = (ma == major + 1 and mi in (0, 1)) or (ma == major and mi == minor + 1 and major)
        at_line_start = _LINE_START.search(text, max(0, m.start(1) - 20), m.start(1)) is not None
        if not cuts:
            ok = ok and (
                m.start(1) - start <= 3
                or (at_line_start and m.start(1) - start <= FIRST_POINT_WINDOW)
            )
        else:
            restart = ma == 1 and mi in (0, 1) and _PER.search(text, cuts[-1], m.start(1))
            ok = (ok or bool(restart)) and at_line_start
        if ok:
            cuts.append(m.start(1))
            major, minor = ma, mi
    if not cuts:
        cuts = [start]
    bounds = [*cuts, end]
    points = []
    for s, e in itertools.pairwise(bounds):
        hp = _held_point(text, s, e)
        if hp:
            points.append(hp)
    return points


def _zone_era_c(
    text: str,
    settings: Settings,
    aliases: Mapping[str, str],
    unmatched: list[str] | None = None,
) -> Zones:
    n = len(text)
    missing: list[Zone] = []
    held_rx = _C_HELD if _C_HELD.search(text) else _C_HELD_TITLE
    first_held = held_rx.search(text)
    limit = min(BENCH_WINDOW, first_held.start() if first_held else len(text))
    bench = _C_BENCH.search(text, 0, limit)
    if bench:
        header_span = (0, bench.end())
        ed_start = _line_end(text, bench.end())
    else:
        header_span = (0, 0)
        ed_start = 0
        missing.append(Zone.HEADER)
    helds = list(held_rx.finditer(text, ed_start))
    search_from = helds[0].end() if helds else ed_start
    juris = _first(_C_JURISDICTION, text, search_from) or _first(_C_DELIVERED, text, search_from)
    case_law = _first(_C_CASE_LAW, text, search_from)
    if case_law and juris and case_law.start() > juris.start():
        case_law = None
    ed_end = min(m.start() for m in (case_law, juris) if m) if (case_law or juris) else n
    helds = [h for h in helds if h.start() < ed_end]

    if not case_law:
        missing.append(Zone.REFERENCES)
    if juris:
        judgment_span = (juris.start(), n)
    else:
        judgment_span = (0, n)
        missing.append(Zone.JUDGMENT)

    catch_end = helds[0].start() if helds else ed_end
    acts = extract_era_c_acts(text[ed_start:catch_end], settings, aliases, unmatched)

    editorial: list[HeldPoint] = []
    for i, h in enumerate(helds):
        block_end = helds[i + 1].start() if i + 1 < len(helds) else ed_end
        editorial.extend(_era_c_points(text, h.end(), block_end))
    if not editorial and ed_start >= ed_end:
        missing.append(Zone.EDITORIAL)

    return Zones(
        header_span=header_span,
        issue_span=None,
        editorial=editorial,
        acts=acts,
        keywords=[],
        judgment_span=judgment_span,
        missing=missing,
    )


# --- Era B -----------------------------------------------------------------------

# The bench line when OCR breaks the Era C form: "JJ,]", "JJ.].", "JJ.J" or
# "JJ � J" at the end, or no opening bracket ("tS. MURTAZA FAZAL ALI, ...
# JJ.J"). A line ending in "JJ" or "C.J.", or "J." with a closing bracket.
_B_BENCH = re.compile(
    r"^[^\n]*(?:\bJ[ \t]?J[ \t]?[.,]?[ \t]?[\])J}]?|\bJ[ \t]?[.,][ \t]?[\])}]"
    r"|\bC[ \t]?\.?[ \t]?J[ \t]?I?[ \t]?[.,]?[ \t]?[\])}])[^\w\n]{0,4}$",
    re.MULTILINE,
)
# "HELD:" with the OCR forms of 1965-2009 scans: "HELB :", "HBLD:", "HELD;".
_B_HELD = re.compile(r"\bH[EB][LI1][DB][ \t]*[:;]")
# The jurisdiction line with OCR damage: "CtVIL APPELLATE JURISDICTION",
# "C1v1L APPHLATE JURISDICTION", "CRIMINAL APPELLATE JuR1smcnoN", ".E CIVIL ..."
# (a margin letter or speck first). Upper-case first word, as in Era C.
_B_JURISDICTION = re.compile(
    r"^[ \t]*[^\w\n]{0,3}[ \t]*(?:[A-H][ \t]+)?"
    r"(?:C[A-Za-z0-9.!]{2,3}L\b|CR[A-Za-z1!]{3,6}L|OR[A-Za-z1!]{3,6}L|WR[A-Z1l]{1,2}T|APP[A-Z1l]{3,6}E)"
    r"[A-Za-z0-9 \t()/&,.\-]{0,30}?\bJ[A-Za-z0-9!]{3,12}",
    re.MULTILINE,
)
# "The Judgment of the Court was delivered by", "The following Order of the
# Court was delivered:", "The following Judgments were deliver~d;", with OCR
# forms ("J udi:mcnt of the Cou:t", "of' the Court"); "delivered" may wrap.
_B_DELIVERED = re.compile(
    r"^[^\n]{0,60}?(?:Court|Judgments?|Orders?|C\S{3,4}[tl])[^\n\w]{0,3}[ \t]+"
    r"(?:was|were)\s*[.,]?\s*deliver",
    _F,
)
# The author line that opens the judgment when no "delivered by" line is
# printed: "KRISHNA IYER, J. The detenu ...", "S.B. SINHA, J. A Will was ...",
# "MADON J. These two Appeals" (upper-case name; counsel lines are mixed case).
_B_AUTHOR = re.compile(
    r"^[ \t]*[A-Z][A-Z0-9. ]{2,40},?[ \t]*(?:C\.J|J)\.(?![A-Za-z])", re.MULTILINE
)
# "JUDGMENT" or "ORDER" alone on a line (upper case).
_B_JUDGMENT_LINE = re.compile(
    r"^[ \t]*(?:J[ \t]?U[ \t]?D[ \t]?G[ \t]?(?:E[ \t]?)?M[ \t]?E[ \t]?N[ \t]?T"
    r"|O[ \t]?R[ \t]?D[ \t]?E[ \t]?R)[ \t]*:?[ \t]*$",
    re.MULTILINE,
)
# Where the case details start when no jurisdiction line is found.
_B_DETAILS = re.compile(
    r"^[ \t]*(?:(?:Appeals?|Petitions?)[ \t]+(?:by|from|under)\b"
    r"|From[ \t]+the[ \t]+(?:Judgment|Order)|\((?:Under|Appeal)\b)",
    re.MULTILINE,
)
# How far after the jurisdiction line the "delivered by" line may be: the case
# number, the order appealed from and the counsel lines come between.
DELIVERED_WINDOW = 3000
# Held points numbered "(1)", "(i)", "(I)" (OCR for "(1)").
_B_POINT = re.compile(
    r"(?:^|(?<=\s))\([ \t]*([0-9]{1,2}|[ivx]{1,5}|[IVX]{1,5}|l|!)[ \t]*\)\s+(?=\S)"
)
_ROMAN = {"i": 1, "v": 5, "x": 10}


def _point_value(label: str) -> int | None:
    if label.isdigit():
        return int(label)
    if label in ("l", "!", "I"):
        return 1
    total, prev = 0, 0
    for ch in reversed(label.lower()):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _era_b_points(text: str, start: int, end: int) -> list[HeldPoint]:
    """Held points of one Era B HELD block.

    Numbered like Era C ("1.", "1.1") or in brackets ("(1)", "(i)"): the
    bracketed split is used when the Era C one finds a single point. Bracketed
    numbers must run 1, 2, 3 ... and start a line; the first may instead
    follow "HELD:" directly, and must come within FIRST_POINT_WINDOW
    characters.
    """
    points = _era_c_points(text, start, end)
    if len(points) > 1:
        return points
    cuts: list[int] = []
    for m in _B_POINT.finditer(text, start, end):
        v = _point_value(m.group(1))
        if v != len(cuts) + 1:
            continue
        line_start = _LINE_START.search(text, max(0, m.start() - 20), m.start()) is not None
        if cuts:
            ok = line_start
        else:
            ok = m.start() - start <= 3 or (line_start and m.start() - start <= FIRST_POINT_WINDOW)
        if ok:
            cuts.append(m.start())
    if len(cuts) < 2:
        return points
    # Text between "HELD:" and the first number stays with the first point.
    bounds = [start, *cuts[1:], end]
    out = []
    for s, e in itertools.pairwise(bounds):
        hp = _held_point(text, s, e)
        if hp:
            out.append(hp)
    return out


def _zone_era_b(
    text: str,
    settings: Settings,
    aliases: Mapping[str, str],
    unmatched: list[str] | None = None,
) -> Zones:
    """Era B zones: as Era C, with the OCR forms of HELD and the jurisdiction
    line, bracketed Held points, no "Case Law Reference" table, and the
    judgment starting at "The Judgment of the Court was delivered by" (or a
    "JUDGMENT" / "ORDER" line, or the author's "NAME, J." line) after the
    case details and counsel."""
    n = len(text)
    missing: list[Zone] = []
    held_rx = (
        _B_HELD if _B_HELD.search(text) else _C_HELD_TITLE if _C_HELD_TITLE.search(text) else None
    )
    first_held = held_rx.search(text) if held_rx else None
    limit = min(BENCH_WINDOW, first_held.start() if first_held else n)
    bench = _C_BENCH.search(text, 0, limit) or _B_BENCH.search(text, 0, limit)
    if bench:
        header_span = (0, bench.end())
        ed_start = _line_end(text, bench.end())
    else:
        header_span = (0, 0)
        ed_start = 0
        missing.append(Zone.HEADER)
    helds = list(held_rx.finditer(text, ed_start)) if held_rx else []
    search_from = helds[0].end() if helds else ed_start
    juris = _first(_B_JURISDICTION, text, search_from)
    after = juris.end() if juris else search_from
    window_end = after + DELIVERED_WINDOW if juris else n
    marks = [
        m
        for m in (
            _first(_B_DELIVERED, text, after),
            _first(_B_JUDGMENT_LINE, text, after),
            _first(_B_AUTHOR, text, after),
        )
        if m is not None and m.start() < window_end
    ]
    delivered = min(marks, key=lambda m: m.start()) if marks else None
    if juris:
        ed_end = juris.start()
    elif delivered:
        details = _first(_B_DETAILS, text, search_from)
        ed_end = (
            details.start()
            if details and details.start() < delivered.start()
            else delivered.start()
        )
    else:
        ed_end = n
    helds = [h for h in helds if h.start() < ed_end]
    if delivered:
        judgment_span = (delivered.start(), n)
    elif juris:
        judgment_span = (juris.start(), n)
    else:
        judgment_span = (0, n)
        missing.append(Zone.JUDGMENT)

    catch_end = helds[0].start() if helds else ed_end
    acts = extract_era_c_acts(
        text[ed_start:catch_end], settings, aliases, unmatched, clean=title_run
    )
    editorial: list[HeldPoint] = []
    for i, h in enumerate(helds):
        block_end = helds[i + 1].start() if i + 1 < len(helds) else ed_end
        editorial.extend(_era_b_points(text, h.end(), block_end))
    if not editorial and ed_start >= ed_end:
        missing.append(Zone.EDITORIAL)
    return Zones(
        header_span=header_span,
        issue_span=None,
        editorial=editorial,
        acts=acts,
        keywords=[],
        judgment_span=judgment_span,
        missing=missing,
    )


def _zone_era_d(text: str, aliases: Mapping[str, str]) -> Zones:
    n = len(text)
    marks = {
        Zone.ISSUE: _first(_D_ISSUE, text),
        Zone.EDITORIAL: _first(_D_HEADNOTES, text),
        Zone.REFERENCES: _first(_D_CASE_LAW, text),
        Zone.JUDGMENT: _first(_D_JUDGMENT, text),
    }
    sub = {
        "acts": _first(_D_ACTS, text),
        "keywords": _first(_D_KEYWORDS, text),
        "arising": _first(_D_ARISING, text),
    }
    starts = sorted(m.start() for m in [*marks.values(), *sub.values()] if m is not None)

    def next_mark(pos: int) -> int:
        return next((s for s in starts if s > pos), n)

    missing = [z for z, m in marks.items() if m is None]
    header_end = starts[0] if starts else n
    header_span = (0, header_end)

    issue_span = None
    if (m := marks[Zone.ISSUE]) is not None:
        issue_span = _strip_span(text, _line_end(text, m.start()), next_mark(m.start()))

    editorial: list[HeldPoint] = []
    if (m := marks[Zone.EDITORIAL]) is not None:
        ed_end = next_mark(m.start())
        helds = list(_D_HELD.finditer(text, m.end(), ed_end))
        for i, h in enumerate(helds):
            e = helds[i + 1].start() if i + 1 < len(helds) else ed_end
            hp = _held_point(text, h.start(), e)
            if hp:
                editorial.append(hp)

    acts: list[str] = []
    seen: set[str] = set()
    if (m := sub["acts"]) is not None:
        for name in _split_list(text[_line_end(text, m.start()) : next_mark(m.start())]):
            key = canonical_act(name, aliases)
            if key not in seen:
                seen.add(key)
                acts.append(name)
    keywords: list[str] = []
    if (m := sub["keywords"]) is not None:
        keywords = _split_list(text[_line_end(text, m.start()) : next_mark(m.start())])

    if (m := marks[Zone.JUDGMENT]) is not None:
        judgment_span = (_line_end(text, m.start()), n)
    else:
        judgment_span = (0, n)

    return Zones(
        header_span=header_span,
        issue_span=issue_span,
        editorial=editorial,
        acts=acts,
        keywords=keywords,
        judgment_span=judgment_span,
        missing=missing,
    )


def zone(
    parsed: ParsedDoc,
    settings: Settings | None = None,
    aliases: Mapping[str, str] | None = None,
    unmatched: list[str] | None = None,
) -> Zones:
    """Zones of one judgment. All offsets point into parsed.clean_text.

    Era C catchword groups that name no Act are appended to `unmatched` when
    it is given (the build logs them as `act_unmatched`).
    """
    settings = settings or default_settings()
    if aliases is None:
        aliases = load_act_aliases(settings)
    if parsed.era is Era.D:
        return _zone_era_d(parsed.clean_text, aliases)
    if parsed.era is Era.B:
        return _zone_era_b(parsed.clean_text, settings, aliases, unmatched)
    return _zone_era_c(parsed.clean_text, settings, aliases, unmatched)
