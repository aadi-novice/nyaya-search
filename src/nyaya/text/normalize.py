"""Text normalisation used everywhere text is compared (safety-critical).

`normalize_text` is used for quote matching, Act names and case titles.
`normalize_judge` is used for judge names. Citation strings use
`canonical_citation()` in link/aliases.py, not these functions.
"""

from __future__ import annotations

import re
import unicodedata

# Curly quotes to straight quotes, en and em dashes to a hyphen.
_CHAR_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201a": "'",  # single low-9 quotation mark
        "\u201b": "'",  # single high-reversed-9 quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u201e": '"',  # double low-9 quotation mark
        "\u201f": '"',  # double high-reversed-9 quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)


def normalize_text(text: str) -> str:
    """NFKC, straight quotes and hyphens, collapsed whitespace, lower case.

    Whitespace runs become one space and the ends are trimmed. Hyphenated line
    breaks are already rejoined by the parser, so they are not handled here.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP)
    text = " ".join(text.split())
    return text.lower()


# Title prefix: "DR." (optionally in brackets) at the start of the name.
_PREFIX_RE = re.compile(r"^\s*\(?DR\.?\)?\s+")
# Title suffix at the end of the name, after a comma or a space:
# "J.", "JJ.", "CJI", "C.J." (the full stop optional), then an optional colon.
_SUFFIX_RE = re.compile(r"(?:^|[\s,]+)(?:JJ|J|CJI|C\.\s?J)\.?\s*:?\s*$")


def normalize_judge(name: str) -> str:
    """Upper case; remove "J.", "JJ.", "CJI", "C.J." and "DR."; full stops to spaces.

    "D.Y. CHANDRACHUD, J." -> "D Y CHANDRACHUD". Titles are removed only where
    titles sit (a "DR." prefix, a trailing "J."/"JJ."/"CJI"/"C.J."), so an
    initial such as the "J." in "J. CHELAMESWAR" is kept. Commas, which only
    separate a name from its title, also become spaces.
    """
    s = name.upper()
    s = _PREFIX_RE.sub("", s)
    while True:
        stripped = _SUFFIX_RE.sub("", s)
        if stripped == s:
            break
        s = stripped
    s = s.replace(".", " ").replace(",", " ")
    return " ".join(s.split())
