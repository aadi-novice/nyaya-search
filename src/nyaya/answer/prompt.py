"""Build the answer prompt (design, "Answer generation"; template prompts/<version>.md).

Passages are labelled P1, P2, ... in rank order. Each block gives the label,
case name, zone, opinion type and paragraphs, then the chunk text. The chunk
header is not shown: quotes are checked against the text only.

The input budget is `llm.max_input_tokens`. Tokens are estimated from
characters (`estimate_tokens`); when the prompt is over budget, the
lowest-ranked passages are dropped, but at least one is always kept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nyaya.config import REPO_ROOT, Settings
from nyaya.contracts import ChunkRecord, PassageRef, Zone

PROMPTS_DIR = REPO_ROOT / "prompts"
CHARS_PER_TOKEN = 3.5  # deliberately low for English legal text, so the estimate is high

_ZONE_TEXT = {
    Zone.JUDGMENT: "judgment (the Court's own text)",
    Zone.EDITORIAL: "editorial (a headnote written by the reporter, not the Court)",
    Zone.ISSUE: "issue (an issue summary written by the reporter, not the Court)",
}
_SECTION_RE = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Passage:
    """One reranked passage: what the user sees (ref) and what is quoted (chunk)."""

    ref: PassageRef
    chunk: ChunkRecord


@dataclass(frozen=True)
class Template:
    system: str
    user: str


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def load_template(version: str, prompts_dir: Path = PROMPTS_DIR) -> Template:
    raw = (prompts_dir / f"{version}.md").read_text(encoding="utf-8")
    parts = _SECTION_RE.split(raw)
    # [preamble, "system", body, "user", body]
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    if set(sections) != {"system", "user"}:
        raise ValueError(f"{version}.md needs '## system' and '## user' sections")
    return Template(system=sections["system"].strip(), user=sections["user"].strip())


def _fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def passage_block(label: str, p: Passage) -> str:
    ref = p.ref
    opinion = ref.opinion_type.value if ref.opinion_type is not None else "none"
    paras = ", ".join(ref.paras) if ref.paras else "none"
    return (
        f"[{label}]\n"
        f"Case: {ref.case_name}\n"
        f"Zone: {_ZONE_TEXT.get(ref.zone, ref.zone.value)}\n"
        f"Opinion: {opinion}\n"
        f"Paragraphs: {paras}\n"
        f"Text:\n<<<\n{p.chunk.text}\n>>>"
    )


def build_prompt(
    question: str,
    passages: Sequence[Passage],
    version: str,
    settings: Settings,
) -> tuple[str, str, dict[str, PassageRef]]:
    """Return (system, user, label_map); label_map maps "P1".. to the passage shown."""
    template = load_template(version)
    system = _fill(
        template.system,
        {"min_words": str(settings.quote.min_words), "max_words": str(settings.quote.max_words)},
    )
    kept = list(passages)
    while True:
        labels = [f"P{i}" for i in range(1, len(kept) + 1)]
        blocks = "\n\n".join(passage_block(lab, p) for lab, p in zip(labels, kept, strict=True))
        user = _fill(template.user, {"question": question.strip(), "passages": blocks})
        over = estimate_tokens(system + user) > settings.llm.max_input_tokens
        if not over or len(kept) <= 1:
            break
        kept.pop()
    return system, user, {lab: p.ref for lab, p in zip(labels, kept, strict=True)}
