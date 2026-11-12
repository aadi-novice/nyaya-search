"""One-off: a first development-set draft, before the index exists.

Writes eval/datasets/dev_draft_v0.jsonl (and a hand-check CSV) from Era D
(2024-2025) PDFs, reading the tars read-only. Held points and their
"[Para N]" links come from a simple local extraction, not from the real
parser and zoner, so `source_held_chunk` is null. The questions are drafted
and filtered by nyaya.eval.devset_from_held (the same paraphrase rule as
`nyaya-eval build-devset`). The draft still needs the author's 50-item hand
check; `nyaya-eval build-devset` replaces it with dev_v1.jsonl after the build.

    uv run python scripts/draft_devset_v0.py --data-root D:/Coding_me/rag/docs \
        --env-file D:/Coding_me/rag/.env --size 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pymupdf

from nyaya.config import Secrets, load_settings
from nyaya.eval.datasets import DATASETS_DIR, write_jsonl
from nyaya.eval.devset_from_held import (
    HeldCandidate,
    OpenAICompatLLM,
    draft_devset,
    order_candidates,
    write_handcheck_csv,
)

PARA_LINK_RE = re.compile(r"\[\s*Paras?\s+([0-9][0-9.,&\s\-\u2013andC]*?)\s*[\])]")
FURNITURE_RES = [
    re.compile(r"^\s*\[\d{4}\]\s*\d+\s*S\.C\.R\.\s*(\d+)?\s*$"),
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*Digital Supreme Court Reports\s*$"),
    re.compile(r"^\s*\*\s*Author\s*$"),
]


def parse_para_refs(inner: str) -> list[str]:
    inner = re.sub(r"(?<![0-9A-Za-z])[A-H](?![0-9A-Za-z])", " ", inner)  # stray margin letter
    out: list[str] = []
    for part in re.split(r",|&|\band\b", inner):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)", part)
        if m:
            a, b = m.groups()
            if "." not in a and "." not in b and int(a) <= int(b) and int(b) - int(a) < 30:
                out.extend(str(i) for i in range(int(a), int(b) + 1))
            else:
                out.extend([a, b])
        elif re.fullmatch(r"\d+(?:\.\d+)?", part):
            out.append(part)
    return list(dict.fromkeys(out))


def clean_lines(text: str, title_line: str) -> str:
    keep = []
    for line in text.splitlines():
        if any(r.match(line) for r in FURNITURE_RES) or line.strip() == title_line:
            continue
        keep.append(line.rstrip())
    return "\n".join(keep)


def extract_held(pdf_bytes: bytes, title: str) -> tuple[list[tuple[str, list[str]]], str]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    raw = "".join(page.get_text("text") for page in doc)
    text = clean_lines(raw, title.strip())
    h = re.search(r"(?m)^Headnotes\b", text)
    e = re.search(r"(?m)^Case Law Cited\s*$", text)
    j = re.search(r"(?m)^Judgment\s*/\s*Order of the Supreme Court\s*$", text)
    if not (h and e and j) or e.start() < h.start():
        return [], ""
    zone = text[h.end() : e.start()]
    judgment = text[j.end() :]
    points = []
    starts = [m.start() for m in re.finditer(r"(?m)^Held\s*:|(?<=:\n)Held\s*:", zone)]
    if not starts:
        starts = [m.start() for m in re.finditer(r"Held\s*:", zone)]
    for k, s in enumerate(starts):
        block = zone[s : starts[k + 1] if k + 1 < len(starts) else len(zone)]
        links = list(PARA_LINK_RE.finditer(block))
        if not links:
            continue
        block = block[: links[-1].end()]
        refs: list[str] = []
        for m in links:
            refs.extend(parse_para_refs(m.group(1)))
        flat = " ".join(re.sub(r"-\n(?=[a-z])", "", block).split())
        points.append((flat, list(dict.fromkeys(refs))))
    return points, judgment


class PrecomputedLLM:
    """Answers with questions drafted elsewhere, keyed by Held text (same rule check)."""

    def __init__(self, by_held: dict[str, str]):
        self.by_held = by_held

    def complete_json(self, system: str, user: str, timeout: float) -> dict:
        held = user.split("Held point:\n", 1)[1].split("\n\nParty names", 1)[0]
        return {"question": self.by_held[held]}


def para_present(judgment: str, label: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(label)}\.?(\s|$)", judgment) is not None


def collect(data_root: Path, years: list[int], per_year: int, seed: int) -> list[HeldCandidate]:
    rng = random.Random(seed)
    out: list[HeldCandidate] = []
    for year in years:
        meta = pd.read_parquet(
            data_root / "metadata" / "parquet" / f"year={year}" / "metadata.parquet"
        )
        meta = meta.drop_duplicates("path").set_index("path")
        tar_path = data_root / "data" / "tar" / f"year={year}" / "english" / "english.tar"
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith("_EN.pdf")]
            rng.shuffle(members)
            used = 0
            for m in members:
                if used >= per_year:
                    break
                path = m.name.removesuffix("_EN.pdf")
                if path not in meta.index:
                    continue
                row = meta.loc[path]
                f = tar.extractfile(m)
                if f is None:
                    continue
                points, judgment = extract_held(f.read(), str(row["title"] or ""))
                kept = 0
                for held, refs in points:
                    labels = [r for r in refs if para_present(judgment, r)]
                    if not labels or len(held.split()) < 15:
                        continue
                    out.append(
                        HeldCandidate(
                            path=path,
                            year=year,
                            held_text=held,
                            para_labels=labels,
                            source_held_chunk=None,
                            petitioner=str(row["petitioner"] or ""),
                            respondent=str(row["respondent"] or ""),
                            title=str(row["title"] or ""),
                        )
                    )
                    kept += 1
                used += 1 if kept else 0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True, help="the docs/ folder")
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--per-year", type=int, default=40, help="judgments sampled per year")
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true", help="extract only; no LLM calls")
    ap.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="JSON {index in the seeded order: question} drafted elsewhere; no LLM calls",
    )
    ap.add_argument("--out", type=Path, default=DATASETS_DIR / "dev_draft_v0.jsonl")
    args = ap.parse_args(argv)

    settings = load_settings(env_file=None)
    ds = settings.devset
    secrets = Secrets(_env_file=args.env_file)  # type: ignore[call-arg]
    needs_llm = args.questions_file is None and not args.dry_run
    if needs_llm and not (secrets.llm_base_url and secrets.llm_api_key and secrets.llm_model):
        print("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required", file=sys.stderr)
        return 2
    years = [int(y) for y in args.years.split(",")]
    cands = collect(args.data_root, years, args.per_year, ds.seed)
    n_judgments = len({c.path for c in cands})
    print(f"held points with resolved links: {len(cands)} from {n_judgments} judgments")

    if args.dry_run:
        for c in cands[:5]:
            print(c.path, c.para_labels, c.held_text[:300])
        return 0
    retries = 1
    if args.questions_file is not None:
        ordered = order_candidates(cands, ds.seed)
        given = json.loads(args.questions_file.read_text(encoding="utf-8"))
        by_held = {ordered[int(i)].held_text: q for i, q in given.items()}
        cands = [c for c in cands if c.held_text in by_held]
        llm = PrecomputedLLM(by_held)
        retries = 0  # a rejected question is rejected, not redrafted
    else:
        llm = OpenAICompatLLM(
            secrets.llm_base_url, secrets.llm_api_key.get_secret_value(), secrets.llm_model, 2048
        )
    result = draft_devset(
        cands,
        llm,
        size=args.size,
        max_shared_words=ds.max_shared_words,
        seed=ds.seed,
        retries=retries,
    )
    write_jsonl(args.out, result.items)
    hc = write_handcheck_csv(
        args.out.with_name(args.out.stem + "_handcheck.csv"), result, ds.handcheck_size, ds.seed
    )
    stats = {
        "candidates": len(cands),
        "kept": len(result.items),
        "rejected": len(result.rejected),
        "rejected_by_reason": _count(r["reason"].split(":")[0] for r in result.rejected),
        "llm_calls": 0 if args.questions_file else result.llm_calls,
        "drafts_checked": result.llm_calls,
        "drafted_by": "questions file" if args.questions_file else secrets.llm_model,
        "years": _count(str(result.candidates[i.id].year) for i in result.items),
    }
    rej = args.out.with_name(args.out.stem + "_rejected.jsonl")
    with rej.open("w", encoding="utf-8", newline="\n") as f:
        for r in result.rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}, {hc}, {rej}")
    return 0


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
