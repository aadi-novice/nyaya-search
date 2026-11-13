"""`nyaya-build`: the offline build (design file 26, Ground rules "Build entry points").

Subcommands, run in this order (`run` does them all, skipping finished work):

    nyaya-build manifest                       build/manifest.parquet
    nyaya-build calibrate                      build/bands.json (Era C bands)
    nyaya-build parse-chunk --years 2025       parse, zone, chunk, link per decision year
    nyaya-build embed --years 2025             build/vectors/<year>.npy (+ .ids.txt), GPU
    nyaya-build load --version <name>          a new `building` index version, checked
    nyaya-build activate --version <name>
    nyaya-build run --years 2025 --version <name>

`--years` takes `2025`, `2010-2026` or `2024,2025`. Years are decision years.

Build files (the layout `store/loader.py` reads):

    judgments/<year>.parquet    one row per judgment (with or without a PDF), the
                                judgments-table columns, clean_text, position_map (JSON)
    chunks/<year>.jsonl         one ChunkRecord per line
    links/<year>.parquet        from_chunk_id, to_chunk_id (Held links)
    refs/<year>.parquet         path, kind (act | keyword), value, value_norm
    vectors/<year>.npy          float32 (n, embed.dim), rows in chunks file order
    vectors/<year>.ids.txt      the n chunk IDs, one per line
    aliases.parquet             alias_norm, kind, path, for every judgment built
    chunk_metrics.json          G6 automatic metrics over the years built (or loaded)
    link_summary.json           Held-link resolution over the years built
    act_names_unmapped.csv      the 200 most frequent Act names the alias file misses
    skipped.csv                 path, stage, reason, detail
    errors.jsonl                one exception per line

Per-year working files live in `logs/` (skipped, errors, doc metrics, link
summary, unmapped Act counts) and `state/` (stamps). A year is skipped when
its stamp exists and records the same clean_text, chunker and config versions
(for parse-chunk) or the same embed model, revision and chunk IDs (for embed);
`--force` redoes it. A year's files are written only after the whole year
succeeds, so an interrupted run resumes at the first unfinished year.

Exit codes: 0 success; 2 calibration or usage error; 3 a G6 metric failed;
4 errors above `build.max_error_share`; 5 a post-load check failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nyaya.config import REPO_ROOT, Settings, load_settings
from nyaya.contracts import ChunkRecord, Era

log = logging.getLogger("nyaya.build")

ERROR_EXIT_CODE = 4
G6_EXIT_CODE = 3
UNMAPPED_TOP_N = 200
SKIPPED_COLUMNS = ("path", "stage", "reason", "detail")
PROGRESS_EVERY = 100


# --- arguments ----------------------------------------------------------------------


def parse_years(text: str) -> list[int]:
    """'2025', '2010-2026' or '2024,2025' -> sorted distinct years."""
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if a > b:
                raise argparse.ArgumentTypeError(f"bad year range {part!r}")
            years.extend(range(a, b + 1))
        else:
            years.append(int(part))
    if not years:
        raise argparse.ArgumentTypeError("no years given")
    return sorted(set(years))


# --- paths and small file helpers ---------------------------------------------------------


@dataclass(frozen=True)
class BuildPaths:
    root: Path

    @classmethod
    def from_settings(cls, settings: Settings, repo_root: Path | None = None) -> BuildPaths:
        p = Path(settings.paths.build)
        return cls(p if p.is_absolute() else (repo_root or REPO_ROOT) / p)

    def year_file(self, kind: str, year: int, suffix: str) -> Path:
        return self.root / kind / f"{year}{suffix}"

    def judgments(self, year: int) -> Path:
        return self.year_file("judgments", year, ".parquet")

    def chunks(self, year: int) -> Path:
        return self.year_file("chunks", year, ".jsonl")

    def links(self, year: int) -> Path:
        return self.year_file("links", year, ".parquet")

    def refs(self, year: int) -> Path:
        return self.year_file("refs", year, ".parquet")

    def vectors(self, year: int) -> Path:
        return self.year_file("vectors", year, ".npy")

    def vector_ids(self, year: int) -> Path:
        return self.year_file("vectors", year, ".ids.txt")

    def vector_hashes(self, year: int) -> Path:
        return self.year_file("vectors", year, ".sha1.txt")

    def log(self, name: str, year: int) -> Path:
        return self.root / "logs" / f"{name}_{year}"

    def stamp(self, stage: str, year: int) -> Path:
        return self.root / "state" / f"{stage}_{year}.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.parquet"

    @property
    def bands(self) -> Path:
        return self.root / "bands.json"

    def built_years(self) -> list[int]:
        """Decision years with a finished parse-chunk stamp."""
        out = []
        for f in (self.root / "state").glob("parse_*.json"):
            try:
                out.append(int(f.stem.split("_", 1)[1]))
            except ValueError:
                continue
        return sorted(out)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, default=str) + "\n")


# --- one judgment -------------------------------------------------------------------------


@dataclass
class DocResult:
    """Everything parse-chunk produces for one judgment."""

    path: str
    judgment: dict[str, Any]
    chunks: list[ChunkRecord] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)
    refs: list[tuple[str, str, str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str, str, str]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    link_summary: dict[str, Any] | None = None
    unmapped_acts: Counter[str] = field(default_factory=Counter)
    era: str | None = None
    seconds: float = 0.0


def judgment_base(row: dict[str, Any]) -> dict[str, Any]:
    """A manifest row as a judgments-table row with empty text fields."""

    def text(v: Any) -> Any:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    judges = row.get("judges")
    judges = list(judges) if judges is not None else []
    return {
        "path": row["path"],
        "title": text(row.get("title")),
        "petitioner": text(row.get("petitioner")),
        "respondent": text(row.get("respondent")),
        "judges": judges,
        "author_judge": text(row.get("author_judge")),
        "scr_citation": text(row.get("scr_citation")),
        "neutral_citation": text(row.get("neutral_citation")),
        "cnr": text(row.get("cnr")),
        "disposal": text(row.get("disposal")),
        "outcome": text(row.get("outcome")),
        "decision_date": row.get("decision_date"),
        "decision_year": int(row["decision_year"]),
        "era": None,
        "has_pdf": bool(row.get("has_pdf")),
        "tar_file": text(row.get("tar_path")),
        "offset_data": None if pd.isna(row.get("offset_data")) else int(row["offset_data"]),
        "byte_size": None if pd.isna(row.get("size")) else int(row["size"]),
        "pages": None,
        "clean_text": None,
        "position_map": None,
    }


def opinion_judges(row: dict[str, Any]) -> list[str]:
    """Names for split_opinions: the `judge` list plus `author_judge`."""
    names = [j for j in (row.get("judges") or []) if j]
    author = row.get("author_judge")
    if isinstance(author, str) and author.strip() and author not in names:
        names.append(author.strip())
    return names


class DocProcessor:
    """parse -> zone -> opinions -> chunk -> link for one manifest row."""

    def __init__(
        self,
        settings: Settings,
        *,
        count_tokens: Callable[[str], int] | None = None,
        read_pdf: Callable[[str, int, int], bytes] | None = None,
    ) -> None:
        from nyaya.corpus.tarstore import read_pdf as tar_read_pdf
        from nyaya.zone.zoner import load_act_aliases

        self.settings = settings
        self.aliases = load_act_aliases(settings)
        self.count_tokens = count_tokens
        self.read_pdf = read_pdf or tar_read_pdf

    def __call__(self, row: dict[str, Any]) -> DocResult:
        t0 = time.perf_counter()
        res = DocResult(path=row["path"], judgment=judgment_base(row))
        stage = "parse"
        try:
            if not res.judgment["has_pdf"]:
                res.skipped.append((res.path, "parse", "no_pdf", ""))
                return res
            stage = self._process(row, res)
        except Exception as exc:
            res.error = {
                "path": res.path,
                "stage": stage,
                "era": res.era,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            res.chunks, res.links, res.refs, res.metrics = [], [], [], None
            res.judgment.update(clean_text=None, position_map=None, pages=None)
        finally:
            res.seconds = time.perf_counter() - t0
        return res

    def _process(self, row: dict[str, Any], res: DocResult) -> str:
        from nyaya.chunk.chunker import chunk
        from nyaya.chunk.metrics import doc_metrics
        from nyaya.link.linker import resolve_held_links
        from nyaya.parse.eras import classify_pdf
        from nyaya.parse.pdf_layout import parse
        from nyaya.text.normalize import normalize_text
        from nyaya.zone.opinions import split_opinions
        from nyaya.zone.zoner import act_refs, unmapped_act_names, zone

        s = self.settings
        data = self.read_pdf(row["tar_path"], int(row["offset_data"]), int(row["size"]))
        era = classify_pdf(data, int(row["decision_year"]), s.era_markers)
        res.era = era.value
        res.judgment["era"] = era.value
        if era is Era.OUT_OF_SCOPE:
            res.skipped.append((res.path, "parse", "era_out_of_scope", ""))
            return "parse"

        parsed = parse(data, res.path, era, s)
        res.judgment.update(
            pages=parsed.pages,
            clean_text=parsed.clean_text,
            position_map=json.dumps([list(p) for p in parsed.position_map]),
        )

        stage = "zone"
        unmatched: list[str] = []
        zones = zone(parsed, s, self.aliases, unmatched)
        for z in zones.missing:
            res.skipped.append((res.path, "zone", "zone_missing", z.value))
        for group in unmatched:
            res.skipped.append((res.path, "acts", "act_unmatched", group[:300]))
        res.unmapped_acts = unmapped_act_names(zones.acts, self.aliases)

        opinions = split_opinions(parsed, zones, opinion_judges(row), s)
        stage = "chunk"
        meta = {"title": row.get("title"), "decision_year": row.get("decision_year")}
        res.chunks = chunk(meta, parsed, zones, opinions, s, self.aliases)
        res.metrics = doc_metrics(
            parsed,
            zones,
            opinions,
            res.chunks,
            s,
            count_tokens=self.count_tokens,
            token_limit=s.encoders.max_tokens,
        )

        stage = "link"
        res.links, res.link_summary = resolve_held_links(res.chunks)
        for u in res.link_summary["unresolved_refs"]:
            res.skipped.append((res.path, "link", "para_unresolved", f"{u['chunk_id']} {u['ref']}"))
        for value, value_norm in act_refs(zones.acts, self.aliases):
            res.refs.append((res.path, "act", value, value_norm))
        for kw in dict.fromkeys(zones.keywords):
            res.refs.append((res.path, "keyword", kw, normalize_text(kw)))
        return stage


# Worker processes build their own DocProcessor once (spawn on Windows).
_WORKER: DocProcessor | None = None


def _init_worker(root: str | None, tokens: bool) -> None:  # pragma: no cover - subprocess
    global _WORKER
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings(root) if root else load_settings()
    counter = None
    if tokens:
        from nyaya.chunk.metrics import load_token_counter

        counter = load_token_counter(settings)
    _WORKER = DocProcessor(settings, count_tokens=counter)


def _work(row: dict[str, Any]) -> DocResult:  # pragma: no cover - subprocess
    assert _WORKER is not None
    return _WORKER(row)


# --- parse-chunk -----------------------------------------------------------------------


def parse_stamp(settings: Settings) -> dict[str, str]:
    v = settings.versions
    return {"clean_text": v.clean_text, "chunker": v.chunker, "config": v.config}


@dataclass
class YearReport:
    year: int
    judgments: int
    with_pdf: int
    parsed: int
    chunks: int
    errors: int
    out_of_scope: int
    by_era: dict[str, int]
    errors_by_era: dict[str, int]
    seconds: float
    skipped: bool = False


def _year_rows(manifest: pd.DataFrame, year: int, limit: int | None) -> list[dict[str, Any]]:
    rows = manifest[manifest["decision_year"] == year].sort_values("path", kind="mergesort")
    if limit is not None:
        rows = rows.head(limit)
    out = []
    for r in rows.to_dict(orient="records"):
        j = r.get("judges")
        r["judges"] = list(j) if j is not None else []
        out.append(r)
    return out


def _iter_results(
    rows: list[dict[str, Any]],
    processor: Callable[[dict[str, Any]], DocResult] | None,
    workers: int,
    settings_root: str | None,
    tokens: bool,
) -> Iterable[DocResult]:
    if workers <= 1:
        assert processor is not None
        for r in rows:
            yield processor(r)
        return
    with ProcessPoolExecutor(  # pragma: no cover - exercised by the real build
        max_workers=workers, initializer=_init_worker, initargs=(settings_root, tokens)
    ) as pool:
        yield from pool.map(_work, rows, chunksize=4)


def write_year(paths: BuildPaths, year: int, results: list[DocResult]) -> None:
    """Write one year's build files and logs (each file atomically)."""
    from nyaya.store.loader import JUDGMENT_LOAD_COLUMNS

    jrows = [r.judgment for r in results]
    jdf = pd.DataFrame(jrows, columns=[c for c in JUDGMENT_LOAD_COLUMNS if c != "judges_norm"])
    jdf["decision_date"] = pd.to_datetime(jdf["decision_date"]).dt.date
    _write_parquet(jdf, paths.judgments(year))

    lines = [c.model_dump_json() for r in results for c in r.chunks]
    _atomic_write_text(paths.chunks(year), "".join(line + "\n" for line in lines))

    links = [lk for r in results for lk in r.links]
    _write_parquet(
        pd.DataFrame(links, columns=["from_chunk_id", "to_chunk_id"], dtype=object),
        paths.links(year),
    )
    refs = [rf for r in results for rf in r.refs]
    _write_parquet(
        pd.DataFrame(refs, columns=["path", "kind", "value", "value_norm"], dtype=object),
        paths.refs(year),
    )

    skipped = [s for r in results for s in r.skipped]
    _atomic_write_text(paths.log("skipped", year).with_suffix(".csv"), _csv(skipped))
    errors = [r.error for r in results if r.error]
    _atomic_write_text(
        paths.log("errors", year).with_suffix(".jsonl"),
        "".join(json.dumps(e) + "\n" for e in errors),
    )
    metrics = [r.metrics for r in results if r.metrics]
    _atomic_write_text(
        paths.log("doc_metrics", year).with_suffix(".jsonl"),
        "".join(json.dumps(m) + "\n" for m in metrics),
    )
    from nyaya.link.linker import merge_summaries

    _write_json(
        paths.log("link_summary", year).with_suffix(".json"),
        merge_summaries(r.link_summary for r in results if r.link_summary),
    )
    unmapped: Counter[str] = Counter()
    for r in results:
        unmapped.update(r.unmapped_acts)
    _write_json(paths.log("acts_unmapped", year).with_suffix(".json"), dict(unmapped))


def _csv(rows: Sequence[Sequence[Any]]) -> str:
    import io

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(SKIPPED_COLUMNS)
    w.writerows(rows)
    return out.getvalue()


def parse_chunk(
    settings: Settings,
    years: Sequence[int],
    *,
    paths: BuildPaths | None = None,
    force: bool = False,
    workers: int = 1,
    limit: int | None = None,
    processor: Callable[[dict[str, Any]], DocResult] | None = None,
    settings_root: str | None = None,
    count_tokens: bool = True,
) -> list[YearReport]:
    """Parse, zone, chunk and link every judgment of `years`; then merge the summaries.

    Raises SystemExit(4) if errors exceed `build.max_error_share`, and
    SystemExit(3) if a G6 metric over all built years misses its threshold.
    """
    paths = paths or BuildPaths.from_settings(settings)
    manifest = pd.read_parquet(paths.manifest)
    stamp = parse_stamp(settings)
    if processor is None and workers <= 1:
        counter = None
        if count_tokens:
            from nyaya.chunk.metrics import load_token_counter

            counter = load_token_counter(settings)
        processor = DocProcessor(settings, count_tokens=counter)

    reports: list[YearReport] = []
    total_errors = total_docs = 0
    for year in years:
        old = _read_json(paths.stamp("parse", year))
        if not force and old is not None and old.get("versions") == stamp and limit is None:
            log.info("parse-chunk %d: done already (%s judgments), skipped", year, old["judgments"])
            reports.append(YearReport(**{**old["report"], "skipped": True}))
            continue
        rows = _year_rows(manifest, year, limit)
        if not rows:
            log.info("parse-chunk %d: no judgments in the manifest", year)
            continue
        t0 = time.perf_counter()
        results: list[DocResult] = []
        for i, res in enumerate(
            _iter_results(rows, processor, workers, settings_root, count_tokens), 1
        ):
            results.append(res)
            if res.error:
                log.warning("%s: %s failed: %s", res.path, res.error["stage"], res.error["error"])
            if i % PROGRESS_EVERY == 0 or i == len(rows):
                el = time.perf_counter() - t0
                log.info("parse-chunk %d: %d/%d judgments, %.1f s (%.2f s each)", year, i,
                         len(rows), el, el / i)  # fmt: skip
        write_year(paths, year, results)
        by_era = Counter(r.era or "no_pdf" for r in results)
        errors_by_era = Counter(r.era or "unclassified" for r in results if r.error)
        report = YearReport(
            year=year,
            judgments=len(results),
            with_pdf=sum(1 for r in results if r.judgment["has_pdf"]),
            parsed=sum(1 for r in results if r.judgment["clean_text"] is not None),
            chunks=sum(len(r.chunks) for r in results),
            errors=sum(1 for r in results if r.error),
            out_of_scope=by_era.get(Era.OUT_OF_SCOPE.value, 0),
            by_era=dict(by_era),
            errors_by_era=dict(errors_by_era),
            seconds=round(time.perf_counter() - t0, 1),
        )
        log.info("parse-chunk %d: %s", year, report)
        total_errors += report.errors
        total_docs += report.with_pdf
        if limit is None:
            _write_json(
                paths.stamp("parse", year),
                {"versions": stamp, "judgments": report.judgments, "report": report.__dict__},
            )
        reports.append(report)

    write_summaries(settings, paths, paths.built_years() or list(years))
    if total_docs and total_errors / total_docs > settings.build.max_error_share:
        log.error(
            "%d errors in %d judgments with a PDF (> %.1f%%): see %s",
            total_errors,
            total_docs,
            settings.build.max_error_share * 100,
            paths.root / "errors.jsonl",
        )
        raise SystemExit(ERROR_EXIT_CODE)
    failed = g6_failures(settings, paths)
    if failed:
        metrics_file = paths.root / "chunk_metrics.json"
        log.error("G6 gate failed: %s (see %s)", ", ".join(failed), metrics_file)
        raise SystemExit(G6_EXIT_CODE)
    return reports


def _concat_logs(paths: BuildPaths, name: str, suffix: str, years: Sequence[int]) -> list[Path]:
    return [f for y in years if (f := paths.log(name, y).with_suffix(suffix)).exists()]


def write_summaries(settings: Settings, paths: BuildPaths, years: Sequence[int]) -> None:
    """chunk_metrics.json, link_summary.json, skipped.csv, errors.jsonl, aliases.parquet
    and act_names_unmapped.csv over `years`."""
    from nyaya.chunk.metrics import summarize, write_chunk_metrics
    from nyaya.link.aliases import build_alias_rows
    from nyaya.link.linker import merge_summaries, write_link_summary

    docs = [
        json.loads(line)
        for f in _concat_logs(paths, "doc_metrics", ".jsonl", years)
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(docs, settings)
    summary["years"] = list(years)
    write_chunk_metrics(summary, paths.root / "chunk_metrics.json")

    links = [_read_json(f) for f in _concat_logs(paths, "link_summary", ".json", years)]
    write_link_summary(merge_summaries(s for s in links if s), paths.root / "link_summary.json")

    skipped_rows: list[list[str]] = []
    for f in _concat_logs(paths, "skipped", ".csv", years):
        with f.open(encoding="utf-8", newline="") as fh:
            skipped_rows.extend(list(csv.reader(fh))[1:])
    _atomic_write_text(paths.root / "skipped.csv", _csv(skipped_rows))

    errors = "".join(
        f.read_text(encoding="utf-8") for f in _concat_logs(paths, "errors", ".jsonl", years)
    )
    _atomic_write_text(paths.root / "errors.jsonl", errors)

    unmapped: Counter[str] = Counter()
    for f in _concat_logs(paths, "acts_unmapped", ".json", years):
        unmapped.update(_read_json(f) or {})
    with_header = [("name", "count"), *unmapped.most_common(UNMAPPED_TOP_N)]
    _atomic_write_text(
        paths.root / "act_names_unmapped.csv",
        "".join(f"{_csv_field(n)},{c}\n" for n, c in with_header),
    )

    jfiles = [f for y in years if (f := paths.judgments(y)).exists()]
    if jfiles:
        manifest = pd.read_parquet(paths.manifest)
        loaded = set().union(*(set(pd.read_parquet(f, columns=["path"])["path"]) for f in jfiles))
        rows = build_alias_rows(manifest, loaded)
        _write_parquet(
            pd.DataFrame(rows, columns=["alias_norm", "kind", "path"], dtype=object),
            paths.root / "aliases.parquet",
        )


def _csv_field(s: str) -> str:
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


def g6_failures(settings: Settings, paths: BuildPaths) -> list[str]:
    from nyaya.chunk.metrics import failing_metrics

    summary = _read_json(paths.root / "chunk_metrics.json")
    if summary is None:
        return ["chunk_metrics.json missing"]
    failed = failing_metrics(summary, settings)
    if not summary.get("tokens_counted"):
        failed.append("over_model_limit (tokens not counted)")
    return failed


# --- embed ---------------------------------------------------------------------------


def read_chunk_texts(path: Path) -> tuple[list[str], list[str]]:
    """(chunk IDs, embedded texts: header, blank line, text) of one chunks file."""
    from nyaya.models.encoders import document_text

    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = ChunkRecord.model_validate_json(line)
                ids.append(c.chunk_id)
                texts.append(document_text(c.header, c.text))
    return ids, texts


def token_batches(lengths: Sequence[int], budget: int, max_batch: int) -> list[list[int]]:
    """Indexes grouped longest first, so that batch size x longest length <= budget.

    Sorting by length keeps padding small; the budget keeps peak memory flat
    whatever the chunk lengths. A single over-budget text gets its own batch.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * longest > budget):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def text_hash(text: str) -> str:
    """Identity of an embedded text: equal hashes mean an equal vector can be reused."""
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_stamp(settings: Settings, ids: Sequence[str], hashes: Sequence[str]) -> dict[str, Any]:
    import hashlib

    rows = "".join(f"{i} {h}\n" for i, h in zip(ids, hashes, strict=True))
    return {
        "model": settings.embed.model,
        "revision": settings.embed.revision,
        "dim": settings.embed.dim,
        "n": len(ids),
        "rows_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest()[:16],
    }


def reusable_vectors(
    paths: BuildPaths, year: int, stamp: dict[str, Any], dim: int
) -> dict[str, np.ndarray]:
    """{text hash: vector} from the year's previous vectors, if made by the same model."""
    old = _read_json(paths.stamp("embed", year)) or {}
    hfile = paths.vector_hashes(year)
    if not (hfile.exists() and paths.vectors(year).exists()):
        return {}
    if any(old.get(k) != stamp[k] for k in ("model", "revision", "dim")):
        return {}
    hashes = hfile.read_text(encoding="utf-8").split()
    vecs = np.load(paths.vectors(year))
    if vecs.shape != (len(hashes), dim):
        return {}
    return dict(zip(hashes, vecs, strict=True))


@dataclass
class EmbedReport:
    year: int
    chunks: int
    seconds: float
    per_second: float
    peak_vram_gb: float | None
    skipped: bool = False


def embed(
    settings: Settings,
    years: Sequence[int],
    *,
    paths: BuildPaths | None = None,
    force: bool = False,
    embedder: Any = None,
    count_tokens: Callable[[str], int] | None = None,
) -> list[EmbedReport]:
    """Embed each year's chunks (documents, no prefix) into vectors/<year>.npy."""
    paths = paths or BuildPaths.from_settings(settings)
    reports: list[EmbedReport] = []
    batch = settings.encoders.embed_batch_size
    budget = batch * 1024  # tokens per forward pass: 16 chunks of ~1k tokens
    for year in years:
        cfile = paths.chunks(year)
        if not cfile.exists():
            log.info("embed %d: no chunks file, skipped", year)
            continue
        ids, texts = read_chunk_texts(cfile)
        hashes = [text_hash(t) for t in texts]
        stamp = embed_stamp(settings, ids, hashes)
        old = _read_json(paths.stamp("embed", year))
        if (
            not force
            and old == stamp
            and paths.vectors(year).exists()
            and paths.vector_ids(year).exists()
        ):
            log.info("embed %d: done already (%d chunks), skipped", year, len(ids))
            reports.append(EmbedReport(year, len(ids), 0.0, 0.0, None, skipped=True))
            continue
        if embedder is None:
            from nyaya.models.encoders import Embedder

            embedder = Embedder(settings)
        if count_tokens is None:
            tok = getattr(getattr(embedder, "model", None), "tokenizer", None)
            if tok is not None:

                def count_tokens(s: str, _tok: Any = tok) -> int:
                    return len(_tok(s, add_special_tokens=True)["input_ids"])

            else:

                def count_tokens(s: str) -> int:
                    return len(s) // 4 + 1

        cuda = _cuda()
        if cuda is not None:
            cuda.reset_peak_memory_stats()
        out = np.zeros((len(texts), settings.embed.dim), dtype=np.float32)
        cache = {} if force else reusable_vectors(paths, year, stamp, settings.embed.dim)
        todo = []
        for i, h in enumerate(hashes):
            if h in cache:
                out[i] = cache[h]
            else:
                todo.append(i)
        if cache:
            log.info("embed %d: %d of %d vectors reused", year, len(texts) - len(todo), len(texts))
        lengths = [count_tokens(texts[i]) for i in todo]
        t0 = time.perf_counter()
        done = 0
        batches = [[todo[k] for k in bt] for bt in token_batches(lengths, budget, batch)]
        for n, idx in enumerate(batches, 1):
            vecs = embedder.embed_documents([texts[i] for i in idx], batch_size=len(idx))
            out[idx] = vecs
            done += len(idx)
            if n % 50 == 0 or n == len(batches):
                el = time.perf_counter() - t0
                peak = cuda.max_memory_allocated() / 2**30 if cuda is not None else None
                log.info(
                    "embed %d: %d/%d chunks, %.0f s, %.1f chunks/s, peak VRAM %s",
                    year,
                    done,
                    len(todo),
                    el,
                    done / el if el else 0.0,
                    f"{peak:.2f} GB" if peak is not None else "n/a",
                )
        seconds = time.perf_counter() - t0
        buf = _npy_bytes(out)
        _atomic_write_bytes(paths.vectors(year), buf)
        _atomic_write_text(paths.vector_ids(year), "".join(i + "\n" for i in ids))
        _atomic_write_text(paths.vector_hashes(year), "".join(h + "\n" for h in hashes))
        _write_json(paths.stamp("embed", year), stamp)
        peak = cuda.max_memory_allocated() / 2**30 if cuda is not None else None
        rep = EmbedReport(
            year,
            len(todo),
            round(seconds, 1),
            round(len(todo) / seconds, 2) if seconds else 0.0,
            round(peak, 2) if peak is not None else None,
        )
        log.info("embed %d: %s", year, rep)
        reports.append(rep)
    return reports


def _npy_bytes(a: np.ndarray) -> bytes:
    import io

    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def _cuda() -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    return torch.cuda if torch.cuda.is_available() else None


# --- load and activate ------------------------------------------------------------------


def load_version(
    settings: Settings,
    version: str,
    *,
    paths: BuildPaths | None = None,
    years: Sequence[int] | None = None,
) -> Any:  # pragma: no cover - needs Postgres and the GPU
    """Load the built years (all, or `years`) as `version`, with the post-load checks."""
    from nyaya.models.encoders import Embedder, RerankerChain
    from nyaya.store.db import connect
    from nyaya.store.loader import load

    paths = paths or BuildPaths.from_settings(settings)
    years = list(years) if years else paths.built_years()
    missing = [y for y in years if not paths.vectors(y).exists()]
    if missing:
        raise SystemExit(f"years without vectors (run nyaya-build embed): {missing}")
    write_summaries(settings, paths, years)
    t0 = time.perf_counter()
    embedder = Embedder(settings)
    reranker = RerankerChain.from_settings(settings)
    with connect(settings, autocommit=True) as conn:
        report = load(
            version,
            paths.root,
            settings=settings,
            conn=conn,
            embedder=embedder,
            reranker=reranker,
            smoke_path=REPO_ROOT / "eval" / "datasets" / "phase1_smoke.jsonl",
            years=years,
        )
    log.info("load %s: %s in %.0f s", version, report.counts, time.perf_counter() - t0)
    for o in report.smoke:
        log.info("smoke %s: route %s ok=%s hit=%s", o.id, o.route, o.route_ok, o.hit)
    return report


def activate_version(settings: Settings, version: str) -> dict[str, Any]:  # pragma: no cover
    from nyaya.store.db import connect
    from nyaya.store.loader import activate

    with connect(settings, autocommit=True) as conn:
        out = activate(conn, version)
    log.info("activate: %s", out)
    return out


def version_status(settings: Settings, version: str) -> str | None:  # pragma: no cover
    from nyaya.store.db import connect

    with connect(settings, autocommit=True) as conn:
        row = conn.execute("SELECT status FROM index_versions WHERE id = %s", (version,)).fetchone()
    return row[0] if row else None


# --- CLI -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nyaya-build", description="Nyaya Search offline build")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="build/manifest.parquet from the metadata and tars")
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("calibrate", help="build/bands.json (Era C margin and furniture bands)")
    c.add_argument("--force", action="store_true")

    for name, help_ in (
        ("parse-chunk", "parse, zone, chunk and link the judgments of --years"),
        ("embed", "embed the chunks of --years on the GPU"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--years", type=parse_years, required=True)
        s.add_argument("--force", action="store_true")
        if name == "parse-chunk":
            s.add_argument("--workers", type=int, default=1)
            s.add_argument("--limit", type=int, default=None, help="first N judgments per year")

    ld = sub.add_parser("load", help="load the built years as a new index version")
    ld.add_argument("--version", required=True)
    ld.add_argument("--years", type=parse_years, default=None, help="default: every built year")
    a = sub.add_parser("activate", help="make a ready index version active")
    a.add_argument("--version", required=True)

    r = sub.add_parser("run", help="every stage for --years, skipping finished work")
    r.add_argument("--years", type=parse_years, required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--no-activate", action="store_true")
    return p


def run_manifest(settings: Settings, paths: BuildPaths, force: bool) -> None:
    from nyaya.corpus.manifest import build_manifest

    if paths.manifest.exists() and not force:
        log.info("manifest: %s exists, skipped", paths.manifest)
        return
    out = build_manifest(settings)
    log.info("manifest: %s", _read_json(out.with_name("manifest_summary.json")))


def run_calibrate(settings: Settings, paths: BuildPaths, force: bool) -> Settings:
    """Calibrate if needed; return settings reloaded with the bands merged."""
    from nyaya.parse.calibrate import calibrate

    if paths.bands.exists() and not force:
        log.info("calibrate: %s exists, skipped", paths.bands)
    else:
        out = calibrate(settings)
        log.info("calibrate: %s", _read_json(out))
    return load_settings()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    paths = BuildPaths.from_settings(settings)
    cmd = args.command
    if cmd == "manifest":
        run_manifest(settings, paths, args.force)
    elif cmd == "calibrate":
        run_calibrate(settings, paths, args.force)
    elif cmd == "parse-chunk":
        parse_chunk(settings, args.years, force=args.force, workers=args.workers, limit=args.limit)
    elif cmd == "embed":
        embed(settings, args.years, force=args.force)
    elif cmd == "load":
        load_version(settings, args.version, years=args.years)
    elif cmd == "activate":
        activate_version(settings, args.version)
    elif cmd == "run":
        run_manifest(settings, paths, False)
        settings = run_calibrate(settings, paths, False)
        parse_chunk(settings, args.years, workers=args.workers)
        embed(settings, args.years)
        status = version_status(settings, args.version)
        if status is None:
            load_version(settings, args.version)
        else:
            log.info("load: version %s exists (%s), skipped", args.version, status)
        if not args.no_activate and version_status(settings, args.version) == "ready":
            activate_version(settings, args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
