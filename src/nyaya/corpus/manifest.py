"""The manifest: one row per judgment, every year (design file 16, step 1).

`build_manifest(settings)` reads every `year=YYYY/metadata.parquet` under
`paths.metadata_root`, then:

* maps the metadata columns to schema columns (`map_columns`), taking
  `folder_year` from each file's path;
* keeps one row per `path` (`dedupe`), preferring the copy whose PDF is in its
  own folder's tar, then the later `scraped_at`, then the lexically lowest tar
  path;
* joins each kept row to `<path>_EN.pdf`, looking in the `folder_year` tar
  first and then in every other tar in path order;
* sets `has_pdf`. Rows without a PDF stay, with null tar fields.

It writes `build/manifest.parquet` and `build/manifest_summary.json`.

Manifest columns: the schema columns `path, title, petitioner, respondent,
judges, author_judge, scr_citation, neutral_citation, cnr, disposal, outcome,
decision_date, decision_year`, plus `folder_year`, `has_pdf`, `tar_path`
(absolute, POSIX separators), `offset_data` and `size`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from nyaya.config import REPO_ROOT, OutcomeMap, Settings
from nyaya.corpus.tarstore import scan_offsets

FOLDER_YEAR_RE = re.compile(r"year=(\d{4})")
TAR_GLOB = "year=*/english/english.tar"
PDF_SUFFIX = "_EN.pdf"

MANIFEST_COLUMNS = [
    "path",
    "title",
    "petitioner",
    "respondent",
    "judges",
    "author_judge",
    "scr_citation",
    "neutral_citation",
    "cnr",
    "disposal",
    "outcome",
    "decision_date",
    "decision_year",
    "folder_year",
    "has_pdf",
    "tar_path",
    "offset_data",
    "size",
]

# Schema column <- source column, copied as trimmed text ("" when missing).
_TEXT_COLUMNS = {
    "path": "path",
    "title": "title",
    "petitioner": "petitioner",
    "respondent": "respondent",
    "scr_citation": "citation",
    "cnr": "cnr",
    "disposal": "disposal_nature",
}
# Schema column <- source column, trimmed, None when blank.
_NULLABLE_COLUMNS = {"author_judge": "author_judge", "neutral_citation": "nc_display"}


def resolve(root: Path, p: str) -> Path:
    """`p` from settings, relative to the repository root unless absolute."""
    q = Path(p)
    return q if q.is_absolute() else root / q


def folder_year_of(path: Path | str) -> int:
    m = FOLDER_YEAR_RE.search(Path(path).as_posix())
    if not m:
        raise ValueError(f"no year=YYYY folder in {path}")
    return int(m.group(1))


def read_metadata(metadata_root: Path) -> pd.DataFrame:
    """Every `year=YYYY/metadata.parquet`, concatenated, with `folder_year`."""
    files = sorted(metadata_root.glob("year=*/metadata.parquet"))
    if not files:
        raise FileNotFoundError(f"no year=*/metadata.parquet under {metadata_root}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["folder_year"] = folder_year_of(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _key(s: str) -> str:
    return " ".join(s.split()).casefold()


def outcome_for(disposal: str, outcome_map: OutcomeMap) -> str:
    """The `outcome_map` label for a raw disposal value.

    Matching ignores case and runs of whitespace; anything unlisted, including
    the empty value, maps to `outcome_map.default`.
    """
    k = _key(disposal)
    if k:
        for label, values in outcome_map.values.items():
            if any(_key(v) == k for v in values):
                return label
    return outcome_map.default


def split_judges(judge: str) -> list[str]:
    return [j.strip() for j in judge.split(",") if j.strip()]


def map_columns(df: pd.DataFrame, outcome_map: OutcomeMap) -> pd.DataFrame:
    """Metadata rows -> schema columns, plus `folder_year` and `scraped_at`.

    `scraped_at` is kept only for the duplicate rule and dropped afterwards.
    """
    out = pd.DataFrame(index=df.index)
    for dst, src in _TEXT_COLUMNS.items():
        out[dst] = _text(df[src]).astype(object)
    for dst, src in _NULLABLE_COLUMNS.items():
        v = _text(df[src])
        out[dst] = v.where(v != "", None).astype(object)
    out["judges"] = pd.Series(
        [split_judges(j) for j in _text(df["judge"])], index=df.index, dtype=object
    )
    out["outcome"] = pd.Series(
        [outcome_for(d, outcome_map) for d in out["disposal"]], index=df.index, dtype=object
    )
    dates = pd.to_datetime(_text(df["decision_date"]), format="%d-%m-%Y", errors="raise")
    out["decision_date"] = dates.dt.date
    out["decision_year"] = dates.dt.year.astype("int64")
    out["folder_year"] = df["folder_year"].astype("int64")
    out["scraped_at"] = pd.to_datetime(_text(df["scraped_at"]), errors="coerce")
    blank = out["path"] == ""
    if blank.any():
        raise ValueError(f"{int(blank.sum())} metadata rows have a blank path")
    return out


def tar_path_for(tar_root: Path, folder_year: int) -> Path:
    return tar_root / f"year={folder_year}" / "english" / "english.tar"


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per `path`.

    Needs the columns `pdf_in_own_tar` (bool), `scraped_at` and `own_tar`
    (the row's folder tar path as text). Prefers a PDF in the row's own tar,
    then the later `scraped_at`, then the lexically lowest tar path.
    """
    ranked = df.assign(_scraped=df["scraped_at"].fillna(pd.Timestamp.min)).sort_values(
        ["path", "pdf_in_own_tar", "_scraped", "own_tar"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    kept = ranked.drop_duplicates("path", keep="first").drop(columns="_scraped")
    return kept.sort_index()


def join_tars(
    df: pd.DataFrame,
    offsets_by_tar: Mapping[str, Mapping[str, tuple[int, int]]],
) -> pd.DataFrame:
    """Set `has_pdf`, `tar_path`, `offset_data`, `size` on deduplicated rows.

    `offsets_by_tar` maps a tar path (text, as in `own_tar`) to its offsets.
    The row's own tar is tried first, then every tar in sorted path order.
    """
    order = sorted(offsets_by_tar)
    tar_paths, offs, sizes = [], [], []
    for path, own in zip(df["path"], df["own_tar"], strict=True):
        name = path + PDF_SUFFIX
        hit = None
        for t in ([own] if own in offsets_by_tar else []) + order:
            o = offsets_by_tar[t].get(name)
            if o is not None:
                hit = (t, o)
                break
        if hit is None:
            tar_paths.append(None)
            offs.append(pd.NA)
            sizes.append(pd.NA)
        else:
            tar_paths.append(hit[0])
            offs.append(hit[1][0])
            sizes.append(hit[1][1])
    out = df.copy()
    out["tar_path"] = pd.Series(tar_paths, index=df.index, dtype=object)
    out["offset_data"] = pd.array(offs, dtype="Int64")
    out["size"] = pd.array(sizes, dtype="Int64")
    out["has_pdf"] = out["tar_path"].notna()
    return out


def assemble(
    meta: pd.DataFrame,
    outcome_map: OutcomeMap,
    tar_root: Path,
    offsets_by_tar: Mapping[str, Mapping[str, tuple[int, int]]],
) -> tuple[pd.DataFrame, dict]:
    """Raw metadata -> (manifest, summary). Pure apart from reading nothing."""
    rows_in = len(meta)
    df = map_columns(meta, outcome_map)
    df["own_tar"] = [tar_path_for(tar_root, y).as_posix() for y in df["folder_year"]]
    df["pdf_in_own_tar"] = [
        (p + PDF_SUFFIX) in offsets_by_tar.get(t, {})
        for p, t in zip(df["path"], df["own_tar"], strict=True)
    ]
    df = dedupe(df)
    df = join_tars(df, offsets_by_tar)
    df = df.sort_values(["decision_year", "path"], kind="mergesort").reset_index(drop=True)
    manifest = df[MANIFEST_COLUMNS]
    per_year = manifest.groupby("decision_year").agg(
        judgments=("path", "size"), with_pdf=("has_pdf", "sum")
    )
    summary = {
        "rows_in": rows_in,
        "duplicates_removed": rows_in - len(manifest),
        "judgments": len(manifest),
        "judgments_without_pdf": int((~manifest["has_pdf"]).sum()),
        "per_decision_year": {
            str(y): {"judgments": int(r.judgments), "with_pdf": int(r.with_pdf)}
            for y, r in per_year.iterrows()
        },
    }
    return manifest, summary


def build_manifest(settings: Settings, root: Path | str | None = None) -> Path:
    """Build `build/manifest.parquet` and `build/manifest_summary.json`; return the manifest path.

    `root` is the repository root that relative `paths.*` resolve against
    (default: this checkout). Tar offsets are cached in `build/offsets/`.
    """
    root = Path(root) if root is not None else REPO_ROOT
    build_dir = resolve(root, settings.paths.build)
    tar_root = resolve(root, settings.paths.tar_root)
    meta = read_metadata(resolve(root, settings.paths.metadata_root))
    offsets_by_tar = {
        t.as_posix(): scan_offsets(t, build_dir / "offsets")
        for t in sorted(tar_root.glob(TAR_GLOB))
    }
    manifest, summary = assemble(meta, settings.outcome_map, tar_root, offsets_by_tar)
    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / "manifest.parquet"
    manifest.to_parquet(out, index=False)
    (build_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return out


def load_manifest(settings: Settings, root: Path | str | None = None) -> pd.DataFrame:
    """Read `build/manifest.parquet`."""
    root = Path(root) if root is not None else REPO_ROOT
    return pd.read_parquet(resolve(root, settings.paths.build) / "manifest.parquet")
