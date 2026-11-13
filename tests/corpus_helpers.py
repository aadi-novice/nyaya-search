"""Helpers for the corpus and parse tests: synthetic PDFs, tars and data paths."""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import pymupdf

from nyaya.config import Settings, load_settings

from conftest import ROOT

DATA_TAR_ROOT = ROOT / "docs" / "data" / "tar"
DATA_META_ROOT = ROOT / "docs" / "metadata" / "parquet"

# A page in the shape of the born-digital Era C volumes (2018-2023).
PAGE_W, PAGE_H = 612.0, 792.0
LEFT_LETTER_X = 144.0  # even pages
RIGHT_LETTER_X = 473.0  # odd pages
BODY_X0 = 163.0
TOP_HEAD_Y = 115.0
FIRST_BODY_Y = 158.0
LINE_STEP = 13.2
MARGIN_LETTERS = "ABCDEFGH"


def clean_settings() -> Settings:
    """Settings from config/ alone, ignoring any build/bands.json and .env in the checkout."""
    tmp = Path(tempfile.mkdtemp(prefix="nyaya_cfg_"))
    shutil.copytree(ROOT / "config", tmp / "config")
    return load_settings(tmp, env_file=None)


def tar_for(year: int) -> Path:
    return DATA_TAR_ROOT / f"year={year}" / "english" / "english.tar"


def era_c_page(
    doc: pymupdf.Document,
    pno: int,
    body: Sequence[str],
    *,
    printed_page: int,
    title: str = "ALPHA LTD. v. STATE OF BETA",
    head: bool = True,
    letters: bool = True,
) -> None:
    """Add one Era C page: margin letters A-H, a running head, and body lines."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    even = pno % 2 == 0
    if head and pno > 0:
        if even:
            page.insert_text((140, TOP_HEAD_Y), str(printed_page), fontsize=10)
            page.insert_text((205, TOP_HEAD_Y), "SUPREME COURT REPORTS", fontsize=10)
            page.insert_text((398, TOP_HEAD_Y), "[2020] 11 S.C.R.", fontsize=10)
        else:
            page.insert_text((200, TOP_HEAD_Y), title, fontsize=10)
            page.insert_text((472, TOP_HEAD_Y), str(printed_page), fontsize=10)
    if letters:
        x = LEFT_LETTER_X if even else RIGHT_LETTER_X
        for i, letter in enumerate(MARGIN_LETTERS):
            page.insert_text((x, FIRST_BODY_Y + i * 72.7), letter, fontsize=10)
    for i, line in enumerate(body):
        page.insert_text(
            (BODY_X0 if even else 149.0, FIRST_BODY_Y + i * LINE_STEP), line, fontsize=10
        )


def era_c_pdf(pages: Sequence[Sequence[str]], *, start_page: int = 786, **kw) -> bytes:
    doc = pymupdf.open()
    for pno, body in enumerate(pages):
        era_c_page(doc, pno, body, printed_page=start_page + pno, **kw)
    data = doc.tobytes()
    doc.close()
    return data


def plain_pdf(
    pages: Sequence[Sequence[str]], *, width: float = 439.35, height: float = 666.15
) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page(width=width, height=height)
        for i, line in enumerate(body):
            page.insert_text((75, 100 + i * 13), line, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def write_tar(tar_path: Path, members: Iterable[tuple[str, bytes]]) -> Path:
    """Write an uncompressed tar holding `members` (name, bytes)."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path
