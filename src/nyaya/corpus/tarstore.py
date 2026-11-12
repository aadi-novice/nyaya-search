"""Random access to PDFs inside the uncompressed per-year tars (design file 15).

Ports the offset logic of scripts/data_check.py (lines 65-100): the tar is
scanned once with `tarfile`, and each member's `offset_data` (the offset of
the file's content, after its header) and `size` are kept. A PDF is then read
with one seek and one read, without opening the tar again.

Offsets are cached as `build/offsets/<tar>.parquet`, where `<tar>` names the
tar by its year folder (every tar is called english.tar), for example
`year=2015_english.parquet`. The cache records the tar's size and mtime and is
ignored when either changes.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pandas as pd

Offsets = dict[str, tuple[int, int]]

_CACHE_COLUMNS = ["name", "offset_data", "size", "tar_size", "tar_mtime_ns"]


def cache_name(tar_path: Path | str) -> str:
    """Cache file name for a tar, e.g. `year=2015_english.parquet`.

    The nearest `year=YYYY` folder names the tar (all tars are english.tar);
    without one, the parent folder does.
    """
    p = Path(tar_path)
    owner = next((a.name for a in p.parents if a.name.startswith("year=")), p.parent.name)
    return f"{owner}_{p.stem}.parquet"


def _scan(tar_path: Path) -> Offsets:
    out: Offsets = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.isfile():
                out[os.path.basename(m.name)] = (int(m.offset_data), int(m.size))
    return out


def scan_offsets(tar_path: Path | str, cache_dir: Path | str | None = None) -> Offsets:
    """Map each member's base name to `(offset_data, size)`.

    With `cache_dir`, the result is read from (or written to)
    `<cache_dir>/<cache_name(tar)>`; a cache made for a tar of a different size
    or mtime is rebuilt.
    """
    tar_path = Path(tar_path)
    st = tar_path.stat()
    cache: Path | None = None
    if cache_dir is not None:
        cache = Path(cache_dir) / cache_name(tar_path)
        if cache.exists():
            df = pd.read_parquet(cache)
            if (
                list(df.columns) == _CACHE_COLUMNS
                and len(df) > 0
                and int(df["tar_size"].iloc[0]) == st.st_size
                and int(df["tar_mtime_ns"].iloc[0]) == st.st_mtime_ns
            ):
                return {
                    n: (int(o), int(s))
                    for n, o, s in zip(df["name"], df["offset_data"], df["size"], strict=True)
                }
    offsets = _scan(tar_path)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "name": list(offsets),
                "offset_data": [o for o, _ in offsets.values()],
                "size": [s for _, s in offsets.values()],
                "tar_size": st.st_size,
                "tar_mtime_ns": st.st_mtime_ns,
            },
            columns=_CACHE_COLUMNS,
        )
        df.to_parquet(cache, index=False)
    return offsets


def read_pdf(tar_path: Path | str, offset_data: int, size: int) -> bytes:
    """Read `size` bytes at `offset_data` from the tar."""
    with open(tar_path, "rb") as f:
        f.seek(offset_data)
        data = f.read(size)
    if len(data) != size:
        raise OSError(f"{tar_path}: short read at offset {offset_data} ({len(data)} of {size})")
    return data
