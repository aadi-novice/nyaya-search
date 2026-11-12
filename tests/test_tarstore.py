"""tarstore: offsets from a tar scan, reads by offset, and the offset cache."""

import os
import random
import tarfile

import pytest

from corpus_helpers import tar_for, write_tar
from nyaya.corpus.tarstore import cache_name, read_pdf, scan_offsets


@pytest.fixture
def small_tar(tmp_path):
    members = [
        ("2025_1_1_5_EN.pdf", b"%PDF-1.7 first"),
        ("sub/2025_1_6_9_EN.pdf", b"%PDF-1.7 second, longer" * 50),
        ("S_2005_5_75_82_EN.pdf", b""),
    ]
    return write_tar(tmp_path / "tar" / "year=2025" / "english" / "english.tar", members), members


def test_scan_offsets_keys_are_basenames(small_tar):
    tar_path, members = small_tar
    offsets = scan_offsets(tar_path)
    assert set(offsets) == {os.path.basename(n) for n, _ in members}


def test_read_pdf_matches_member_bytes(small_tar):
    tar_path, members = small_tar
    offsets = scan_offsets(tar_path)
    for name, data in members:
        off, size = offsets[os.path.basename(name)]
        assert size == len(data)
        assert read_pdf(tar_path, off, size) == data


def test_read_pdf_short_read_raises(small_tar):
    tar_path, _ = small_tar
    with pytest.raises(OSError, match="short read"):
        read_pdf(tar_path, tar_path.stat().st_size - 3, 100)


def test_cache_name_uses_year_folder(small_tar):
    tar_path, _ = small_tar
    assert cache_name(tar_path) == "year=2025_english.parquet"
    assert cache_name("x/other.tar") == "x_other.parquet"


def test_offsets_are_cached_and_reused(small_tar, tmp_path, monkeypatch):
    tar_path, _ = small_tar
    cache_dir = tmp_path / "build" / "offsets"
    first = scan_offsets(tar_path, cache_dir)
    assert (cache_dir / "year=2025_english.parquet").exists()

    import nyaya.corpus.tarstore as ts

    def boom(_):
        raise AssertionError("tar rescanned despite a valid cache")

    monkeypatch.setattr(ts, "_scan", boom)
    assert scan_offsets(tar_path, cache_dir) == first


def test_stale_cache_is_rebuilt(small_tar, tmp_path):
    tar_path, members = small_tar
    cache_dir = tmp_path / "offsets"
    scan_offsets(tar_path, cache_dir)
    write_tar(tar_path, [*members, ("2025_2_1_3_EN.pdf", b"new")])
    assert "2025_2_1_3_EN.pdf" in scan_offsets(tar_path, cache_dir)


@pytest.mark.skipif(not tar_for(2015).exists(), reason="docs/data is absent")
def test_offset_reads_match_tarfile_on_real_tar():
    tar_path = tar_for(2015)
    offsets = scan_offsets(tar_path)
    names = random.Random(5).sample(sorted(offsets), 5)
    with tarfile.open(tar_path, "r:") as tf:
        by_base = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}
        for name in names:
            expected = tf.extractfile(by_base[name]).read()
            off, size = offsets[name]
            got = read_pdf(tar_path, off, size)
            assert got == expected
            assert got.startswith(b"%PDF")
