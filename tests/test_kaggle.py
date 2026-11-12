"""kaggle/embed_corpus.py and kaggle/push.py: parity with nyaya, output format, CLI.

No GPU, model or network: the encoder is a fake, the Kaggle API is a fake.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from nyaya import build as b
from nyaya.config import Secrets, load_settings
from nyaya.contracts import ChunkRecord
from nyaya.models import encoders as enc
from nyaya.store import loader

ROOT = Path(__file__).resolve().parents[1]
DIM = 1024


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ec = load("embed_corpus")
push = load("push")


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def record(cid: str, header: str, text: str) -> ChunkRecord:
    jid, zone, _ = cid.split("#")
    return ChunkRecord(
        chunk_id=cid,
        judgment_id=jid,
        zone=zone,
        para_labels=["1"],
        para_refs=[],
        text=text,
        header=header,
        char_start=0,
        char_end=len(text),
        pages=[1],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch4",
    )


CHUNKS = [
    record("2025_1_1_9#issue#001", "A v. B (2025) · Issue", "Whether bail lies."),
    record("2025_1_1_9#judgment#001", "", "No header: the text alone."),
    record("2025_1_1_9#judgment#002", "A v. B (2025) · Judgment", "Unicode – ‘quotes’ … ₹ 5"),
    record("2025_1_1_9#judgment#003", "H", "line one\nline two\n\nafter a blank line"),
    record("2025_2_10_20#judgment#001", "C v. D (2025)", "x " * 300),
    record("2025_2_10_20#judgment#002", "C v. D (2025)", "short"),
    record("2025_2_10_20#editorial#001", "C v. D (2025) · Held", "Held: appeal allowed."),
]


def write_chunks(path: Path, chunks=CHUNKS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    return path


class FakeST:
    """Deterministic unit vectors from each text's hash; records batch sizes."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.batches: list[int] = []

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        self.batches.append(len(texts))
        rows = []
        for t in texts:
            seed = int(ec.text_hash(t)[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float32)


# --- parity with the nyaya code -----------------------------------------------------------


def test_document_text_matches_nyaya():
    cases = [("Head", "body"), ("", "body"), ("H · x", "a\n\nb"), ("", ""), ("h", "")]
    cases += [(c.header, c.text) for c in CHUNKS]
    for header, text in cases:
        assert ec.document_text(header, text) == enc.document_text(header, text)


def test_padding_side_matches_nyaya():
    assert ec.PADDING_SIDE == enc.PADDING_SIDE
    assert ec.DEFAULT_PADDING_SIDE == enc.DEFAULT_PADDING_SIDE
    for m in ("Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3", "other/model"):
        assert ec.padding_side(m) == enc.padding_side(m)


def test_text_hash_and_token_batches_match_nyaya():
    rng = random.Random(3)
    for _ in range(50):
        lengths = [rng.randint(1, 3000) for _ in range(rng.randint(0, 60))]
        budget, cap = rng.choice([(4096, 4), (16 * 1024, 16), (32 * 1024, 32)])
        assert ec.token_batches(lengths, budget, cap) == b.token_batches(lengths, budget, cap)
    for t in ("", "a", "₹ – ‘x’", "x" * 5000):
        assert ec.text_hash(t) == b.text_hash(t)


def test_read_chunks_and_stamp_match_nyaya(tmp_path, settings):
    path = write_chunks(tmp_path / "chunks" / "2025.jsonl")
    ids, texts, raw = ec.read_chunks(path)
    assert (ids, texts) == b.read_chunk_texts(path)
    assert [r["chunk_id"] for r in raw] == ids
    hashes = [ec.text_hash(t) for t in texts]
    e = settings.embed
    assert ec.embed_stamp(e.model, e.revision, e.dim, ids, hashes) == b.embed_stamp(
        settings, ids, hashes
    )


# --- output format --------------------------------------------------------------------------


def run_kernel(tmp_path: Path, model: str, revision: str, extra=()):
    inp = tmp_path / "input" / "nyaya-search-chunks-2025"
    write_chunks(inp / "chunks" / "2025.jsonl")
    args = ec.build_parser().parse_args(
        [
            "--model",
            model,
            "--revision",
            revision,
            "--dim",
            str(DIM),
            "--input",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--batch",
            "3",
            *extra,
        ]
    )
    st = FakeST()
    rep = ec.run(args, model_loader=lambda m, r, d: st, count_tokens=len)
    return tmp_path / "out" / ec.model_dir_name(model), rep, st


def test_output_is_what_the_loader_and_startup_check_read(tmp_path, settings):
    e = settings.embed
    out, rep, st = run_kernel(tmp_path, e.model, e.revision)
    assert out.name == "Qwen__Qwen3-Embedding-0.6B"
    assert max(st.batches) <= 3

    # the loader's reader accepts it, rows in chunks-file order
    ids, vecs = loader.read_vectors(out, DIM, [2025])
    assert ids == [c.chunk_id for c in CHUNKS]
    assert vecs.dtype == np.float32 and vecs.shape == (len(CHUNKS), DIM)
    assert loader.check_ids([c.chunk_id for c in CHUNKS], ids) == []
    fake = FakeST()
    want = fake.encode([enc.document_text(c.header, c.text) for c in CHUNKS], **_norm())
    assert np.allclose(vecs, want, atol=1e-6)

    # the reference files pass startup_check with an embedder that agrees
    passages = json.loads((out / "vectors" / "reference" / "passages.json").read_text("utf-8"))
    assert len(passages) == 5 and {"chunk_id", "header", "text"} <= set(passages[0])
    embedder = enc.Embedder(settings, model=FakeST())
    check = enc.startup_check(settings, embedder, out, cuda_available=lambda: True)
    assert check["reference_checked"] and check["min_cosine"] > 0.9999

    # a different model's vectors fail it
    other = enc.Embedder(settings, model=_ShiftedST())
    with pytest.raises(enc.EncoderCheckError, match="disagrees"):
        enc.startup_check(settings, other, out, cuda_available=lambda: True)

    # report
    assert rep["chunks"] == len(CHUNKS) and rep["dim"] == DIM
    assert rep["padding_side"] == "left" and rep["years"]["2025"]["chunks"] == len(CHUNKS)
    assert json.loads((out / "embed_report.json").read_text("utf-8"))["model"] == e.model


def _norm():
    return {"normalize_embeddings": True}


class _ShiftedST(FakeST):
    def encode(self, texts, **kwargs):
        return FakeST.encode(self, [t + " shifted" for t in texts], **kwargs)


def test_kaggle_output_counts_as_done_for_nyaya_build_embed(tmp_path, settings):
    """Copied into a build directory, the year is skipped by `nyaya-build embed`."""
    e = settings.embed
    out, _, _ = run_kernel(tmp_path, e.model, e.revision)
    paths = b.BuildPaths(out)
    write_chunks(paths.chunks(2025))

    class Never:
        def embed_documents(self, texts, batch_size=16):
            raise AssertionError("should not embed")

    (rep,) = b.embed(settings, [2025], paths=paths, embedder=Never(), count_tokens=len)
    assert rep.skipped


def test_bge_m3_run_pads_right(tmp_path, settings):
    p = enc.embed_profile(settings, "BAAI/bge-m3")
    out, rep, _ = run_kernel(tmp_path, p.model, p.revision)
    assert out.name == "BAAI__bge-m3" and rep["padding_side"] == "right"
    stamp = json.loads((out / "state" / "embed_2025.json").read_text("utf-8"))
    assert stamp["model"] == "BAAI/bge-m3" and stamp["revision"] == p.revision


def test_find_chunk_files_and_reference_indexes(tmp_path):
    write_chunks(tmp_path / "a" / "chunks" / "2025.jsonl")
    write_chunks(tmp_path / "a" / "2024.jsonl")
    (tmp_path / "a" / "notes.jsonl").write_text("{}\n", encoding="utf-8")
    assert sorted(ec.find_chunk_files(tmp_path, None)) == [2024, 2025]
    assert list(ec.find_chunk_files(tmp_path, [2025])) == [2025]
    with pytest.raises(SystemExit, match="2019"):
        ec.find_chunk_files(tmp_path, [2019])
    assert ec.reference_indexes(3) == [0, 1, 2]
    assert ec.reference_indexes(101) == [0, 25, 50, 75, 100]


def test_write_year_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        ec.write_year(tmp_path, 2025, ["a"], ["t"], np.zeros((1, 8)), model="m", revision="", dim=4)


# --- embed_corpus CLI -----------------------------------------------------------------------


def test_embed_corpus_arguments():
    a = ec.build_parser().parse_args(["--model", "BAAI/bge-m3", "--dim", "1024"])
    assert a.revision == "" and a.batch == 32 and a.token_budget == 32 * 1024
    assert a.input is None and a.output is None and a.device == "cuda" and a.pip == []
    with pytest.raises(SystemExit):
        ec.build_parser().parse_args(["--model", "m"])  # --dim is required
    assert ec.parse_years("") is None
    assert ec.parse_years("2025") == [2025]
    assert ec.parse_years("2023-2025,2020") == [2020, 2023, 2024, 2025]


def test_embed_corpus_main_runs_kernel_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(ec, "run", lambda args: seen.append(args.model))
    monkeypatch.setattr(sys, "argv", ["embed_corpus.py"])
    monkeypatch.setattr(
        ec, "KERNEL_RUNS", [["--model", "a", "--dim", "4"], ["--model", "b", "--dim", "4"]]
    )
    assert ec.main() == 0 and seen == ["a", "b"]
    monkeypatch.setattr(ec, "KERNEL_RUNS", [])
    assert ec.main() == 2
    # a notebook's own flags (e.g. -f kernel.json) do not count as arguments
    monkeypatch.setattr(sys, "argv", ["embed_corpus.py", "-f", "k.json"])
    assert ec.main() == 2


def test_cuda_run_without_a_gpu_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "_cuda_info", lambda: {"cuda": False})
    write_chunks(tmp_path / "chunks" / "2025.jsonl")
    args = ec.build_parser().parse_args(
        ["--model", "m", "--dim", "4", "--input", str(tmp_path), "--output", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit, match="CUDA"):
        ec.run(args, model_loader=lambda *a: FakeST(4), count_tokens=len)


# --- push.py ---------------------------------------------------------------------------


def test_kernel_source_sets_kernel_runs(settings, tmp_path):
    runs = push.kernel_runs(settings, [], ["--years", "2025"])
    assert [r[1] for r in runs] == ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3"]
    bge = runs[1]
    assert bge[bge.index("--revision") + 1] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert bge[bge.index("--dim") + 1] == "1024" and bge[-2:] == ["--years", "2025"]
    assert "sentence-transformers==6.3.0" in bge
    src = push.kernel_source(runs)
    ns: dict = {}
    exec(compile(src, "kernel.py", "exec"), ns)  # the script still defines everything
    assert ns["KERNEL_RUNS"] == runs
    assert push.kernel_runs(settings, ["BAAI/bge-m3"], [])[0][1] == "BAAI/bge-m3"
    with pytest.raises(KeyError, match="not configured"):
        push.kernel_runs(settings, ["no/such-model"], [])
    bad = tmp_path / "x.py"
    bad.write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(push.KaggleSetupError):
        push.kernel_source(runs, bad)


def test_metadata_is_private_gpu_and_neutral(tmp_path, settings):
    meta = push.kernel_metadata("someone", "nyaya-search-embed", ["someone/ds"])
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_type"] == "script"
    assert meta["title"] == "nyaya search embed"  # slugifies back to the id
    ds = push.dataset_metadata("someone", push.dataset_slug([2025]), [2025])
    assert ds["id"] == "someone/nyaya-search-chunks-2025"
    assert ds["licenses"] == [{"name": "CC-BY-4.0"}]
    for text in (json.dumps(meta), json.dumps(ds)):
        low = text.lower()
        assert "claude" not in low and "anthropic" not in low and " ai " not in low
    assert push.dataset_slug([2024, 2025, 2026]) == "nyaya-search-chunks-2024-2026"
    assert push.dataset_slug([2020, 2025]) == "nyaya-search-chunks-2020-2025"
    folder = push.stage_kernel(tmp_path / "k", "someone", "nyaya-search-embed", ["someone/ds"], [])
    assert (folder / "embed_corpus.py").exists()
    assert (
        json.loads((folder / "kernel-metadata.json").read_text())["code_file"] == "embed_corpus.py"
    )


def test_stage_dataset_copies_chunk_files(tmp_path):
    build = tmp_path / "build"
    write_chunks(build / "chunks" / "2025.jsonl")
    folder = push.stage_dataset(build, [2025], tmp_path / "ds", "someone", "slug-2025")
    assert (folder / "chunks" / "2025.jsonl").read_bytes() == (
        build / "chunks" / "2025.jsonl"
    ).read_bytes()
    assert json.loads((folder / "dataset-metadata.json").read_text())["id"] == "someone/slug-2025"
    with pytest.raises(push.KaggleSetupError, match="2024"):
        push.stage_dataset(build, [2024], tmp_path / "ds2", "someone", "x")


def _clear_kaggle_env(monkeypatch):
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def test_credentials_token_legacy_and_missing(tmp_path, monkeypatch, capsys):
    _clear_kaggle_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("KAGGLE_USERNAME=someone\nKAGGLE_KEY=KGAT_secretvalue123\n", encoding="utf-8")
    assert push.kaggle_credentials(env) == {"KAGGLE_API_TOKEN": "KGAT_secretvalue123"}
    env.write_text("KAGGLE_USERNAME=someone\nKAGGLE_KEY=0123abcd\n", encoding="utf-8")
    assert push.kaggle_credentials(env) == {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "0123abcd"}
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_fromenv")
    assert push.kaggle_credentials(env) == {"KAGGLE_API_TOKEN": "KGAT_fromenv"}
    _clear_kaggle_env(monkeypatch)
    with pytest.raises(push.KaggleSetupError, match="no Kaggle credentials"):
        push.kaggle_credentials(None)
    out = capsys.readouterr()
    assert "secretvalue" not in out.out + out.err and "0123abcd" not in out.out + out.err


def test_main_without_credentials_exits_2_and_prints_no_secret(tmp_path, monkeypatch, capsys):
    _clear_kaggle_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("KAGGLE_USERNAME=someone\n", encoding="utf-8")
    assert push.main(["--env-file", str(env), "status", "--kernel", "a/b"]) == 2
    assert "KAGGLE_API_TOKEN" in capsys.readouterr().err


def test_secrets_hold_kaggle_values():
    s = Secrets(_env_file=None, kaggle_key="k", kaggle_api_token="t")
    assert "k" not in repr(s.kaggle_key) and s.kaggle_api_token.get_secret_value() == "t"


class FakeStatus:
    def __init__(self, name, msg=""):
        self.status = type("E", (), {"name": name})()
        self.failure_message = msg


class FakeApi:
    def __init__(self, states):
        self.states = list(states)
        self.pushed = None
        self.downloaded = None

    def kernels_status(self, kernel):
        return FakeStatus(self.states.pop(0))

    def kernels_push(self, folder, timeout=None):
        folder = Path(folder)
        self.pushed = {
            "meta": json.loads((folder / "kernel-metadata.json").read_text()),
            "source": (folder / "embed_corpus.py").read_text(encoding="utf-8"),
            "timeout": timeout,
        }
        return type("R", (), {"error": None, "version_number": 3})()

    def kernels_output(self, kernel, path, force=False, quiet=True):
        self.downloaded = (kernel, path)
        return ["a.npy", "b.txt"], ""


def test_poll_until_done(capsys):
    api = FakeApi(["QUEUED", "RUNNING", "COMPLETE"])
    naps = []
    assert push.poll(api, "o/k", wait=True, interval=5, sleep=naps.append) == "complete"
    assert naps == [5, 5]
    assert push.poll(FakeApi(["RUNNING"]), "o/k", wait=False, interval=5) == "running"
    assert push.status_text(FakeStatus("CANCEL_ACKNOWLEDGED")) in push.DONE_STATES
    assert push.status_text("KernelWorkerStatus.ERROR") == "error"


def test_push_kernel_and_download(tmp_path, settings):
    api = FakeApi([])
    args = push.build_parser().parse_args(
        [
            "kernel",
            "--dataset",
            "someone/nyaya-search-chunks-2025",
            "--model",
            "BAAI/bge-m3",
            "--years",
            "2025",
            "--batch",
            "8",
            "--timeout",
            "3600",
        ]
    )
    assert push.push_kernel(api, "someone", args, settings) == "someone/nyaya-search-embed"
    meta = api.pushed["meta"]
    assert meta["dataset_sources"] == ["someone/nyaya-search-chunks-2025"] and meta["is_private"]
    assert api.pushed["timeout"] == "3600"
    assert (
        '"BAAI/bge-m3"' in api.pushed["source"]
        and '"--batch",\n        "8"' in api.pushed["source"]
    )
    assert push.download(api, "someone/nyaya-search-embed", tmp_path / "o") == ["a.npy", "b.txt"]


def test_push_parser_arguments():
    p = push.build_parser()
    a = p.parse_args(["dataset", "--years", "2024-2025"])
    assert a.years == [2024, 2025] and a.build is None and a.slug is None
    a = p.parse_args(["kernel", "--dataset", "o/d"])
    assert a.model == [] and a.slug == "nyaya-search-embed" and a.machine == "NvidiaTeslaT4"
    a = p.parse_args(["status", "--kernel", "o/k", "--wait"])
    assert a.wait and a.interval == 60.0
    a = p.parse_args(["download", "--kernel", "o/k"])
    assert a.out.name == "kaggle_out"
    with pytest.raises(SystemExit):
        p.parse_args(["kernel"])  # --dataset is required


# --- Embedder profiles --------------------------------------------------------------------


def test_embed_profiles_and_bge_m3_embedder(settings):
    names = [p.model for p in enc.embed_profiles(settings)]
    assert names == ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3"]
    st = FakeST()
    e = enc.Embedder(settings, model=st, model_name="BAAI/bge-m3")
    assert e.model_name == "BAAI/bge-m3" and e.query_prefix == "" and e.dim == 1024
    assert e.profile.padding_side == "right"
    e.embed_query("anticipatory bail")
    assert st.batches == [1]
    q = enc.Embedder(settings, model=FakeST())
    assert q.model_name == settings.embed.model and q.profile.padding_side == "left"
    assert q.query_prefix == settings.embed.query_prefix
    with pytest.raises(KeyError):
        enc.embed_profile(settings, "no/such-model")


def test_bge_m3_can_be_the_configured_model(settings):
    alt = settings.embed.alternatives["BAAI/bge-m3"]
    embed = settings.embed.model_copy(
        update={
            "model": "BAAI/bge-m3",
            "revision": alt.revision,
            "dim": alt.dim,
            "query_prefix": alt.query_prefix,
        }
    )
    s = settings.model_copy(update={"embed": embed})
    p = enc.embed_profile(s)
    assert (p.model, p.query_prefix, p.padding_side) == ("BAAI/bge-m3", "", "right")
    assert [x.model for x in enc.embed_profiles(s)] == ["BAAI/bge-m3"]
