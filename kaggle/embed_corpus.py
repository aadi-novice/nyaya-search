"""Embed Nyaya Search chunk files on a Kaggle GPU (design file 28).

Self-contained: it needs numpy, torch and sentence-transformers, not the
`nyaya` package. It reads `chunks/<year>.jsonl` from an attached Kaggle
dataset and writes, per model, the files `nyaya-build embed` writes:

    <out>/<model-dir>/vectors/<year>.npy          float32 (n, dim), chunks file order
    <out>/<model-dir>/vectors/<year>.ids.txt      the n chunk IDs, one per line
    <out>/<model-dir>/vectors/<year>.sha1.txt     SHA-1 of each embedded text
    <out>/<model-dir>/state/embed_<year>.json     the embed stamp (model, revision, dim, rows)
    <out>/<model-dir>/vectors/reference/passages.json   5 chunks, for startup_check
    <out>/<model-dir>/vectors/reference/vectors.npy     their vectors
    <out>/<model-dir>/embed_report.json           device, versions, time, chunks/s, VRAM

`<model-dir>` is the model name with "/" replaced by "__". Copy (or point
`paths.build` at) a model directory and `nyaya-build load` reads it as is.

Documents are embedded as `nyaya.models.encoders.document_text()` builds them
(header, a blank line, the text; no prefix), normalised, fp16 on CUDA. The
small helpers below are copies of the `nyaya` ones; tests/test_kaggle.py
proves they match.

Run on Kaggle (`kaggle/push.py kernel` fills KERNEL_RUNS), or by hand:

    python embed_corpus.py --model Qwen/Qwen3-Embedding-0.6B \\
        --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --dim 1024
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("embed_corpus")

# Filled in by `kaggle/push.py kernel`: one argument list per model, run in order.
KERNEL_RUNS: list[list[str]] = []

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
REFERENCE_COUNT = 5

# --- copies of nyaya helpers (tests/test_kaggle.py checks each against the original) ---

# nyaya.models.encoders.PADDING_SIDE / DEFAULT_PADDING_SIDE
PADDING_SIDE = {"BAAI/bge-m3": "right"}
DEFAULT_PADDING_SIDE = "left"


def padding_side(model_name: str) -> str:
    return PADDING_SIDE.get(model_name, DEFAULT_PADDING_SIDE)


def document_text(header: str, text: str) -> str:
    """nyaya.models.encoders.document_text: header, a blank line, the text."""
    return f"{header}\n\n{text}" if header else text


def text_hash(text: str) -> str:
    """nyaya.build.text_hash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def token_batches(lengths: Sequence[int], budget: int, max_batch: int) -> list[list[int]]:
    """nyaya.build.token_batches: longest first, batch size x longest length <= budget."""
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


def embed_stamp(
    model: str, revision: str, dim: int, ids: Sequence[str], hashes: Sequence[str]
) -> dict[str, Any]:
    """nyaya.build.embed_stamp, from explicit values instead of Settings."""
    rows = "".join(f"{i} {h}\n" for i, h in zip(ids, hashes, strict=True))
    return {
        "model": model,
        "revision": revision,
        "dim": dim,
        "n": len(ids),
        "rows_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest()[:16],
    }


def read_chunks(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """(chunk IDs, embedded texts, raw {chunk_id, header, text}) of one chunks file.

    nyaya.build.read_chunk_texts, with plain JSON instead of ChunkRecord.
    """
    ids: list[str] = []
    texts: list[str] = []
    raw: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                cid, header, text = rec["chunk_id"], rec["header"], rec["text"]
            except KeyError as exc:
                raise ValueError(f"{path}:{n}: missing {exc}") from None
            ids.append(cid)
            texts.append(document_text(header, text))
            raw.append({"chunk_id": cid, "header": header, "text": text})
    return ids, texts, raw


# --- files ---------------------------------------------------------------------------


def model_dir_name(model: str) -> str:
    return model.replace("/", "__")


def find_chunk_files(root: Path, years: Sequence[int] | None) -> dict[int, Path]:
    """{year: path} of `<year>.jsonl` files under `root`, preferring a `chunks/` folder."""
    found: dict[int, Path] = {}
    for p in sorted(root.rglob("*.jsonl"), key=lambda p: (p.parent.name != "chunks", str(p))):
        try:
            year = int(p.stem)
        except ValueError:
            continue
        if (years is None or year in years) and year not in found:
            found[year] = p
    missing = sorted(set(years or []) - set(found))
    if missing:
        raise SystemExit(f"no chunk files under {root} for years {missing}")
    if not found:
        raise SystemExit(f"no <year>.jsonl chunk files under {root}")
    return dict(sorted(found.items()))


def reference_indexes(n: int, count: int = REFERENCE_COUNT) -> list[int]:
    """`count` row indexes spread evenly over n rows (fewer if n is small)."""
    if n <= count:
        return list(range(n))
    return sorted({round(k * (n - 1) / (count - 1)) for k in range(count)})


def save_npy(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, a)
    os.replace(tmp, path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def save_json(path: Path, obj: Any) -> None:
    save_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_year(
    out: Path,
    year: int,
    ids: Sequence[str],
    texts: Sequence[str],
    vectors: np.ndarray,
    *,
    model: str,
    revision: str,
    dim: int,
) -> dict[str, Any]:
    """The nyaya-build embed files of one year under the build directory `out`."""
    if vectors.shape != (len(ids), dim):
        raise ValueError(f"vectors have shape {vectors.shape}, expected ({len(ids)}, {dim})")
    hashes = [text_hash(t) for t in texts]
    vdir = out / "vectors"
    save_npy(vdir / f"{year}.npy", vectors.astype(np.float32, copy=False))
    save_text(vdir / f"{year}.ids.txt", "".join(i + "\n" for i in ids))
    save_text(vdir / f"{year}.sha1.txt", "".join(h + "\n" for h in hashes))
    stamp = embed_stamp(model, revision, dim, ids, hashes)
    save_json(out / "state" / f"embed_{year}.json", stamp)
    return stamp


def write_reference(
    out: Path, raw: Sequence[dict[str, str]], vectors: np.ndarray, count: int = REFERENCE_COUNT
) -> list[int]:
    """vectors/reference/passages.json and vectors.npy, read by startup_check()."""
    idx = reference_indexes(len(raw), count)
    ref = out / "vectors" / "reference"
    save_json(ref / "passages.json", [raw[i] for i in idx])
    save_npy(ref / "vectors.npy", vectors[idx].astype(np.float32, copy=False))
    return idx


# --- encoding ---------------------------------------------------------------------------


def load_model(model: str, revision: str, device: str) -> Any:
    """SentenceTransformer at fp16 with the model's padding side, as nyaya's Embedder."""
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model,
        revision=revision or None,
        device=device,
        model_kwargs={"dtype": torch.float16},
        processor_kwargs={"padding_side": padding_side(model)},
    )


def encode_texts(
    st: Any,
    texts: Sequence[str],
    *,
    dim: int,
    batch: int,
    budget: int,
    count_tokens: Any = None,
    progress_every: int = 50,
) -> tuple[np.ndarray, float]:
    """(normalised float32 vectors in input order, seconds), batched by token budget."""
    if count_tokens is None:
        tok = st.tokenizer

        def count_tokens(s: str) -> int:
            return len(tok(s, add_special_tokens=True)["input_ids"])

    lengths = [count_tokens(t) for t in texts]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    batches = token_batches(lengths, budget, batch)
    t0 = time.perf_counter()
    done = 0
    for n, idx in enumerate(batches, 1):
        v = st.encode(
            [texts[i] for i in idx],
            batch_size=len(idx),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (len(idx), dim):
            raise ValueError(f"model returned shape {v.shape}, expected ({len(idx)}, {dim})")
        out[idx] = v
        done += len(idx)
        if n % progress_every == 0 or n == len(batches):
            el = time.perf_counter() - t0
            log.info("%d/%d chunks, %.0f s, %.1f chunks/s", done, len(texts), el, done / el)
    return out, time.perf_counter() - t0


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:  # pragma: no cover - Kaggle always has torch
        return {"cuda": False}
    if not torch.cuda.is_available():
        return {"cuda": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "total_vram_gb": round(props.total_memory / 2**30, 2),
    }


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("sentence_transformers", "transformers", "torch"):
        with contextlib.suppress(ImportError):
            out[name] = __import__(name).__version__
    return out


def ensure_packages(requirements: Sequence[str]) -> None:
    """pip install `requirements` without letting pip replace the installed torch."""
    if not requirements:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *requirements]
    try:
        import torch

        cmd.append(f"torch=={torch.__version__}")
    except ImportError:  # pragma: no cover
        pass
    log.info("pip install %s", " ".join(cmd[5:]))
    subprocess.run(cmd, check=True)


# --- run ------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embed_corpus.py",
        description="embed Nyaya Search chunk files on a GPU, in the nyaya-build embed format",
    )
    p.add_argument("--model", required=True, help="Hugging Face model name")
    p.add_argument("--revision", default="", help="Hugging Face commit (default: main)")
    p.add_argument("--dim", type=int, required=True, help="vector dimension, checked")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="folder searched for <year>.jsonl (default: /kaggle/input)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output root; each model writes <output>/<model-dir> (default: /kaggle/working)",
    )
    p.add_argument("--years", default="", help="e.g. 2025 or 2024,2025 (default: every file)")
    p.add_argument("--batch", type=int, default=32, help="max chunks per forward pass")
    p.add_argument(
        "--token-budget", type=int, default=32 * 1024, help="max batch size x longest tokens"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--pip",
        action="append",
        default=[],
        metavar="REQ",
        help="pip requirement installed first (repeatable), e.g. sentence-transformers==6.3.0",
    )
    return p


def parse_years(text: str) -> list[int] | None:
    if not text.strip():
        return None
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise argparse.ArgumentTypeError(f"bad year range {part!r}")
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def run(args: argparse.Namespace, *, model_loader: Any = None, count_tokens: Any = None) -> dict:
    """Embed every chunks file for one model; returns the run report."""
    years = parse_years(args.years)
    root = args.input or KAGGLE_INPUT
    out = (args.output or KAGGLE_WORKING) / model_dir_name(args.model)
    files = find_chunk_files(root, years)
    ensure_packages(args.pip)
    info = _cuda_info() if args.device.startswith("cuda") else {"cuda": False}
    if args.device.startswith("cuda") and not info["cuda"]:
        raise SystemExit("CUDA is not available: turn on a GPU accelerator in the notebook")
    log.info("model %s @ %s on %s", args.model, args.revision or "main", info)

    import_t0 = time.perf_counter()
    st = (model_loader or load_model)(args.model, args.revision, args.device)
    load_s = time.perf_counter() - import_t0
    cuda = None
    if info["cuda"]:
        import torch

        cuda = torch.cuda
        cuda.reset_peak_memory_stats()

    report: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "dim": args.dim,
        "padding_side": padding_side(args.model),
        "batch": args.batch,
        "token_budget": args.token_budget,
        "load_seconds": round(load_s, 1),
        "environment": {**info, **_versions()},
        "years": {},
    }
    reference_done = False
    total_n, total_s = 0, 0.0
    for year, path in files.items():
        ids, texts, raw = read_chunks(path)
        log.info("%s %d: %d chunks from %s", args.model, year, len(ids), path)
        vecs, seconds = encode_texts(
            st,
            texts,
            dim=args.dim,
            batch=args.batch,
            budget=args.token_budget,
            count_tokens=count_tokens,
        )
        stamp = write_year(
            out, year, ids, texts, vecs, model=args.model, revision=args.revision, dim=args.dim
        )
        if not reference_done:
            write_reference(out, raw, vecs)
            reference_done = True
        total_n += len(ids)
        total_s += seconds
        report["years"][str(year)] = {
            "chunks": len(ids),
            "seconds": round(seconds, 1),
            "chunks_per_s": round(len(ids) / seconds, 2) if seconds else None,
            "rows_sha256": stamp["rows_sha256"],
        }
    report["chunks"] = total_n
    report["seconds"] = round(total_s, 1)
    report["chunks_per_s"] = round(total_n / total_s, 2) if total_s else None
    if cuda is not None:
        report["peak_vram_allocated_gb"] = round(cuda.max_memory_allocated() / 2**30, 2)
        report["peak_vram_reserved_gb"] = round(cuda.max_memory_reserved() / 2**30, 2)
    save_json(out / "embed_report.json", report)
    log.info("%s: %s", args.model, json.dumps({k: v for k, v in report.items() if k != "years"}))
    del st
    if cuda is not None:
        import gc

        gc.collect()
        cuda.empty_cache()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if argv is not None:
        runs = [list(argv)]
    else:
        # a notebook passes its own flags (e.g. -f kernel.json); a script on Kaggle has none
        cli = [a for a in sys.argv[1:] if a.startswith("--")]
        runs = [sys.argv[1:]] if cli else KERNEL_RUNS
    if not runs:
        build_parser().print_usage()
        print("no arguments and KERNEL_RUNS is empty", file=sys.stderr)
        return 2
    parser = build_parser()
    for r in runs:
        run(parser.parse_args(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
