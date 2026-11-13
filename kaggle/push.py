"""Kaggle helpers: upload chunk files, push the embedding kernel, poll it, fetch its output.

Uses the official `kaggle` package (`uv sync --extra kaggle`). Credentials are
KAGGLE_USERNAME and KAGGLE_KEY, read from the environment or from `.env`
(through nyaya.config.Secrets); they are passed to the Kaggle client and never
printed or logged. Every dataset and kernel this creates is private.

    uv run python kaggle/push.py dataset --years 2025
    uv run python kaggle/push.py kernel --dataset <user>/nyaya-search-chunks-2025
    uv run python kaggle/push.py status --kernel <user>/nyaya-search-embed --wait
    uv run python kaggle/push.py download --kernel <user>/nyaya-search-embed --out kaggle_out

See kaggle/README.md for the whole round trip.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = Path(__file__).resolve().with_name("embed_corpus.py")
KERNEL_FILE = "embed_corpus.py"
DEFAULT_KERNEL_SLUG = "nyaya-search-embed"
DEFAULT_MACHINE = "NvidiaTeslaT4"  # a free-tier GPU
DATASET_LICENSE = "CC-BY-4.0"  # the judgments and metadata are CC BY 4.0
DONE_STATES = {"complete", "error", "cancel_requested", "cancel_acknowledged"}
# Kaggle's preinstalled packages vary; pin the versions in uv.lock instead.
KERNEL_PIP = ("sentence-transformers==6.3.0", "transformers==5.22.2")
KERNEL_RUNS_LINE = "KERNEL_RUNS: list[list[str]] = []"


class KaggleSetupError(RuntimeError):
    pass


# --- credentials -------------------------------------------------------------------------


ACCESS_TOKEN_PREFIX = "KGAT_"  # Kaggle API tokens made since 2025; older keys are 32 hex digits


def kaggle_credentials(env_file: Path | None) -> dict[str, str]:
    """Environment variables for the Kaggle client, from the environment or `env_file`.

    An API token (KAGGLE_API_TOKEN, or a KAGGLE_KEY starting "KGAT_") is passed
    as KAGGLE_API_TOKEN; a legacy key as KAGGLE_USERNAME plus KAGGLE_KEY.
    """
    from nyaya.config import Secrets

    s = Secrets(_env_file=env_file)  # type: ignore[call-arg]  # the environment wins
    user = s.kaggle_username or ""
    key = s.kaggle_key.get_secret_value() if s.kaggle_key else ""
    token = s.kaggle_api_token.get_secret_value() if s.kaggle_api_token else ""
    if not token and key.startswith(ACCESS_TOKEN_PREFIX):
        token = key
    if token:
        return {"KAGGLE_API_TOKEN": token}
    if user and key:
        return {"KAGGLE_USERNAME": user, "KAGGLE_KEY": key}
    raise KaggleSetupError(
        "no Kaggle credentials: set KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY,"
        " in the environment or --env-file"
    )


def kaggle_api(env_file: Path | None) -> tuple[Any, str]:
    """An authenticated KaggleApi and the account's username. Secrets are never printed."""
    os.environ.update(kaggle_credentials(env_file))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise KaggleSetupError(f"cannot import kaggle ({exc}): uv sync --extra kaggle") from exc
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value(api.CONFIG_NAME_USER)
    if not user:
        raise KaggleSetupError("Kaggle authentication gave no username (check the credentials)")
    return api, str(user)


# --- dataset ---------------------------------------------------------------------------


def dataset_slug(years: Sequence[int]) -> str:
    if not years:
        raise ValueError("no years")
    ys = sorted(set(years))
    if len(ys) == 1:
        part = str(ys[0])
    elif ys == list(range(ys[0], ys[-1] + 1)):
        part = f"{ys[0]}-{ys[-1]}"
    else:
        part = "-".join(str(y) for y in ys)
    return f"nyaya-search-chunks-{part}"


def dataset_metadata(owner: str, slug: str, years: Sequence[int]) -> dict[str, Any]:
    title = slug.replace("-", " ")
    return {
        "title": title[:50],
        "id": f"{owner}/{slug}",
        "licenses": [{"name": DATASET_LICENSE}],
        "subtitle": "Chunk files of the Nyaya Search build, for embedding",
        "description": (
            "Chunk JSONL files (`chunks/<year>.jsonl`, one ChunkRecord per line) of "
            "Supreme Court of India judgments, decision years "
            + ", ".join(str(y) for y in sorted(years))
            + ". Source: the Indian Supreme Court Judgments dataset (CC BY 4.0)."
        ),
    }


def stage_dataset(build: Path, years: Sequence[int], folder: Path, owner: str, slug: str) -> Path:
    """Copy chunks/<year>.jsonl into `folder` and write dataset-metadata.json."""
    src = build / "chunks"
    missing = [y for y in years if not (src / f"{y}.jsonl").exists()]
    if missing:
        raise KaggleSetupError(f"no chunk files in {src} for years {missing}")
    dst = folder / "chunks"
    dst.mkdir(parents=True, exist_ok=True)
    for y in years:
        shutil.copy2(src / f"{y}.jsonl", dst / f"{y}.jsonl")
    meta = dataset_metadata(owner, slug, years)
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def _dataset_exists(api: Any, ref: str) -> bool:
    try:
        api.dataset_status(ref)
    except Exception:
        return False
    return True


def push_dataset(api: Any, owner: str, args: argparse.Namespace) -> str:
    years = args.years
    slug = args.slug or dataset_slug(years)
    ref = f"{owner}/{slug}"
    with tempfile.TemporaryDirectory(prefix="nyaya-kaggle-ds-") as tmp:
        folder = stage_dataset(Path(args.build), years, Path(tmp), owner, slug)
        if _dataset_exists(api, ref):
            print(f"dataset {ref} exists: uploading a new version")
            api.dataset_create_version(
                str(folder), version_notes=args.notes, quiet=False, dir_mode="zip"
            )
        else:
            print(f"creating private dataset {ref}")
            api.dataset_create_new(str(folder), public=False, quiet=False, dir_mode="zip")
    print(f"dataset: {ref}")
    return ref


# --- kernel ----------------------------------------------------------------------------


def run_args(profile: Any, extra: Sequence[str] = ()) -> list[str]:
    """embed_corpus.py arguments for one EmbedProfile."""
    out = ["--model", profile.model, "--revision", profile.revision, "--dim", str(profile.dim)]
    out += list(extra)
    return out


def kernel_source(runs: Sequence[Sequence[str]], source: Path = KERNEL_SOURCE) -> str:
    """embed_corpus.py with KERNEL_RUNS set to `runs`."""
    text = source.read_text(encoding="utf-8")
    if text.count(KERNEL_RUNS_LINE) != 1:
        raise KaggleSetupError(f"{source}: expected one line {KERNEL_RUNS_LINE!r}")
    value = json.dumps([list(r) for r in runs], indent=4)
    return text.replace(KERNEL_RUNS_LINE, f"KERNEL_RUNS: list[list[str]] = {value}")


def kernel_metadata(
    owner: str, slug: str, datasets: Sequence[str], machine: str = DEFAULT_MACHINE
) -> dict[str, Any]:
    return {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": KERNEL_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,  # model weights and pinned packages come from the internet
        "machine_shape": machine,
        "dataset_sources": list(datasets),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def stage_kernel(
    folder: Path,
    owner: str,
    slug: str,
    datasets: Sequence[str],
    runs: Sequence[Sequence[str]],
    machine: str = DEFAULT_MACHINE,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / KERNEL_FILE).write_text(kernel_source(runs), encoding="utf-8", newline="\n")
    meta = kernel_metadata(owner, slug, datasets, machine)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def kernel_runs(settings: Any, models: Sequence[str], extra: Sequence[str]) -> list[list[str]]:
    from nyaya.models.encoders import embed_profile, embed_profiles

    profiles = [embed_profile(settings, m) for m in models] if models else embed_profiles(settings)
    pip = [a for req in KERNEL_PIP for a in ("--pip", req)]
    return [run_args(p, [*pip, *extra]) for p in profiles]


def push_kernel(api: Any, owner: str, args: argparse.Namespace, settings: Any) -> str:
    extra: list[str] = []
    if args.years:
        extra += ["--years", ",".join(str(y) for y in args.years)]
    if args.batch:
        extra += ["--batch", str(args.batch)]
    if args.token_budget:
        extra += ["--token-budget", str(args.token_budget)]
    runs = kernel_runs(settings, args.model, extra)
    ref = f"{owner}/{args.slug}"
    with tempfile.TemporaryDirectory(prefix="nyaya-kaggle-kn-") as tmp:
        folder = stage_kernel(Path(tmp), owner, args.slug, args.dataset, runs, args.machine)
        resp = api.kernels_push(str(folder), timeout=args.timeout)
    err = getattr(resp, "error", None)
    if err:
        raise KaggleSetupError(f"kernel push failed: {err}")
    print(f"kernel: {ref} (version {getattr(resp, 'version_number', '?')})")
    for r in runs:
        print("  run:", r[1])
    return ref


# --- status and output -----------------------------------------------------------------


def status_text(resp: Any) -> str:
    """'complete', 'running', 'error', ... from a kernels_status response."""
    s = getattr(resp, "status", resp)
    s = getattr(s, "name", s)
    return re.sub(r"^kernelworkerstatus\.", "", str(s).lower()).strip()


def poll(api: Any, kernel: str, *, wait: bool, interval: float, sleep: Any = time.sleep) -> str:
    while True:
        resp = api.kernels_status(kernel)
        state = status_text(resp)
        msg = getattr(resp, "failure_message", "") or ""
        print(f"{time.strftime('%H:%M:%S')} {kernel}: {state}" + (f" ({msg})" if msg else ""))
        if not wait or state in DONE_STATES:
            return state
        sleep(interval)


def download(api: Any, kernel: str, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files, _ = api.kernels_output(kernel, str(out), force=True, quiet=True)
    names = [str(f) for f in files or []]
    print(f"downloaded {len(names)} files to {out}")
    return names


# --- CLI -------------------------------------------------------------------------------


def _years(text: str) -> list[int]:
    from nyaya.build import parse_years

    return parse_years(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kaggle/push.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--env-file",
        type=Path,
        default=REPO / ".env",
        help="where KAGGLE_USERNAME / KAGGLE_KEY are read if not in the environment",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("dataset", help="create or update a private dataset of chunk files")
    d.add_argument("--years", type=_years, required=True)
    d.add_argument("--build", type=Path, default=None, help="build directory (default: config)")
    d.add_argument("--slug", default=None, help="default: nyaya-search-chunks-<years>")
    d.add_argument("--notes", default="chunk files update")

    k = sub.add_parser("kernel", help="push and run the embedding kernel (private, GPU)")
    k.add_argument("--dataset", action="append", required=True, help="<owner>/<slug> (repeatable)")
    k.add_argument(
        "--model",
        action="append",
        default=[],
        help="embed.model or one of embed.alternatives (repeatable; default: all)",
    )
    k.add_argument("--slug", default=DEFAULT_KERNEL_SLUG)
    k.add_argument("--years", type=_years, default=None, help="default: every attached file")
    k.add_argument("--batch", type=int, default=None)
    k.add_argument("--token-budget", type=int, default=None)
    k.add_argument("--machine", default=DEFAULT_MACHINE, help="Kaggle machine shape")
    k.add_argument("--timeout", default=None, help="session timeout in seconds")

    s = sub.add_parser("status", help="print a kernel's status")
    s.add_argument("--kernel", required=True, help="<owner>/<slug>")
    s.add_argument("--wait", action="store_true", help="poll until it finishes")
    s.add_argument("--interval", type=float, default=60.0)

    o = sub.add_parser("download", help="download a kernel's output files")
    o.add_argument("--kernel", required=True)
    o.add_argument("--out", type=Path, default=REPO / "kaggle_out")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from nyaya.config import load_settings

    settings = load_settings(env_file=None)
    try:
        api, owner = kaggle_api(args.env_file if args.env_file.exists() else None)
    except KaggleSetupError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "dataset":
        if args.build is None:
            b = Path(settings.paths.build)
            args.build = b if b.is_absolute() else REPO / b
        push_dataset(api, owner, args)
    elif args.command == "kernel":
        push_kernel(api, owner, args, settings)
    elif args.command == "status":
        state = poll(api, args.kernel, wait=args.wait, interval=args.interval)
        return 0 if state != "error" else 1
    elif args.command == "download":
        download(api, args.kernel, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
