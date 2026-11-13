"""`nyaya-eval`: the evaluation command dispatcher (design file 39).

Subcommands and the phase in which each is first needed:

    build-devset    Phase 1   development questions from Held points
    g6-holdout      Phase 2   (not built yet)
    retrieval       Phase 2   recall@5 / recall@50 / MRR per method (G3)
    size-steps      Phase 2   G4 size steps in temporary schemas
    bm25-trigger    Phase 2   Postgres full-text search against bm25s
    tune-threshold  Phase 2   refusal thresholds (G5)
    answers, latency, judge-validate, lawyer-export, lawyer-import
                    Phase 3   (not built yet)

Exit codes: 0 success; 1 a gate failed (size-steps); 2 usage or setup error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nyaya.config import ConfigError, Settings, load_settings
from nyaya.eval.datasets import DATASETS_DIR, DevItem, read_jsonl
from nyaya.eval.retrieval import METHODS

NOT_BUILT = {
    "g6-holdout": "Phase 2",
    "answers": "Phase 3",
    "latency": "Phase 3",
    "judge-validate": "Phase 3",
    "lawyer-export": "Phase 3",
    "lawyer-import": "Phase 3",
}


def parse_years(text: str) -> list[int]:
    """'2025', '2010-2026' or '2024,2025'."""
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if a > b:
                raise argparse.ArgumentTypeError(f"bad year range {part!r}")
            years.extend(range(a, b + 1))
        elif part:
            years.append(int(part))
    if not years:
        raise argparse.ArgumentTypeError("no years given")
    return sorted(set(years))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nyaya-eval", description="Nyaya Search evaluation")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-devset", help="draft dev_v1.jsonl from Held points (Phase 1)")
    b.add_argument("--years", type=parse_years, default=parse_years("2010-2026"))
    b.add_argument("--build-dir", type=Path, default=None, help="default: paths.build")
    b.add_argument("--out", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    b.add_argument("--size", type=int, default=None, help="default: devset.size")
    b.add_argument("--retries", type=int, default=1)

    r = sub.add_parser("retrieval", help="G3 retrieval metrics per method")
    r.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    r.add_argument("--methods", default=",".join(METHODS))
    r.add_argument("--index-version", default=None, help="default: the active version")
    r.add_argument("--reranker", choices=["qwen", "cohere"], default="qwen")

    s = sub.add_parser("size-steps", help="G4: metrics at nested index sizes")
    s.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    s.add_argument("--targets", default="10000,50000,100000,full", help="chunk counts")
    s.add_argument("--source-version", default=None, help="default: the active version")
    s.add_argument("--methods", default=",".join(METHODS))
    s.add_argument("--keep-schemas", action="store_true")

    t = sub.add_parser("bm25-trigger", help="Postgres full-text search against bm25s")
    t.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    t.add_argument("--index-version", default=None)

    h = sub.add_parser("tune-threshold", help="G5: refusal thresholds per reranker")
    h.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    h.add_argument("--unanswerable", type=Path, default=DATASETS_DIR / "unanswerable_v1.jsonl")
    h.add_argument("--rerankers", default="qwen,cohere")
    h.add_argument("--dry-run", action="store_true", help="report only; do not write")

    for name, phase in NOT_BUILT.items():
        sub.add_parser(name, help=f"{phase} (not built yet)")
    return p


# --- commands ----------------------------------------------------------------------------


def cmd_build_devset(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.datasets import write_jsonl
    from nyaya.eval.devset_from_held import (
        OpenAICompatLLM,
        draft_devset,
        load_candidates_from_build,
        write_handcheck_csv,
    )

    sec = settings.secrets
    if not (sec.llm_base_url and sec.llm_api_key and sec.llm_model):
        print("build-devset: LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required", file=sys.stderr)
        return 2
    ds = settings.devset
    build_dir = args.build_dir or Path(settings.paths.build)
    cands = load_candidates_from_build(build_dir, args.years)
    llm = OpenAICompatLLM(sec.llm_base_url, sec.llm_api_key.get_secret_value(), sec.llm_model)
    result = draft_devset(
        cands,
        llm,
        size=args.size or ds.size,
        max_shared_words=ds.max_shared_words,
        seed=ds.seed,
        retries=args.retries,
    )
    write_jsonl(args.out, result.items)
    handcheck = write_handcheck_csv(
        args.out.with_name(args.out.stem + "_handcheck.csv"), result, ds.handcheck_size, ds.seed
    )
    _print(
        {
            "candidates": len(cands),
            "kept": len(result.items),
            "rejected": len(result.rejected),
            "llm_calls": result.llm_calls,
            "aborted": result.aborted,
            "out": str(args.out),
            "handcheck": str(handcheck),
        }
    )
    return 2 if result.aborted else 0


def _connect(settings: Settings) -> Any:
    from nyaya.store.db import connect

    return connect(settings, autocommit=True)


def cmd_retrieval(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.retrieval import evaluate
    from nyaya.eval.runs import save_run
    from nyaya.eval.wiring import active_version, make_encoders, make_engine_retriever

    items = read_jsonl(args.dataset, DevItem)
    methods = _methods(args.methods)
    with _connect(settings) as conn:
        version = args.index_version or active_version(conn)
        embedder, reranker = make_encoders(settings, args.reranker)
        retriever, engine = make_engine_retriever(
            settings, index_version=version, embedder=embedder, reranker=reranker
        )
        report = evaluate(items, retriever, engine, methods)
        metrics = {"summary": report.summary(), "questions": len(report.item_ids)}
        run_id, path = save_run(
            "retrieval",
            metrics,
            config={"methods": methods, "reranker": args.reranker},
            dataset_version=args.dataset.stem,
            index_version=version,
            conn=conn,
        )
    _print({"run_id": run_id, "file": str(path), **metrics})
    return 0


def cmd_size_steps(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.runs import save_run
    from nyaya.eval.size_steps import parse_targets, run_size_steps
    from nyaya.eval.wiring import active_version, make_encoders, make_engine_retriever

    items = read_jsonl(args.dataset, DevItem)
    embedder, reranker = make_encoders(settings, "qwen")
    with _connect(settings) as conn:
        source = args.source_version or active_version(conn)
        result = run_size_steps(
            conn,
            items,
            source_version=source,
            targets=parse_targets(args.targets),
            make_retriever=lambda name: make_engine_retriever(
                settings, index_version=name, schema=name, embedder=embedder, reranker=reranker
            ),
            seed=settings.eval.seed,
            bootstrap_samples=settings.eval.bootstrap_samples,
            max_drop_points=settings.g4.max_drop_points,
            methods=_methods(args.methods),
            keep_schemas=args.keep_schemas,
        )
        metrics = result.as_dict()
        run_id, path = save_run(
            "size-steps",
            metrics,
            config={"targets": args.targets, "source_version": source},
            dataset_version=args.dataset.stem,
            index_version=source,
            conn=conn,
        )
    _print({"run_id": run_id, "file": str(path), "passed": result.passed})
    return 0 if result.passed else 1


def cmd_bm25_trigger(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.bm25_trigger import Bm25Index, load_corpus, run_bm25_trigger
    from nyaya.eval.runs import save_run
    from nyaya.eval.wiring import active_version
    from nyaya.search.postgres import PostgresSearch

    items = read_jsonl(args.dataset, DevItem)
    with _connect(settings) as conn:
        version = args.index_version or active_version(conn)
        index = Bm25Index(*load_corpus(conn, version))
        engine = PostgresSearch(index_version=version)
        metrics = run_bm25_trigger(
            items,
            engine,
            index,
            k=settings.retrieval.keyword_k,
            samples=settings.eval.bootstrap_samples,
            seed=settings.eval.seed,
            min_gap_points=settings.bm25_trigger.min_gap_points,
        )
        run_id, path = save_run(
            "bm25-trigger",
            metrics,
            config={"k": settings.retrieval.keyword_k},
            dataset_version=args.dataset.stem,
            index_version=version,
            conn=conn,
        )
    _print({"run_id": run_id, "file": str(path), "fires": metrics["fires"]})
    return 0


def cmd_tune_threshold(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.runs import save_run
    from nyaya.eval.threshold import as_dict, tune, write_thresholds
    from nyaya.eval.wiring import active_version, make_encoders, make_engine_retriever

    items = read_jsonl(args.dataset, DevItem) + read_jsonl(args.unanswerable, DevItem)
    names = [n.strip() for n in args.rerankers.split(",") if n.strip()]
    with _connect(settings) as conn:
        version = active_version(conn)
        retrievers = {}
        for name in names:
            embedder, reranker = make_encoders(settings, name)
            retrievers[name], _ = make_engine_retriever(
                settings, index_version=version, embedder=embedder, reranker=reranker
            )
        choices = tune(items, retrievers, settings.refusal.max_wrong_refusal_share)
        written = None if args.dry_run else write_thresholds(conn, choices)
        metrics = {"choices": as_dict(choices), "written_to": written}
        run_id, path = save_run(
            "tune-threshold",
            metrics,
            config={"rerankers": names, "dry_run": args.dry_run},
            dataset_version=f"{args.dataset.stem}+{args.unanswerable.stem}",
            index_version=version,
            conn=conn,
        )
    _print({"run_id": run_id, "file": str(path), **metrics})
    return 0


COMMANDS = {
    "build-devset": cmd_build_devset,
    "retrieval": cmd_retrieval,
    "size-steps": cmd_size_steps,
    "bm25-trigger": cmd_bm25_trigger,
    "tune-threshold": cmd_tune_threshold,
}


def _methods(text: str) -> list[str]:
    methods = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; expected some of {list(METHODS)}")
    return methods


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in NOT_BUILT:
        print(
            f"nyaya-eval {args.command}: not built yet ({NOT_BUILT[args.command]})",
            file=sys.stderr,
        )
        return 2
    try:
        settings = load_settings()
        return COMMANDS[args.command](args, settings)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"nyaya-eval {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
