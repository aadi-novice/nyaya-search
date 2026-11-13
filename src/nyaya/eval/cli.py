"""`nyaya-eval`: the evaluation command dispatcher (design file 39).

Subcommands and the phase in which each is first needed:

    build-devset    Phase 1   development questions from Held points
    g6-holdout      Phase 2   held-out PDFs laid out for the G6 hand check
    retrieval       Phase 2   recall@5 / recall@50 / MRR per method (G3)
    size-steps      Phase 2   G4 size steps in temporary schemas
    bm25-trigger    Phase 2   Postgres full-text search against bm25s
    tune-threshold  Phase 2   refusal thresholds (G5)
    answers         Phase 3   G1, G2, G5, G8, G9 over the dev, unanswerable and
                              adversarial sets (resumable across days)
    latency         Phase 3   G7: p50 / p95 of a fixed query set, warm, one at a time
    judge-validate  Phase 3   Cohen's kappa of the judge against the hand labels
    lawyer-export   Phase 3   a seeded sample of answers to CSV and HTML
    lawyer-import   Phase 3   the lawyer's ratings into eval_runs (never a gate)
    ragas           Phase 5   RAGAS metrics over an answers run, beside ours (never a
                              gate; needs the `ragas` extra; resumable)

Exit codes: 0 success; 1 a gate failed (size-steps, answers, latency,
judge-validate); 2 usage or setup error (g6-holdout: also fewer in-scope PDFs
than asked for). An `answers` session that stops
early (free quota used up, or --limit) exits 0 and prints how to resume.
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

SETS = ("dev", "unanswerable", "adversarial")


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

    g = sub.add_parser("g6-holdout", help="held-out PDFs for the G6 hand check (Phase 2)")
    g.add_argument("--years", type=parse_years, default=parse_years("2010-2026"))
    g.add_argument("--out", type=Path, default=None, help="default: eval/runs/g6_holdout")
    g.add_argument("--size", type=int, default=None, help="default: g6.holdout_size")
    g.add_argument("--dpi", type=int, default=None, help="page image resolution (default 110)")

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

    a = sub.add_parser("answers", help="G1, G2, G5, G8, G9 answer evaluation (Phase 3)")
    a.add_argument("--dev", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    a.add_argument("--unanswerable", type=Path, default=DATASETS_DIR / "unanswerable_v1.jsonl")
    a.add_argument("--adversarial", type=Path, default=DATASETS_DIR / "adversarial_v1.jsonl")
    a.add_argument("--sets", default=",".join(SETS), help=f"some of {','.join(SETS)}")
    a.add_argument("--run-id", default=None, help="resume this run (skips finished questions)")
    a.add_argument("--limit", type=int, default=None, help="at most N new questions this session")
    a.add_argument("--redo-errors", action="store_true", help="redo questions that ended in error")

    lat = sub.add_parser("latency", help="G7: p50 / p95 end-to-end time (Phase 3)")
    lat.add_argument("--dataset", type=Path, default=DATASETS_DIR / "dev_v1.jsonl")
    lat.add_argument("--count", type=int, default=None, help="default: latency.query_count")
    lat.add_argument("--warmup", type=int, default=None, help="default: latency.warmup")

    jv = sub.add_parser("judge-validate", help="Cohen's kappa of the judge (Phase 3)")
    jv.add_argument("--labels", type=Path, default=DATASETS_DIR / "hand_labels_v1.jsonl")

    le = sub.add_parser("lawyer-export", help="sample answers for the lawyer review (Phase 3)")
    le.add_argument("--run-id", default=None, help="the answers run (default: the latest)")
    le.add_argument("--out-dir", type=Path, default=None, help="default: eval/runs/lawyer/<run>")

    li = sub.add_parser("lawyer-import", help="the lawyer's ratings into eval_runs (Phase 3)")
    li.add_argument("--run-id", default=None, help="the answers run (default: the latest)")
    li.add_argument("--csv", type=Path, default=None, help="default: <export dir>/ratings.csv")

    rg = sub.add_parser("ragas", help="RAGAS metrics over an answers run (never a gate)")
    rg.add_argument("--run-id", default=None, help="the answers run (default: the latest)")
    rg.add_argument(
        "--results", type=Path, default=None, help="an answers results.jsonl (instead of --run-id)"
    )
    rg.add_argument(
        "--references",
        type=Path,
        default=None,
        help="Held text per question id (default: <dev set>_handcheck.csv)",
    )
    rg.add_argument("--limit", type=int, default=None, help="at most N new questions this session")
    rg.add_argument(
        "--no-relevancy", action="store_true", help="skip answer relevancy (no embedding model)"
    )
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


def cmd_g6_holdout(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.corpus.manifest import load_manifest
    from nyaya.eval.g6_holdout import DEFAULT_DPI, DEFAULT_OUT, run_holdout

    result = run_holdout(
        load_manifest(settings),
        settings,
        args.years,
        args.out or DEFAULT_OUT,
        size=args.size,
        dpi=args.dpi or DEFAULT_DPI,
    )
    _print(result.summary())
    wanted = settings.g6.holdout_size if args.size is None else args.size
    return 0 if len(result.reviewed) == wanted else 2


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


# --- Phase 3 ------------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _judge_with_pacing(settings: Settings, clock: Any) -> tuple[Any, Any]:
    """The judge, paced to `judge.rpm` with the [answers] 429 backoff; and its wrapper."""
    from nyaya.eval.judge import Judge
    from nyaya.eval.ratelimit import RateLimitedCompleter

    holder: dict[str, Any] = {}
    a = settings.answers

    def wrap(llm: Any) -> Any:
        holder["llm"] = RateLimitedCompleter(
            llm,
            clock=clock,
            rpm=settings.judge.rpm,
            retries=a.rate_limit_retries,
            backoff_base_s=a.backoff_base_s,
            backoff_max_s=a.backoff_max_s,
        )
        return holder["llm"]

    judge = Judge.from_settings(settings, wrap=wrap)
    return judge, holder["llm"]


def cmd_answers(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import UTC, datetime

    from nyaya.eval import answers as A
    from nyaya.eval.ratelimit import PausableClock
    from nyaya.eval.runs import new_run_id
    from nyaya.pipeline import build_deps

    sets = [x.strip() for x in args.sets.split(",") if x.strip()]
    bad = [x for x in sets if x not in SETS]
    if bad or not sets:
        raise ValueError(f"unknown sets {bad}; expected some of {list(SETS)}")
    paths = {
        "dev": args.dev if "dev" in sets else None,
        "unanswerable": args.unanswerable if "unanswerable" in sets else None,
        "adversarial": args.adversarial if "adversarial" in sets else None,
    }
    for name, path in paths.items():
        if path is not None and not path.exists():
            hint = " (until dev_v1 exists, try --dev eval/datasets/dev_draft_v0.jsonl)"
            raise FileNotFoundError(f"{name} set not found: {path}{hint if name == 'dev' else ''}")
    questions = A.load_questions(paths["dev"], paths["unanswerable"], paths["adversarial"])
    datasets = {k: v.stem for k, v in paths.items() if v is not None}

    results = A.ResultsFile(args.run_id or new_run_id("answers"))
    meta = results.read_meta()
    if args.run_id and not meta:
        raise FileNotFoundError(f"no answers run {args.run_id!r} under {A.ANSWERS_DIR}")
    if meta and meta.get("datasets") != datasets:
        raise ValueError(
            f"run {results.run_id} used datasets {meta.get('datasets')}, not {datasets}; "
            "pass the same --dev/--unanswerable/--adversarial/--sets to resume"
        )

    clock = PausableClock()
    judge, judge_llm = _judge_with_pacing(settings, clock)  # fail fast, before the models
    started = datetime.now(UTC)
    deps = build_deps(settings)
    if meta.get("index_version") not in (None, deps.index_version):
        raise ValueError(
            f"run {results.run_id} was on index {meta['index_version']}, "
            f"the active index is {deps.index_version}; start a new run"
        )
    if (settings.secrets.judge_model or "") == (settings.secrets.llm_model or ""):
        _log("warning: the judge model is the answer model (G2 asks for a different family)")
    ask_fn, answer_llm = A.instrument(deps, settings, clock)
    runner = A.Runner(ask_fn, deps.engine.get_chunks, judge, answer_llm, judge_llm)
    meta = {
        "run_id": results.run_id,
        "datasets": datasets,
        "index_version": deps.index_version,
        "sessions": meta.get("sessions", []),
    }
    results.write_meta(meta)
    progress = A.run_session(
        questions, runner, results, limit=args.limit, redo_errors=args.redo_errors, log=_log
    )
    meta["sessions"].append(
        A.session_entry(
            settings,
            started_at=started,
            progress=progress,
            answer_llm=answer_llm,
            judge_llm=judge_llm,
            index_version=deps.index_version,
        )
    )
    results.write_meta(meta)
    return _finish_answers(settings, questions, results, progress, meta, deps.index_version)


def _finish_answers(
    settings: Settings,
    questions: Sequence[Any],
    results: Any,
    progress: Any,
    meta: dict[str, Any],
    index_version: str,
    conn: Any = None,
) -> int:
    """Summarise the run; write `eval_runs` once every question is finished."""
    from nyaya.eval import answers as A
    from nyaya.eval.runs import save_run

    wanted = {q.id for q in questions}
    records = [r for k, r in results.load().items() if k in wanted]
    s = settings
    metrics = A.summarize(
        records,
        min_supported_share=s.answers.min_supported_share,
        min_unanswerable_refused_share=s.answers.min_unanswerable_refused_share,
        max_wrong_refusal_share=s.refusal.max_wrong_refusal_share,
    )
    metrics["sessions"] = meta["sessions"]
    out: dict[str, Any] = {"run_id": results.run_id, "progress": progress.as_dict()}
    if progress.remaining > 0:
        summary = results.dir / "summary.json"
        summary.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
        out["note"] = f"{progress.remaining} questions left; resume with --run-id {results.run_id}"
        _print({**out, "partial_summary": str(summary)})
        return 0
    datasets = meta["datasets"]
    config = {
        "datasets": datasets,
        "judge_model": s.secrets.judge_model,
        "judge_prompt": s.judge.prompt,
        "llm_model": s.secrets.llm_model,
        "prompt_version": s.versions.prompt,
        "results": str(results.path),
    }

    def save(c: Any) -> tuple[str, Path]:
        return save_run(
            "answers",
            metrics,
            config=config,
            dataset_version="+".join(datasets.values()),
            index_version=index_version,
            conn=c,
            run_id=results.run_id,
        )

    if conn is not None:
        _, path = save(conn)
    else:
        with _connect(settings) as c:
            _, path = save(c)
    gates = {g: metrics[g]["passed"] for g in ("g1", "g2", "g5", "g9")}
    _print({**out, "file": str(path), "gates": gates, "g8": metrics["g8"]["by_type"]})
    return 0 if all(gates.values()) else 1


def cmd_latency(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval import latency as L
    from nyaya.eval.answers import instrument
    from nyaya.eval.ratelimit import PausableClock, Throttle
    from nyaya.eval.runs import save_run
    from nyaya.pipeline import build_deps

    items = read_jsonl(args.dataset, DevItem)
    queries = L.fixed_query_set(items, args.count or settings.latency.query_count)
    deps = build_deps(settings)
    ask_fn, _ = instrument(deps, settings, pace_llm=False)
    clock = PausableClock()
    a = settings.answers
    samples = L.run_latency(
        queries,
        ask_fn,
        warmup=settings.latency.warmup if args.warmup is None else args.warmup,
        request_limit_s=settings.timeouts.request_s,
        throttle=Throttle(a.llm_rpm, clock),
        clock=clock,
        retries=a.rate_limit_retries,
        backoff_base_s=a.backoff_base_s,
        backoff_max_s=a.backoff_max_s,
    )
    metrics = L.summarize(samples, p95_target_s=settings.latency.p95_target_s)
    with _connect(settings) as conn:
        run_id, path = save_run(
            "latency",
            metrics,
            config={"count": len(queries), "llm_model": settings.secrets.llm_model},
            dataset_version=args.dataset.stem,
            index_version=deps.index_version,
            conn=conn,
        )
    keys = ("requests", "failures", "p50_s", "p95_s", "passed")
    _print(
        {
            "run_id": run_id,
            "file": str(path),
            **{k: metrics[k] for k in keys},
            "stages_s": metrics["stages_s"],
        }
    )
    return 0 if metrics["passed"] else 1


def cmd_judge_validate(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.judge import read_hand_labels, validate
    from nyaya.eval.ratelimit import PausableClock
    from nyaya.eval.runs import save_run
    from nyaya.models.encoders import document_text
    from nyaya.search.postgres import PostgresSearch

    labels = read_hand_labels(args.labels)
    judge, judge_llm = _judge_with_pacing(settings, PausableClock())
    j = settings.judge
    with _connect(settings) as conn:
        engine = PostgresSearch(settings=settings, conn=conn)
        chunks = engine.get_chunks(sorted({lab.chunk_id for lab in labels}))
        passages = {c.chunk_id: document_text(c.header, c.text) for c in chunks}
        metrics = validate(
            labels, passages, judge, min_kappa=j.min_kappa, min_labels=j.min_hand_labels
        )
        metrics["tokens"] = dict(judge_llm.tokens)
        run_id, path = save_run(
            "judge-validate",
            metrics,
            config={"judge_model": judge.model, "judge_prompt": j.prompt},
            dataset_version=args.labels.stem,
            index_version=engine.index_version,
            conn=conn,
        )
    keys = ("labels", "scored", "judge_errors", "kappa", "passed", "missing_chunks")
    _print({"run_id": run_id, "file": str(path), **{k: metrics[k] for k in keys}})
    return 0 if metrics["passed"] else 1


def _answers_run(run_id: str | None, root: Path | None = None) -> tuple[Any, list[dict]]:
    """An answers run's results (the latest run when `run_id` is None)."""
    from nyaya.eval.answers import ANSWERS_DIR, ResultsFile

    root = root or ANSWERS_DIR
    if not run_id:
        runs = sorted(p.parent.name for p in root.glob("*/meta.json"))
        if not runs:
            raise FileNotFoundError(f"no answers runs under {root}; run `answers` first")
        run_id = runs[-1]
    results = ResultsFile(run_id, root)
    if not results.exists():
        raise FileNotFoundError(f"no answers run {run_id!r} under {root}")
    return results, list(results.load().values())


def cmd_lawyer_export(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.lawyer import LAWYER_DIR, export

    results, records = _answers_run(args.run_id)
    out_dir = args.out_dir or LAWYER_DIR / results.run_id
    info = export(
        records,
        out_dir,
        run_id=results.run_id,
        size=settings.lawyer.sample_size,
        seed=settings.lawyer.seed,
    )
    _print(info)
    return 0


def cmd_lawyer_import(args: argparse.Namespace, settings: Settings) -> int:
    from nyaya.eval.lawyer import LAWYER_DIR, RUBRIC_VERSION, read_ratings, summarize_ratings
    from nyaya.eval.runs import save_run

    results, _ = _answers_run(args.run_id)
    csv_path = args.csv or LAWYER_DIR / results.run_id / "ratings.csv"
    metrics = summarize_ratings(read_ratings(csv_path))
    meta = results.read_meta()
    with _connect(settings) as conn:
        run_id, path = save_run(
            "lawyer-review",
            metrics,
            config={"answers_run": results.run_id, "rubric": RUBRIC_VERSION, "csv": str(csv_path)},
            dataset_version="+".join(meta.get("datasets", {}).values()) or None,
            index_version=meta.get("index_version"),
            conn=conn,
        )
    means = {q: v["mean"] for q, v in metrics["questions"].items()}
    _print(
        {
            "run_id": run_id,
            "file": str(path),
            "answers": metrics["answers"],
            "means": means,
            "gate": False,
        }
    )
    return 0


def cmd_ragas(args: argparse.Namespace, settings: Settings) -> int:
    from datetime import UTC, datetime

    from nyaya.eval import ragas_metrics as R
    from nyaya.eval.answers import ResultsFile
    from nyaya.eval.runs import save_run

    if R.ragas_version() is None:
        raise RuntimeError("ragas is not installed: uv sync --extra eval --extra ragas")
    if args.results is not None:
        if not args.results.exists():
            raise FileNotFoundError(f"answers results not found: {args.results}")
        answers = ResultsFile(args.results.parent.name, args.results.parent.parent)
        answers.path = args.results
        records = list(answers.load().values())
    else:
        answers, records = _answers_run(args.run_id)
    if not records:
        raise ValueError(f"answers run {answers.run_id} has no finished questions")
    meta = answers.read_meta()
    dev_stem = meta.get("datasets", {}).get("dev")
    references = R.read_references(args.references or R.default_references(dev_stem))

    cfg = settings.ragas
    use_relevancy = cfg.answer_relevancy and not args.no_relevancy
    fetch = R.DbContexts(settings)
    try:
        samples = R.build_samples(
            records, references, max_contexts=cfg.max_contexts, fetch_contexts=fetch
        )
    finally:
        fetch.close()
    if not samples:
        raise ValueError(f"answers run {answers.run_id} has no dev-set questions")

    llm, limited = R.make_llm(settings)
    embeddings = R.make_embeddings(settings) if use_relevancy else None
    metrics = R.make_metrics(llm, embeddings, cfg.strictness)
    run_config = {
        "answers_run": answers.run_id,
        "ragas_version": R.ragas_version(),
        "llm_model": getattr(llm, "model", None),
        "embed_model": cfg.embed_model if use_relevancy else None,
        "metrics": list(metrics),
        "max_contexts": cfg.max_contexts,
        "strictness": cfg.strictness,
    }
    results = ResultsFile(answers.run_id, R.RAGAS_DIR)
    old = results.read_meta()
    if old and old.get("config") != run_config:
        raise ValueError(
            f"{results.dir} was scored with {old.get('config')}; move it away to score "
            "again with these settings"
        )
    sessions = old.get("sessions", [])
    results.write_meta({"config": run_config, "sessions": sessions})
    started = datetime.now(UTC)
    progress = R.run_session(samples, metrics, results, limit=args.limit, log=_log)
    sessions.append(
        {
            "started_at": started.isoformat(),
            "tokens": dict(getattr(limited, "tokens", {}) or {}),
            "calls": getattr(limited, "calls", 0),
            "rate_limited": getattr(limited, "rate_limited", 0),
            "progress": progress.as_dict(),
        }
    )
    results.write_meta({"config": run_config, "sessions": sessions})

    wanted = {s.id for s in samples}
    rows = [r for k, r in results.load().items() if k in wanted]
    ragas = R.aggregate(rows, list(metrics))
    ours = R.our_metrics(
        records,
        min_supported_share=settings.answers.min_supported_share,
        min_unanswerable_refused_share=settings.answers.min_unanswerable_refused_share,
        max_wrong_refusal_share=settings.refusal.max_wrong_refusal_share,
    )
    table = R.markdown_table(ours, ragas)
    report = {
        "complete": progress.remaining == 0,
        "progress": progress.as_dict(),
        "ragas": ragas,
        "ours": ours,
        "per_question": sorted(rows, key=lambda r: r["id"]),
        "table_md": table,
    }
    run_id, path = save_run(
        "ragas",
        report,
        config={**run_config, "references": str(args.references or "default")},
        dataset_version=dev_stem,
        index_version=meta.get("index_version"),
        run_id=f"ragas-{answers.run_id}",
    )
    md = path.with_suffix(".md")
    md.write_text(table, encoding="utf-8", newline="\n")
    out: dict[str, Any] = {
        "run_id": run_id,
        "file": str(path),
        "table": str(md),
        "progress": progress.as_dict(),
        "refusals": ragas["refusals"],
        "means": {k: v["mean"] for k, v in ragas["metrics"].items()},
        "gate": False,
    }
    if progress.remaining > 0:
        out["note"] = f"{progress.remaining} questions left; run the same command to resume"
    _print(out)
    return 0


COMMANDS = {
    "build-devset": cmd_build_devset,
    "g6-holdout": cmd_g6_holdout,
    "retrieval": cmd_retrieval,
    "size-steps": cmd_size_steps,
    "bm25-trigger": cmd_bm25_trigger,
    "tune-threshold": cmd_tune_threshold,
    "answers": cmd_answers,
    "latency": cmd_latency,
    "judge-validate": cmd_judge_validate,
    "lawyer-export": cmd_lawyer_export,
    "lawyer-import": cmd_lawyer_import,
    "ragas": cmd_ragas,
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
    try:
        settings = load_settings()
        return COMMANDS[args.command](args, settings)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"nyaya-eval {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
