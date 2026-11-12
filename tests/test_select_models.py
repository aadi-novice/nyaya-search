import importlib.util
import sys

import numpy as np

from eval_fakes import chunk
from nyaya.eval.datasets import DevItem, Gold
from nyaya.eval.retrieval import evaluate


def load_script(repo_root):
    spec = importlib.util.spec_from_file_location(
        "select_models", repo_root / "scripts" / "select_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def test_vector_retriever_and_report(repo_root):
    sm = load_script(repo_root)
    chunks = [
        chunk("A#judgment#001", labels=["1"]),
        chunk("A#editorial#001", refs=["1"]),
        chunk("B#judgment#001", labels=["1"]),
    ]
    engine = sm.InMemoryEngine(
        {c.chunk_id: c for c in chunks}, {"A#editorial#001": ["A#judgment#001"]}
    )
    ids = [c.chunk_id for c in chunks]
    docs = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    retriever = sm.VectorRetriever(
        ids=ids,
        doc_vecs=docs,
        query_vecs={"qa": np.array([0.6, 0.8], dtype=np.float32)},
        engine=engine,
        k=2,
        rerank_candidates=3,
        rerank=lambda q, cs: [
            1.0 if c.judgment_id == "A" and c.zone == "judgment" else 0.0 for c in cs
        ],
    )
    assert retriever.run("exact", "qa", None).chunk_ids == ["A#editorial#001", "B#judgment#001"]
    items = [
        DevItem(id="q", question="qa", type="answerable", gold=[Gold(path="A", para_labels=["1"])])
    ]
    rep = evaluate(items, retriever, engine, ["exact", "exact_rerank"])
    s = rep.summary()
    assert s["exact"]["paragraph"]["recall@50"] == 1.0  # the Held chunk counts via its link
    assert s["exact_rerank"]["judgment"]["mrr"] == 1.0
    text = sm.render_report(
        {"m": {"summary": s, "chunks_per_s": 12.5, "peak_mib": 900}},
        dataset="dev_v1.jsonl",
        year=2025,
        n=1,
    )
    assert "| m | 1.000 |" in text and "12.5" in text and "**Choice:**" in text


def test_kaggle_vectors_are_read_in_chunk_order(repo_root, tmp_path):
    import json

    import pytest

    sm = load_script(repo_root)
    assert sm.parse_vectors_args(["BAAI/bge-m3=out/BAAI__bge-m3"]) == {
        "BAAI/bge-m3": sm.Path("out/BAAI__bge-m3")
    }
    with pytest.raises(SystemExit, match="MODEL=DIR"):
        sm.parse_vectors_args(["no-equals-sign"])

    vdir = tmp_path / "BAAI__bge-m3"
    (vdir / "vectors").mkdir(parents=True)
    vecs = np.eye(3, 4, dtype=np.float32)
    np.save(vdir / "vectors" / "2025.npy", vecs)
    (vdir / "vectors" / "2025.ids.txt").write_text("a\nb\nc\n", encoding="utf-8")
    got = sm.kaggle_doc_vectors(vdir, 2025, ["c", "a", "b"], 4)
    assert np.array_equal(got, vecs[[2, 0, 1]])
    with pytest.raises(SystemExit, match="do not match"):
        sm.kaggle_doc_vectors(vdir, 2025, ["a", "b"], 4)
    with pytest.raises(SystemExit, match=r"no vectors/2024\.npy"):
        sm.kaggle_doc_vectors(vdir, 2024, ["a"], 4)

    assert sm.kaggle_run_stats(vdir)["chunks_per_s"] is None
    (vdir / "embed_report.json").write_text(
        json.dumps(
            {
                "chunks_per_s": 80.5,
                "peak_vram_allocated_gb": 2.5,
                "environment": {"device": "Tesla T4"},
            }
        ),
        encoding="utf-8",
    )
    stats = sm.kaggle_run_stats(vdir)
    assert stats == {"chunks_per_s": 80.5, "peak_mib": 2560.0, "device": "Kaggle Tesla T4"}


def test_report_shows_missing_measures_as_na(repo_root):
    sm = load_script(repo_root)
    m = {"recall@50": 0.5, "recall@5": 0.25, "mrr": 0.5}
    s = {"exact": {"judgment": m, "paragraph": m}, "exact_rerank": {"judgment": m, "paragraph": m}}
    text = sm.render_report(
        {"BAAI/bge-m3": {"summary": s, "chunks_per_s": None, "peak_mib": None, "device": "Kaggle"}},
        dataset="dev_v1.jsonl",
        year=2025,
        n=3,
    )
    assert "| n/a | n/a | Kaggle |" in text
