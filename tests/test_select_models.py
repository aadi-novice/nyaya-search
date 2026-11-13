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
