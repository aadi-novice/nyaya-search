import csv
import json

import pandas as pd
import pytest

from nyaya.eval.datasets import DevItem, read_jsonl
from nyaya.eval.devset_from_held import (
    HeldCandidate,
    check_paraphrase,
    draft_devset,
    load_candidates_from_build,
    order_candidates,
    party_terms,
    resolved_labels,
    shared_run,
    write_handcheck_csv,
)

HELD = (
    "Held: The High Court erred in granting anticipatory bail without considering "
    "the gravity of the offence and the antecedents of the accused. [Para 12]"
)


def cand(path="2025_1_1_10", year=2025, text=HELD, labels=("12",), n=0, **kw):
    return HeldCandidate(
        path=path,
        year=year,
        held_text=text,
        para_labels=list(labels),
        source_held_chunk=f"{path}#editorial#{n:03d}",
        petitioner=kw.get("petitioner", "Ramesh Kumar Yadav"),
        respondent=kw.get("respondent", "State of Uttar Pradesh"),
        title=kw.get("title", "Ramesh Kumar Yadav v. State of Uttar Pradesh"),
    )


# --- the paraphrase rule -----------------------------------------------------------


def test_shared_run_finds_five_word_copy():
    q = "Did the court err because it ignored the gravity of the offence and the record?"
    assert shared_run(q, HELD, 5) == "the gravity of the offence"


def test_shared_run_ignores_case_and_punctuation():
    q = "the GRAVITY, of the offence?"
    assert shared_run(q, HELD, 5) == "the gravity of the offence"


def test_shared_run_allows_four_words():
    q = "Should a court weigh the gravity of an alleged crime before pre-arrest bail?"
    assert shared_run(q, HELD, 5) is None


def test_party_terms_skip_generic_words():
    terms = party_terms("Ramesh Kumar Yadav", "State of Uttar Pradesh & Ors.", "")
    assert {"ramesh", "kumar", "yadav"} <= terms
    assert "state" not in terms and "ors" not in terms and "of" not in terms


def test_party_terms_from_title():
    assert "sharma" in party_terms("", "", "Anil Sharma v. Union of India")
    assert "union" not in party_terms("", "", "Anil Sharma v. Union of India")


def test_check_paraphrase_reasons():
    c = cand()
    ok = "When may an appellate court set aside pre-arrest bail granted by a High Court?"
    assert check_paraphrase(ok, c, 5) is None
    copied = "Was it wrong to grant bail without considering the gravity of the offence?"
    assert check_paraphrase(copied, c, 5).startswith("shared_run")
    named = "Why was Yadav denied pre-arrest bail by the appellate court?"
    assert check_paraphrase(named, c, 5) == "party_name: yadav"
    assert check_paraphrase("bail?", c, 5) == "too_short"
    assert check_paraphrase("", c, 5) == "too_short"


# --- selection -------------------------------------------------------------------


def test_order_spreads_years_and_judgments():
    cands = [cand(path=f"2024_{i}", year=2024, n=k) for i in range(3) for k in range(2)]
    cands += [cand(path=f"2025_{i}", year=2025, n=k) for i in range(3) for k in range(2)]
    order = order_candidates(cands, seed=11)
    assert len(order) == len(cands)
    years = [c.year for c in order[:6]]
    assert years.count(2024) == 3 and years.count(2025) == 3
    assert len({c.path for c in order[:6]}) == 6  # one point per judgment first
    assert order == order_candidates(list(reversed(cands)), seed=11)  # deterministic


def test_resolved_labels():
    linked = [["9", "10"], ["12.1", "12.2"]]
    assert resolved_labels(["9", "12", "15"], linked) == ["9", "12"]


# --- drafting ----------------------------------------------------------------------


class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete_json(self, system, user, timeout):
        self.calls.append((system, user))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def test_draft_devset_keeps_rejects_and_retries():
    cands = [cand(path=f"2025_1_{i}_9", n=0) for i in range(4)]
    llm = FakeLLM(
        [
            {"question": "When may an appellate court cancel pre-arrest bail?"},
            {"question": "Did the court ignore the gravity of the offence and more?"},  # copy
            {"question": "What factors should guide a grant of protection from arrest?"},
            {"question": "Why did Yadav lose the bail appeal?"},  # party name
            {"question": "Why did Yadav lose the bail appeal again?"},  # party name again: rejected
            RuntimeError("rate limited"),
        ]
    )
    result = draft_devset(cands, llm, size=3, max_shared_words=5, seed=11, retries=1)
    assert len(result.items) == 2
    assert all(isinstance(i, DevItem) for i in result.items)
    assert [i.id for i in result.items] == ["dev-0001", "dev-0002"]
    assert result.llm_calls == 6
    reasons = [r["reason"] for r in result.rejected]
    assert any(r.startswith("shared_run") for r in reasons)
    assert any(r.startswith("party_name") for r in reasons)
    assert any(r.startswith("llm_error") for r in reasons)
    # the retry tells the model why the first draft was rejected
    assert "rejected" in llm.calls[2][1]
    item = result.items[0]
    assert item.gold[0].para_labels == ["12"]
    assert item.source_held_chunk.endswith("#editorial#000")


def test_draft_devset_stops_at_size():
    cands = [cand(path=f"2025_1_{i}_9") for i in range(5)]
    llm = FakeLLM(
        [{"question": f"Which test governs pre-arrest bail, variant {i}?"} for i in range(5)]
    )
    result = draft_devset(cands, llm, size=2, max_shared_words=5, seed=1, retries=0)
    assert len(result.items) == 2 and result.llm_calls == 2


def test_draft_devset_bad_json_shape():
    llm = FakeLLM([{"q": "x"}])
    result = draft_devset([cand()], llm, size=1, max_shared_words=5, seed=1, retries=0)
    assert result.items == [] and result.rejected[0]["reason"] == "bad_response"


def test_write_handcheck_csv(tmp_path):
    cands = [cand(path=f"2025_1_{i}_9") for i in range(3)]
    llm = FakeLLM(
        [{"question": f"Which test governs pre-arrest bail, variant {i}?"} for i in range(3)]
    )
    result = draft_devset(cands, llm, size=3, max_shared_words=5, seed=1, retries=0)
    out = write_handcheck_csv(tmp_path / "h.csv", result, n=2, seed=4)
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0]) >= {"id", "question", "held_text", "path", "para_labels", "ok", "comment"}
    assert rows[0]["ok"] == ""


# --- reading build outputs --------------------------------------------------------


def _chunk(cid, zone, labels=(), refs=(), text="t"):
    jid = cid.split("#")[0]
    return {
        "chunk_id": cid,
        "judgment_id": jid,
        "zone": zone,
        "para_labels": list(labels),
        "para_refs": list(refs),
        "text": text,
        "header": "h",
        "char_start": 0,
        "char_end": 1,
        "pages": [1],
        "flags": [],
        "clean_text_version": "ct1",
        "chunker_version": "ch1",
    }


def test_load_candidates_from_build(tmp_path):
    p = "2025_1_230_238"
    (tmp_path / "chunks").mkdir()
    (tmp_path / "links").mkdir()
    (tmp_path / "judgments").mkdir()
    chunks = [
        _chunk(f"{p}#editorial#001", "editorial", refs=["9", "40"], text=HELD),
        _chunk(f"{p}#editorial#002", "editorial", refs=["99"], text="Held: unresolved."),
        _chunk(f"{p}#editorial#003", "editorial", refs=[], text="Held: no link."),
        _chunk(f"{p}#judgment#010", "judgment", labels=["9", "10"]),
    ]
    with (tmp_path / "chunks" / "2025.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    pd.DataFrame(
        {"from_chunk_id": [f"{p}#editorial#001"], "to_chunk_id": [f"{p}#judgment#010"]}
    ).to_parquet(tmp_path / "links" / "2025.parquet")
    pd.DataFrame(
        {
            "path": [p],
            "title": ["A v. B"],
            "petitioner": ["Jit Vinayak Arolkar"],
            "respondent": ["State of Goa"],
            "decision_year": [2025],
        }
    ).to_parquet(tmp_path / "judgments" / "2025.parquet")

    got = load_candidates_from_build(tmp_path, [2025])
    assert len(got) == 1
    c = got[0]
    assert c.para_labels == ["9"]
    assert c.source_held_chunk == f"{p}#editorial#001"
    assert c.petitioner == "Jit Vinayak Arolkar" and c.year == 2025


def test_load_candidates_missing_year(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates_from_build(tmp_path, [2025])


def test_written_devset_reads_back(tmp_path):
    from nyaya.eval.datasets import write_jsonl

    llm = FakeLLM([{"question": "Which test governs pre-arrest bail on appeal?"}])
    result = draft_devset([cand()], llm, size=1, max_shared_words=5, seed=1, retries=0)
    p = write_jsonl(tmp_path / "dev.jsonl", result.items)
    assert read_jsonl(p, DevItem) == result.items


def test_draft_devset_aborts_after_repeated_provider_errors():
    cands = [cand(path=f"2025_1_{i}_9") for i in range(10)]
    llm = FakeLLM([RuntimeError("402")] * 10)
    result = draft_devset(cands, llm, size=5, max_shared_words=5, seed=1, max_consecutive_errors=3)
    assert result.aborted and result.llm_calls == 3 and result.items == []
