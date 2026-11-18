"""nyaya-eval replay-check: the citation check re-run on saved answers records, offline."""

from __future__ import annotations

import json

import pytest

from nyaya.config import load_settings
from nyaya.eval import answers as A
from nyaya.eval import cli
from nyaya.eval import replay as R
from nyaya.link.aliases import find_citations

SETTINGS = load_settings(env_file=None)
SNAP_ON = SETTINGS.model_copy(update={"quote": SETTINGS.quote.model_copy(update={"snap": True})})

C1 = "2025_1_1_9#judgment#001"
C2 = "2024_2_5_9#editorial#003"
T1 = (
    "12. The power to quash must be exercised sparingly and with circumspection, and the "
    "High Court is not expected to conduct a mini trial. 13. An arbitral tribunal is the "
    "preferred first authority to decide all questions of non-arbitrability."
)
T2 = "HELD: The procedure must be fair, just and reasonable, and not fanciful or arbitrary."
CONTEXTS = [
    f"ALPHA v. STATE (2025) \u00b7 Judgment \u00b7 X, J. \u00b7 paras 12\u201313\n\n{T1}",
    f"BETA v. UNION OF INDIA (2024) \u00b7 Editorial \u00b7 paras 1\u20131\n\n{T2}",
]
Q1 = "The power to quash must be exercised sparingly and with circumspection"
FRAMED = "Indeed, the High Court is not expected to conduct a mini trial."
HYPHEN = (
    "An arbitral tribunal is the preferred first authority to decide all questions of "
    "non\u2011arbitrability."
)


def citation(chunk_id: str, label: str, quote: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "judgment_id": chunk_id.split("#")[0],
        "case_name": "ALPHA versus STATE",
        "scr_citation": "[2025] 1 S.C.R. 1",
        "neutral_citation": None,
        "paras": ["12"],
        "pages": [3],
        "zone": "judgment",
        "opinion_type": "majority",
        "passage_label": label,
        "quote": quote,
    }


def record(rid: str, set_: str = "dev", **kw) -> dict:
    rec = {
        "id": rid,
        "set": set_,
        "status": "answered",
        "retrieved": [C1, C2],
        "contexts": CONTEXTS,
        "answer": [],
        "unverified": [],
        "gaps": [],
        "rule_breaks": [],
    }
    rec.update(kw)
    return rec


def verified_sentence(quote: str = Q1) -> dict:
    return {"sentence": "Quashing is used sparingly.", "citations": [citation(C1, "P1", quote)]}


def unverified(sentence: str, labels=("P1",), reason="quote_not_found") -> dict:
    return {"sentence": sentence, "labels": list(labels), "reason": reason}


def brk(index: int, kind: str, label: str | None = "P1", detail=None, snap=None) -> dict:
    return {"sentence_index": index, "kind": kind, "label": label, "detail": detail, "snap": snap}


# a run saved since 2026-11-18: the failed quote is in the rule break
SAVED = record(
    "dev-1",
    answer=[verified_sentence()],
    unverified=[unverified("No mini trial is held.")],
    rule_breaks=[brk(1, "quote_not_found", detail=FRAMED)],
)
# an older run: no saved quote; the sentence copies P1 with a non-breaking hyphen
OLD = record(
    "dev-2",
    status="refused",
    unverified=[unverified(HYPHEN)],
    rule_breaks=[brk(0, "quote_not_found")],
)
GATED = record("dev-3", status="refused")  # refused before the LLM: nothing to replay
PASSAGES = record("dev-4", status="passages", unverified=[unverified("x")])
ADV = record(
    "adv-1",
    set_="adversarial",
    answer=[verified_sentence()],
    g9={"category": "advice", "expect": "refuse", "passed": False},
)


def test_passages_rebuilds_labels_and_text():
    label_map, chunks = R.passages(SAVED)
    assert list(label_map) == ["P1", "P2"]
    assert label_map["P1"].case_name == "ALPHA versus STATE"  # from the verified citation
    assert label_map["P2"].case_name == "BETA v. UNION OF INDIA"  # from the header
    assert label_map["P2"].zone.value == "editorial"
    assert chunks[C1].text == T1 and chunks[C2].text == T2
    assert R.chunk_text("no header") == "no header"


def test_saved_quote_is_snapped_and_verified():
    out = R.replay_record(SAVED, SNAP_ON, proxy=False)
    assert out["status"] == "answered" and len(out["answer"]) == 2 and not out["unverified"]
    (snap,) = out["snaps"]
    assert snap["quote"] == FRAMED and snap["quote_source"] == "trace"
    assert snap["span"] in T1 and snap["span"].startswith("the High Court is not expected")
    # the snapped citation shows the span, never the LLM's quote
    quotes = [c["quote"] for s in out["answer"] for c in s["citations"]]
    assert FRAMED not in quotes and snap["span"] in quotes


def test_saved_quote_without_snapping_stays_unverified():
    out = R.replay_record(SAVED, SETTINGS, proxy=False)
    assert len(out["answer"]) == 1 and len(out["unverified"]) == 1 and not out["snaps"]
    assert out["rule_breaks"][0]["detail"] == FRAMED


def test_old_record_needs_the_proxy():
    out = R.replay_record(OLD, SETTINGS, proxy=False)
    assert out["status"] == "refused" and out["unverified"][0]["reason"] == "quote_length"
    out = R.replay_record(OLD, SETTINGS, proxy=True)
    # the new normalisation maps U+2011 to "-": verified without any snap
    assert out["status"] == "answered" and not out["unverified"] and not out["snaps"]


def test_proxy_skips_candidates_the_old_check_would_have_found():
    # an exact copy cannot be the quote that failed; the inline quotation can
    sent = f'It says "{HYPHEN}" plainly.'
    assert R.proxy_quote(sent, T1, 8) == HYPHEN  # the closer candidate
    exact = "The power to quash must be exercised sparingly and with circumspection"
    assert R.proxy_quote(exact, T1, 8) == ""
    # a citation that passed (no rule break for it) may be stood in by an exact copy
    assert R.proxy_quote(exact, T1, 8, failed=False) == exact


def test_records_that_are_not_replayed():
    for rec in (GATED, PASSAGES):
        out = R.replay_record(rec, SNAP_ON, proxy=True)
        assert out["status"] == rec["status"] and out["snaps"] == []


def test_alignment_skips_a_verified_sentence_with_a_failed_side_citation():
    rec = record(
        "dev-5",
        answer=[verified_sentence()],
        unverified=[unverified("No mini trial.", labels=("P1", "P2"))],
        rule_breaks=[
            brk(0, "quote_not_found", label="P2", detail="side quote"),
            brk(1, "quote_not_found", label="P1", detail=FRAMED),
            brk(1, "quote_length", label="P2", detail="short"),
        ],
    )
    (failed,) = R._failed_quotes(rec)
    assert {k: [b["detail"] for b in v] for k, v in failed.items()} == {
        "P1": [FRAMED],
        "P2": ["short"],
    }


def test_alignment_with_step_four_and_unknown_labels():
    canonical = next(iter(find_citations("[2001] 1 S.C.R. 1")))[2]
    rec = record(
        "dev-6",
        status="refused",
        unverified=[
            unverified(f'As held in [2001] 1 S.C.R. 1, "{Q1}".', reason="uncited_case_name"),
            unverified("Unknown label.", labels=("P9",), reason="unknown_label"),
            unverified("Not aligned.", labels=("P2",)),
        ],
        rule_breaks=[
            brk(0, "uncited_citation", label=None, detail=canonical),
            brk(1, "unknown_label", label="P9"),
        ],
    )
    out = R.replay_record(rec, SNAP_ON, proxy=True)
    assert out["status"] == "refused"
    assert [u["reason"] for u in out["unverified"]] == [
        "uncited_case_name",
        "unknown_label",
        "quote_length",  # no break recorded: the 2-word sentence stands in
    ]


def test_replay_metrics_before_and_after():
    report = R.replay([SAVED, OLD, GATED, PASSAGES, ADV], SNAP_ON, proxy=True)
    before, after = report["recorded"], report["replayed"]
    assert (before["verified_sentences"], before["unverified_sentences"]) == (2, 3)
    assert (after["verified_sentences"], after["unverified_sentences"]) == (4, 1)
    assert after["snapped_citations"] == 1 and len(report["snaps"]) == 1
    assert before["answerable_refused_ids"] == ["dev-2", "dev-3"]
    assert after["answerable_refused_ids"] == ["dev-3"]
    assert before["answerable_counted"] == 3  # `passages` is neither
    assert after["must_refuse_answered"] == ["adv-1"]
    assert report["changed"] == [{"id": "dev-2", "before": "refused", "after": "answered"}]
    assert report["settings"]["snap"] is True and report["proxy"] is True
    assert R.metrics([])["unverified_share"] is None


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings(env_file=None))
    monkeypatch.setattr(A, "ANSWERS_DIR", tmp_path / "answers")
    run = tmp_path / "answers" / "answers-x"
    run.mkdir(parents=True)
    (run / "meta.json").write_text("{}", encoding="utf-8")
    lines = [json.dumps({**r, "run_id": "answers-x"}) for r in (SAVED, OLD, GATED)]
    (run / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path, run / "results.jsonl"


def test_cli_replay_check(cli_env, capsys):
    tmp_path, results = cli_env
    out = tmp_path / "out" / "replay.json"
    assert cli.main(["replay-check", "--results", str(results), "--snap", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["settings"]["snap"] is True and printed["proxy"] is False
    assert printed["replayed"]["snapped_citations"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["snaps"][0]["quote"] == FRAMED

    assert cli.main(["replay-check", "--proxy", "--no-snap"]) == 0  # the latest run
    printed = json.loads(capsys.readouterr().out)
    assert printed["settings"]["snap"] is False and printed["proxy"] is True
    assert printed["replayed"]["answerable_refused_ids"] == ["dev-3"]

    assert cli.main(["replay-check", "--results", str(tmp_path / "missing.jsonl")]) == 2
