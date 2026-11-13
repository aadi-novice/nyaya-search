"""Encoders: prefixes, yes/no scoring, the OpenRouter fallback and the chain.

Everything here runs without a GPU or network: models, tokenizers and the
HTTP transport are fakes. The real models are exercised by the real check
described in the Phase 1 report.
"""

import json
import math
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch

from nyaya.config import Secrets, load_settings
from nyaya.models import encoders as enc


@pytest.fixture(scope="module")
def settings():
    s = load_settings(env_file=None)
    # no fallback key, whatever the environment holds
    return s.model_copy(update={"secrets": Secrets(_env_file=None, openrouter_api_key=None)})


# --- embedder ----------------------------------------------------------------------


class FakeST:
    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        rng = np.random.default_rng(len(sentences))
        v = rng.normal(size=(len(sentences), self.dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_query_gets_prefix_documents_do_not(settings):
    st = FakeST()
    e = enc.Embedder(settings, model=st)
    q = e.embed_query("anticipatory bail")
    assert q.shape == (1024,) and q.dtype == np.float32
    sent, kwargs = st.calls[0]
    assert sent == [settings.embed.query_prefix + "anticipatory bail"]
    assert sent[0].startswith("Instruct: Given a legal research question")
    assert sent[0].endswith("\nQuery: anticipatory bail")
    assert kwargs["normalize_embeddings"] is True
    docs = e.embed_documents(["H\n\nT", "x"])
    assert docs.shape == (2, 1024)
    assert st.calls[1][0] == ["H\n\nT", "x"]


def test_embed_documents_empty(settings):
    e = enc.Embedder(settings, model=FakeST())
    assert e.embed_documents([]).shape == (0, 1024)


def test_wrong_dimension_rejected(settings):
    e = enc.Embedder(settings, model=FakeST(dim=8))
    with pytest.raises(ValueError, match="expected"):
        e.embed_query("q")


def test_document_text():
    assert enc.document_text("Head", "body") == "Head\n\nbody"
    assert enc.document_text("", "body") == "body"


# --- Qwen reranker -----------------------------------------------------------------

YES, NO, PAD, BAIL = 1, 2, 0, 7


class FakeTokenizer:
    """Word-level: 'bail' -> 7, every other word -> 9; special tokens 'yes'/'no'."""

    def convert_tokens_to_ids(self, tok):
        return {"yes": YES, "no": NO}[tok]

    def _ids(self, text):
        return [BAIL if w.strip(".,:") == "bail" else 9 for w in text.split()]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, texts, **kwargs):
        ids = [self._ids(t)[: kwargs["max_length"]] for t in texts]
        return {"input_ids": ids}

    def pad(self, enc_, padding, return_tensors):
        width = max(len(x) for x in enc_["input_ids"])
        ids = [[PAD] * (width - len(x)) + x for x in enc_["input_ids"]]
        mask = [[0] * (width - len(x)) + [1] * len(x) for x in enc_["input_ids"]]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class FakeLM:
    """'yes' logit at the last position = number of 'bail' tokens; 'no' logit = 1."""

    def __init__(self):
        self.batches = []

    def __call__(self, input_ids, attention_mask, logits_to_keep=0):
        assert logits_to_keep == 1  # only the last position is scored
        self.batches.append(input_ids.shape[0])
        b, length = input_ids.shape
        logits = torch.zeros(b, length, 12)
        logits[:, -1, YES] = (input_ids == BAIL).sum(dim=1).float()
        logits[:, -1, NO] = 1.0
        return SimpleNamespace(logits=logits)


def make_qwen(settings, batch_size=2, max_length=8192):
    return enc.QwenReranker(
        settings,
        model=FakeLM(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        batch_size=batch_size,
        max_length=max_length,
    )


def test_format_pair_matches_model_card():
    s = enc.format_rerank_pair("Ins", "Q?", "Doc")
    assert s == "<Instruct>: Ins\n<Query>: Q?\n<Document>: Doc"
    assert enc.RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether the Document")
    assert enc.RERANK_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_yes_probability_is_softmax_of_yes_vs_no():
    logits = torch.zeros(2, 5)
    logits[0, YES], logits[0, NO] = 2.0, 0.0
    logits[1, YES], logits[1, NO] = 0.0, 0.0
    p = enc.yes_probability(logits, YES, NO)
    assert p[0] == pytest.approx(math.exp(2) / (math.exp(2) + 1))
    assert p[1] == pytest.approx(0.5)


def test_qwen_scores_in_input_order_across_batches(settings):
    r = make_qwen(settings, batch_size=2)
    docs = ["no match here at all", "bail bail bail", "bail", "a long text without it ok"]
    scores = r.scores("question", docs)
    # instruction and query hold no "bail"; each doc's count drives the score
    expected = [1 / (1 + math.exp(1 - n)) for n in (0, 3, 1, 0)]
    assert scores == pytest.approx(expected, abs=1e-6)
    assert r.model.batches == [2, 2]


def test_qwen_rerank_top_n(settings):
    r = make_qwen(settings)
    out = r.rerank("q", ["x", "bail bail", "bail"], top_n=2)
    assert [x.index for x in out] == [1, 2]
    assert out[0].score > out[1].score
    assert r.rerank("q", [], top_n=2) == []
    assert r.kind == "qwen" and r.model_name == settings.rerank.model


def test_qwen_truncation_room(settings):
    r = make_qwen(settings, max_length=len(enc.RERANK_PREFIX.split()) + 12 + 3)
    # room for 3 tokens of the pair: "<Instruct>: Given a" -> no bail tokens survive
    assert r.scores("q", ["bail bail bail"])[0] == pytest.approx(1 / (1 + math.e))


def test_top_results_ties_by_index():
    out = enc.top_results([0.5, 0.9, 0.5], top_n=3)
    assert [(x.index, x.score) for x in out] == [(1, 0.9), (0, 0.5), (2, 0.5)]


# --- OpenRouter fallback -------------------------------------------------------------

CONFIRMED = {
    "id": "gen-rerank-1",
    "model": "rerank-v5.0-fast",
    "results": [
        {"index": 0, "relevance_score": 0.641675, "document": {"text": "a"}},
        {"index": 2, "relevance_score": 0.0063647316, "document": {"text": "c"}},
    ],
    "usage": {"total_tokens": 62, "cost": 0.00000124},
    "provider": "Cohere",
}


def client_returning(status=200, payload=None, text=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fallback_request_and_parse():
    seen = []
    fb = enc.CohereFallbackReranker(
        "sk-test", "cohere/rerank-5-fast", client=client_returning(payload=CONFIRMED, seen=seen)
    )
    out = fb.rerank("q", ["a", "b", "c"], top_n=8)
    assert [(r.index, r.score) for r in out] == [(0, 0.641675), (2, 0.0063647316)]
    req = seen[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/rerank"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body == {
        "model": "cohere/rerank-5-fast",
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_n": 3,
    }


def test_fallback_leading_whitespace_body_parses():
    # OpenRouter sends keep-alive whitespace before the JSON body
    fb = enc.CohereFallbackReranker(
        "k", "m", client=client_returning(text="\n   " + json.dumps(CONFIRMED))
    )
    assert len(fb.rerank("q", ["a", "b", "c"], top_n=2)) == 2


def test_fallback_truncates_to_top_n():
    fb = enc.CohereFallbackReranker("k", "m", client=client_returning(payload=CONFIRMED))
    assert [r.index for r in fb.rerank("q", ["a", "b", "c"], top_n=1)] == [0]


@pytest.mark.parametrize(
    ("status", "payload", "text", "match"),
    [
        (402, {"error": {"message": "Insufficient credits"}}, None, "HTTP 402"),
        (500, None, "oops", "HTTP 500"),
        (200, None, "not json", "invalid JSON"),
        (200, {"data": []}, None, "unexpected"),
        (200, {"results": [{"index": 5, "relevance_score": 0.1}]}, None, "out of range"),
        (200, {"results": [{"index": 0}]}, None, "bad rerank result"),
    ],
)
def test_fallback_errors(status, payload, text, match):
    fb = enc.CohereFallbackReranker(
        "k", "m", client=client_returning(status=status, payload=payload, text=text)
    )
    with pytest.raises(enc.RerankError, match=match):
        fb.rerank("q", ["a", "b"], top_n=2)


def test_fallback_network_error():
    def handler(request):
        raise httpx.ConnectTimeout("slow")

    fb = enc.CohereFallbackReranker(
        "k", "m", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(enc.RerankError, match="ConnectTimeout"):
        fb.rerank("q", ["a"], top_n=1)


def test_fallback_empty_docs_and_missing_key(settings):
    fb = enc.CohereFallbackReranker("k", "m", client=client_returning(payload=CONFIRMED))
    assert fb.rerank("q", [], top_n=3) == []
    with pytest.raises(enc.RerankError, match="OPENROUTER_API_KEY"):
        enc.CohereFallbackReranker.from_settings(settings)


# --- chain ---------------------------------------------------------------------------


class StubReranker:
    def __init__(self, kind, fail=False):
        self.kind = kind
        self.model_name = f"{kind}-model"
        self.fail = fail
        self.calls = 0

    def rerank(self, query, documents, top_n):
        self.calls += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return enc.top_results([float(len(d)) for d in documents], top_n)


def test_chain_uses_primary():
    p, f = StubReranker("qwen"), StubReranker("cohere")
    out = enc.RerankerChain(p, f).rerank("q", ["a", "bbb"], 1)
    assert (out.kind, out.model_name, out.fallback_used) == ("qwen", "qwen-model", False)
    assert [r.index for r in out.results] == [1]
    assert f.calls == 0


def test_chain_falls_back_when_primary_raises():
    p, f = StubReranker("qwen", fail=True), StubReranker("cohere")
    out = enc.RerankerChain(p, f).rerank("q", ["a"], 1)
    assert out.fallback_used and out.kind == "cohere"
    assert "CUDA out of memory" in out.primary_error


def test_chain_without_fallback_raises():
    with pytest.raises(enc.RerankError, match="no fallback"):
        enc.RerankerChain(StubReranker("qwen", fail=True), None).rerank("q", ["a"], 1)
    with pytest.raises(enc.RerankError, match="no reranker"):
        enc.RerankerChain(None, None, "boom")


def test_chain_from_settings_primary_load_failure(settings):
    def broken():
        raise OSError("model not found")

    chain = enc.RerankerChain.from_settings(
        settings, primary_factory=broken, fallback_factory=lambda: StubReranker("cohere")
    )
    assert chain.primary is None and chain.fallback_configured
    out = chain.rerank("q", ["a"], 1)
    assert out.fallback_used and "model not found" in out.primary_error


def test_chain_from_settings_without_fallback_key(settings):
    chain = enc.RerankerChain.from_settings(settings, primary_factory=lambda: StubReranker("qwen"))
    assert not chain.fallback_configured
    assert chain.rerank("q", ["a"], 1).kind == "qwen"


# --- startup check -------------------------------------------------------------------


def test_startup_check_requires_cuda(settings, tmp_path):
    with pytest.raises(enc.EncoderCheckError, match="CUDA"):
        enc.startup_check(settings, build_dir=tmp_path, cuda_available=lambda: False)


def test_startup_check_without_reference(settings, tmp_path):
    rep = enc.startup_check(settings, build_dir=tmp_path, cuda_available=lambda: True)
    assert rep == {"cuda": True, "reference_checked": False, "min_cosine": None}


class FixedEmbedder:
    def __init__(self, vecs):
        self.vecs = vecs

    def embed_documents(self, texts):
        assert len(texts) == len(self.vecs)
        return self.vecs


def write_reference(tmp_path, passages, vecs):
    d = tmp_path / "vectors" / "reference"
    d.mkdir(parents=True)
    (d / "passages.json").write_text(json.dumps(passages), encoding="utf-8")
    np.save(d / "vectors.npy", vecs)


def unit_rows(n, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 1024)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_startup_check_reference_match(settings, tmp_path):
    vecs = unit_rows(5)
    write_reference(tmp_path, ["a", {"header": "H", "text": "b"}, "c", "d", "e"], vecs * 3.0)
    rep = enc.startup_check(settings, FixedEmbedder(vecs), tmp_path, cuda_available=lambda: True)
    assert rep["reference_checked"] and rep["min_cosine"] == pytest.approx(1.0, abs=1e-5)


def test_startup_check_reference_mismatch(settings, tmp_path):
    write_reference(tmp_path, ["a", "b"], unit_rows(2, seed=1))
    with pytest.raises(enc.EncoderCheckError, match="min cosine"):
        enc.startup_check(
            settings, FixedEmbedder(unit_rows(2, seed=2)), tmp_path, cuda_available=lambda: True
        )


def test_startup_check_reference_bad_inputs(settings, tmp_path):
    write_reference(tmp_path, ["a", "b"], unit_rows(3))
    with pytest.raises(enc.EncoderCheckError, match="shape"):
        enc.startup_check(
            settings, FixedEmbedder(unit_rows(2)), tmp_path, cuda_available=lambda: True
        )
    with pytest.raises(enc.EncoderCheckError, match="no embedder"):
        enc.startup_check(settings, None, tmp_path, cuda_available=lambda: True)


def test_startup_check_bad_passage_entry(settings, tmp_path):
    write_reference(tmp_path, [{"nope": 1}], unit_rows(1))
    with pytest.raises(enc.EncoderCheckError, match="each entry"):
        enc.startup_check(
            settings, FixedEmbedder(unit_rows(1)), tmp_path, cuda_available=lambda: True
        )


def test_startup_check_on_the_cpu_deployment(settings, tmp_path):
    rep = enc.startup_check(
        settings, build_dir=tmp_path, cuda_available=lambda: False, require_cuda=False
    )
    assert rep == {"cuda": False, "reference_checked": False, "min_cosine": None}


def test_half_precision_only_on_the_gpu():
    import torch

    assert enc._dtype("cuda") is torch.float16
    assert enc._dtype("cuda:0") is torch.float16
    assert enc._dtype("cpu") is torch.float32


def test_device_switch_comes_from_the_environment(monkeypatch):
    from nyaya.config import ConfigError, load_settings

    monkeypatch.delenv("NYAYA_DEVICE", raising=False)
    assert load_settings(env_file=None).secrets.nyaya_device == "cuda"
    monkeypatch.setenv("NYAYA_DEVICE", "cpu")
    assert load_settings(env_file=None).secrets.nyaya_device == "cpu"
    monkeypatch.setenv("NYAYA_DEVICE", "tpu")
    with pytest.raises(ConfigError, match="nyaya_device"):
        load_settings(env_file=None)
