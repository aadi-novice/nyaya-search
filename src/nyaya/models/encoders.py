"""Embedder, rerankers and the startup encoder check (design file 27).

* `Embedder`: Qwen3-Embedding-0.6B, fp16 on CUDA, normalised output. Queries
  get `embed.query_prefix`; documents ("header\\n\\ntext") get no prefix.
* `QwenReranker`: Qwen3-Reranker-0.6B through `transformers`
  (`AutoModelForCausalLM`, fp16). Each (question, header + text) pair is
  formatted with the model-card template and `rerank.instruction`; the score
  is the softmax probability of "yes" against "no" at the last position.
* `CohereFallbackReranker`: Cohere Rerank through OpenRouter
  (`POST https://openrouter.ai/api/v1/rerank`, httpx).
* `RerankerChain`: the primary, with the fallback used only when the primary
  failed to load or raises.
* `startup_check()`: CUDA is present, and (for a Kaggle-embedded index) the
  reference vectors match the local embedder.

The model objects are injectable, so the logic is tested without a GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from nyaya.config import Settings

log = logging.getLogger(__name__)

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"

# Qwen3-Reranker model-card template.
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
RERANK_PREFIX = f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n"
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def document_text(header: str, text: str) -> str:
    """The text embedded and reranked for a chunk: header, a blank line, the text."""
    return f"{header}\n\n{text}" if header else text


def _revision(rev: str) -> str | None:
    return rev or None


def _dtype(device: str) -> Any:
    """float16 on the GPU; float32 on the CPU, where half precision is slow or unsupported."""
    import torch

    return torch.float16 if device.startswith("cuda") else torch.float32


# --- embedder ----------------------------------------------------------------------


class _Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class Embedder:
    """Qwen3-Embedding: `embed_documents` (no prefix) and `embed_query` (with prefix)."""

    def __init__(self, settings: Settings, model: _Encoder | None = None, device: str = "cuda"):
        self.settings = settings
        self.model_name = settings.embed.model
        self.query_prefix = settings.embed.query_prefix
        self.dim = settings.embed.dim
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.model_name,
                revision=_revision(settings.embed.revision),
                device=device,
                model_kwargs={"dtype": _dtype(device)},
                processor_kwargs={"padding_side": "left"},
            )
        self.model = model

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vecs = np.asarray(out, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"embedder returned shape {vecs.shape}, expected (n, {self.dim})")
        return vecs

    def query_text(self, question: str) -> str:
        return self.query_prefix + question

    def embed_query(self, question: str) -> np.ndarray:
        """One normalised float32 vector for a question, with the instruction prefix."""
        return self._encode([self.query_text(question)], batch_size=1)[0]

    def embed_documents(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Normalised float32 vectors, one row per text, without any prefix."""
        return self._encode(list(texts), batch_size=batch_size)


# --- rerankers ---------------------------------------------------------------------


@dataclass(frozen=True)
class RerankResult:
    index: int  # position in the documents passed
    score: float


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    kind: str  # "qwen" | "cohere": picks the refusal threshold
    model_name: str

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]: ...


def top_results(scores: Sequence[float], top_n: int) -> list[RerankResult]:
    """The best `top_n` (index, score), score descending, ties by index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [RerankResult(i, float(scores[i])) for i in order[:top_n]]


def format_rerank_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def yes_probability(last_logits: Any, yes_id: int, no_id: int) -> list[float]:
    """Softmax over the ("no", "yes") logits at the last position; P("yes") per row."""
    import torch

    pair = torch.stack([last_logits[:, no_id], last_logits[:, yes_id]], dim=1).float()
    return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()


# PyTorch scaled-dot-product attention for the reranker (decision 2026-11-13).
ATTN_IMPLEMENTATION: str | None = "sdpa"


class QwenReranker:
    """Qwen3-Reranker-0.6B, local yes/no scoring (model-card recipe)."""

    kind = "qwen"

    def __init__(
        self,
        settings: Settings,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cuda",
        batch_size: int | None = None,
        max_length: int = 8192,
        attn_implementation: str | None = ATTN_IMPLEMENTATION,
    ) -> None:
        self.model_name = settings.rerank.model
        self.instruction = settings.rerank.instruction
        self.batch_size = batch_size or settings.encoders.rerank_batch_size
        self.max_length = max_length
        self.device = device
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            rev = _revision(settings.rerank.revision)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=rev, padding_side="left"
            )
            extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, revision=rev, dtype=_dtype(device), **extra
                )
                .to(device)
                .eval()
            )
        self.tokenizer = tokenizer
        self.model = model
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)

    def _batch_inputs(self, pairs: list[str]) -> Any:
        room = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=room,
        )
        enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
        batch = self.tokenizer.pad(enc, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        """P("yes") for each document, in input order."""
        import torch

        pairs = [format_rerank_pair(self.instruction, query, d) for d in documents]
        # shortest first, to keep padding small; scores are put back in input order
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]))
        out = [0.0] * len(pairs)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                inputs = self._batch_inputs([pairs[i] for i in idx])
                # logits_to_keep=1: only the last position goes through the
                # 151k-word LM head; full logits cost ~1 GB per batch of 4
                logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
                for i, p in zip(idx, yes_probability(logits, self.yes_id, self.no_id), strict=True):
                    out[i] = p
        return out

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        return top_results(self.scores(query, documents), top_n)


def parse_rerank_response(data: Any, n_documents: int) -> list[RerankResult]:
    """OpenRouter rerank JSON -> results, best first.

    Confirmed shape (2026-11-12): {"id", "model", "results": [{"index": int,
    "relevance_score": float, "document": {"text": str}}], "usage": {...},
    "provider": "Cohere"}; results arrive sorted by relevance_score.
    """
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise RerankError(f"unexpected rerank response: {str(data)[:200]}")
    out: list[RerankResult] = []
    for item in data["results"]:
        try:
            index = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(f"bad rerank result item: {item!r}") from exc
        if not 0 <= index < n_documents:
            raise RerankError(f"rerank index {index} out of range 0..{n_documents - 1}")
        out.append(RerankResult(index, score))
    out.sort(key=lambda r: (-r.score, r.index))
    return out


class CohereFallbackReranker:
    """Cohere Rerank through OpenRouter, sent with httpx."""

    kind = "cohere"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
        url: str = OPENROUTER_RERANK_URL,
    ) -> None:
        if not api_key:
            raise RerankError("OPENROUTER_API_KEY is not set; no fallback reranker")
        self.model_name = model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.url = url
        self.client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> CohereFallbackReranker:
        key = settings.secrets.openrouter_api_key
        return cls(key.get_secret_value() if key else "", settings.rerank.fallback_model, **kwargs)

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        body = {
            "model": self.model_name,
            "query": query,
            "documents": list(documents),
            "top_n": min(top_n, len(documents)),
        }
        try:
            resp = self.client.post(
                self.url,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"OpenRouter rerank request failed: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise RerankError(
                f"OpenRouter rerank returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise RerankError("OpenRouter rerank returned invalid JSON") from exc
        return parse_rerank_response(data, len(documents))[:top_n]


@dataclass(frozen=True)
class RerankOutcome:
    results: list[RerankResult]
    kind: str  # "qwen" | "cohere"
    model_name: str
    fallback_used: bool
    primary_error: str | None = None


class RerankerChain:
    """The primary reranker, falling back only when it failed to load or raises."""

    def __init__(
        self,
        primary: Reranker | None,
        fallback: Reranker | None,
        primary_error: str | None = None,
    ) -> None:
        if primary is None and fallback is None:
            raise RerankError(f"no reranker available (primary: {primary_error})")
        self.primary = primary
        self.fallback = fallback
        self.primary_error = primary_error

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        primary_factory: Callable[[], Reranker] | None = None,
        fallback_factory: Callable[[], Reranker] | None = None,
    ) -> RerankerChain:
        primary: Reranker | None = None
        error: str | None = None
        try:
            primary = (primary_factory or (lambda: QwenReranker(settings)))()
        except Exception as exc:  # any load failure means the fallback is used
            error = f"{type(exc).__name__}: {exc}"
            log.warning("local reranker failed to load; using the fallback: %s", error)
        fallback: Reranker | None = None
        try:
            fallback = (
                fallback_factory or (lambda: CohereFallbackReranker.from_settings(settings))
            )()
        except Exception as exc:
            log.warning("fallback reranker not configured: %s", exc)
        return cls(primary, fallback, error)

    @property
    def fallback_configured(self) -> bool:
        return self.fallback is not None

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> RerankOutcome:
        error = self.primary_error
        if self.primary is not None:
            try:
                results = self.primary.rerank(query, documents, top_n)
                return RerankOutcome(results, self.primary.kind, self.primary.model_name, False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log.warning("local reranker raised; using the fallback: %s", error)
        if self.fallback is None:
            raise RerankError(f"primary reranker unavailable ({error}) and no fallback configured")
        results = self.fallback.rerank(query, documents, top_n)
        return RerankOutcome(results, self.fallback.kind, self.fallback.model_name, True, error)


# --- startup checks -------------------------------------------------------------------


class EncoderCheckError(RuntimeError):
    pass


REFERENCE_DIR = Path("vectors") / "reference"


def _reference_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for item in data:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "text" in item:
            texts.append(document_text(item.get("header", ""), item["text"]))
        else:
            raise EncoderCheckError(f"{path}: each entry must be a string or have 'text'")
    return texts


def startup_check(
    settings: Settings,
    embedder: Embedder | None = None,
    build_dir: Path | str | None = None,
    *,
    cuda_available: Callable[[], bool] | None = None,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Assert CUDA (unless `require_cuda` is False: the CPU deployment), then compare
    Kaggle reference vectors when they exist.

    Returns a small report: {"cuda": bool, "reference_checked": bool,
    "min_cosine": float | None}. Raises EncoderCheckError on failure.
    """
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available
    cuda = bool(cuda_available())
    if require_cuda and not cuda:
        raise EncoderCheckError("CUDA is not available; the encoders need the local GPU")
    build = Path(build_dir) if build_dir is not None else Path(settings.paths.build)
    ref_dir = build / REFERENCE_DIR
    passages, vectors = ref_dir / "passages.json", ref_dir / "vectors.npy"
    if not (passages.exists() and vectors.exists()):
        return {"cuda": cuda, "reference_checked": False, "min_cosine": None}
    if embedder is None:
        raise EncoderCheckError("reference vectors exist but no embedder was given")
    texts = _reference_texts(passages)
    ref = np.load(vectors).astype(np.float32)
    if ref.shape != (len(texts), settings.embed.dim):
        raise EncoderCheckError(
            f"reference vectors have shape {ref.shape},"
            f" expected ({len(texts)}, {settings.embed.dim})"
        )
    local = embedder.embed_documents(texts)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (local * ref).sum(axis=1)
    worst = float(cos.min())
    if worst < settings.encoder_check.min_cosine:
        raise EncoderCheckError(
            f"local embedder disagrees with the Kaggle reference vectors: min cosine {worst:.5f}"
            f" < {settings.encoder_check.min_cosine}"
        )
    return {"cuda": cuda, "reference_checked": True, "min_cosine": worst}
