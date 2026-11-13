"""In-memory SearchEngine, embedder and reranker for search and loader tests."""

from nyaya.contracts import ChunkRecord, SearchHit, Zone
from nyaya.models.encoders import RerankOutcome, top_results


def make_chunk(path, zone="judgment", seq=1, text="text", header="Header"):
    return ChunkRecord(
        chunk_id=f"{path}#{zone}#{seq:03d}",
        judgment_id=path,
        zone=Zone(zone),
        para_labels=[str(seq)],
        para_refs=[],
        text=text,
        header=header,
        char_start=0,
        char_end=len(text),
        pages=[1],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )


class FakeEngine:
    """Keyword = chunks whose text contains any question word; vector = fixed order."""

    def __init__(self, chunks, links=None, aliases=None, vector_order=None):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.links = links or {}
        self.aliases = aliases or {}
        self.vector_order = vector_order if vector_order is not None else list(self.chunks)
        self.calls = []

    def keyword(self, question, filters, k):
        self.calls.append(("keyword", question, filters, k))
        words = set(question.lower().split())
        ids = [c for c, ch in self.chunks.items() if words & set(ch.text.lower().split())]
        return [
            SearchHit(chunk_id=c, keyword_rank=i, fused_rank=i) for i, c in enumerate(ids[:k], 1)
        ]

    def vector(self, qvec, filters, k, exact=False):
        self.calls.append(("vector", filters, k, exact))
        ids = self.vector_order[:k]
        return [SearchHit(chunk_id=c, vector_rank=i, fused_rank=i) for i, c in enumerate(ids, 1)]

    def get_chunks(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]

    def get_judgment(self, path):
        return None

    def resolve_alias(self, canonical):
        return self.aliases.get(canonical)

    def links_from(self, chunk_ids):
        return {c: self.links[c] for c in chunk_ids if c in self.links}


class FakeEmbedder:
    def __init__(self):
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return [0.0] * 4


class WordCountReranker:
    """Score = share of question words found in the document (0..1)."""

    def __init__(self, kind="qwen", fallback_used=False):
        self.kind = kind
        self.fallback_used = fallback_used
        self.calls = []

    def rerank(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        words = set(query.lower().split())
        scores = [len(words & set(d.lower().split())) / max(len(words), 1) for d in documents]
        return RerankOutcome(
            top_results(scores, top_n), self.kind, f"{self.kind}-m", self.fallback_used
        )
