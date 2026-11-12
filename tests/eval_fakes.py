"""Fakes for the evaluation tests: a tiny in-memory SearchEngine."""

from __future__ import annotations

from nyaya.contracts import ChunkRecord, SearchHit


def chunk(cid: str, labels=(), refs=(), text: str = "text", header: str = "h") -> ChunkRecord:
    jid, zone, _ = cid.split("#")
    return ChunkRecord(
        chunk_id=cid,
        judgment_id=jid,
        zone=zone,
        para_labels=list(labels),
        para_refs=list(refs),
        text=text,
        header=header,
        char_start=0,
        char_end=1,
        pages=[1],
        flags=[],
        clean_text_version="ct1",
        chunker_version="ch1",
    )


def hits(ids, kind="keyword") -> list[SearchHit]:
    out = []
    for r, cid in enumerate(ids, start=1):
        kw = {"keyword_rank": r} if kind == "keyword" else {"vector_rank": r}
        out.append(SearchHit(chunk_id=cid, fused_rank=float(r), **kw))
    return out


class FakeEngine:
    def __init__(self, chunks, keyword=None, vector=None, links=None):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.keyword_map = keyword or {}
        self.vector_map = vector or {}
        self.links = links or {}
        self.calls: list[tuple] = []

    def keyword(self, question, filters, k):
        self.calls.append(("keyword", question, k))
        return hits(self.keyword_map.get(question, [])[:k])

    def vector(self, qvec, filters, k, exact):
        self.calls.append(("vector", tuple(qvec), k, exact))
        key = (tuple(qvec), exact)
        ids = self.vector_map.get(key, self.vector_map.get(tuple(qvec), []))
        return hits(ids[:k], kind="vector")

    def get_chunks(self, ids):
        return [self.chunks[i] for i in ids if i in self.chunks]

    def links_from(self, chunk_ids):
        return {c: list(self.links[c]) for c in chunk_ids if c in self.links}


def simple_rrf(lists, k):
    scores: dict[str, float] = {}
    for lst in lists:
        for r, h in enumerate(lst, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + r)
    order = sorted(scores, key=lambda c: -scores[c])
    return [SearchHit(chunk_id=c, fused_rank=float(i)) for i, c in enumerate(order, start=1)]
