"""The `SearchEngine` protocol (design file 30).

Search sits behind this interface so evaluation can swap engines (the G4 size
steps, a BM25 comparison) and tests can use fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from nyaya.contracts import ChunkRecord, Filters, JudgmentRecord, SearchHit


@runtime_checkable
class SearchEngine(Protocol):
    def keyword(self, question: str, filters: Filters | None, k: int) -> list[SearchHit]:
        """Full-text search, best first; hits carry `keyword_rank` (1-based)."""
        ...

    def vector(
        self, qvec: Sequence[float], filters: Filters | None, k: int, exact: bool = False
    ) -> list[SearchHit]:
        """Nearest chunks by cosine distance; hits carry `vector_rank` (1-based).

        `exact=True` runs the same query with index scans disabled (G4).
        """
        ...

    def get_chunks(self, ids: Sequence[str]) -> list[ChunkRecord]:
        """The chunks with these IDs, in the order given; unknown IDs are skipped."""
        ...

    def get_judgment(self, path: str) -> JudgmentRecord | None: ...

    def resolve_alias(self, canonical: str) -> str | None:
        """The loaded judgment `path` for a canonical citation, or None."""
        ...

    def links_from(self, chunk_ids: Sequence[str]) -> dict[str, list[str]]:
        """Held links: {from_chunk_id: [to_chunk_id, ...]} for the IDs that have any."""
        ...
