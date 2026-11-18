"""BM25 keyword search over one index version's chunks (design: "Hybrid
retrieval", BM25 trigger; decision 2026-11-17).

The BM25 trigger fired on the development set, and ParadeDB `pg_search`
cannot be used without replacing the live database image, so the keyword leg
is `bm25s` (the trigger's own library) in the API process. Postgres stays the
store of record: it supplies the chunk texts, applies the filters and finds
quoted phrases; this module only scores.

*   **Text:** `header || ' ' || text`, the string the `tsv` column indexes.
*   **Tokens:** bm25s's tokenizer (lower case, words of 2+ letters or digits,
    English stop words), not stemmed. `retrieval.bm25_stemmer = "english"`
    adds the Snowball stemmer (PyStemmer, not installed by default): on the
    44 development questions it moved judgment recall@50 from 0.80 to 0.82
    but MRR from 0.49 to 0.42, so it stays off.
*   **Files:** `<paths.build>/bm25/<index_version>/`: the bm25s index (loaded
    memory-mapped), `ids.json` (chunk IDs in index order, by chunk_id) and
    `meta.json`. They are built at load time; an engine that finds none, or a
    stale one, builds them from the database and saves them.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FORMAT = 1
STEMMERS = ("english", "")
_BATCH = 5000


def _stem_fn(stemmer: str) -> Callable[[str], str] | None:
    if stemmer not in STEMMERS:
        raise ValueError(f"unknown BM25 stemmer {stemmer!r}; expected one of {STEMMERS}")
    if not stemmer:
        return None
    import Stemmer  # PyStemmer: Snowball, the same family as Postgres's 'english' config

    return Stemmer.Stemmer(stemmer).stemWord


def _tokenizer(stemmer: str) -> Any:
    from bm25s.tokenization import Tokenizer

    return Tokenizer(stopwords="en", stemmer=_stem_fn(stemmer))


@dataclass(frozen=True)
class Bm25Meta:
    index_version: str
    chunks: int
    stemmer: str
    k1: float
    b: float
    format: int = FORMAT


class Bm25Index:
    """A bm25s index plus the chunk ID of each of its documents."""

    def __init__(self, ids: Sequence[str], retriever: Any, meta: Bm25Meta) -> None:
        if len(ids) != meta.chunks:
            raise ValueError(f"{len(ids)} ids for {meta.chunks} indexed chunks")
        self.ids = list(ids)
        self.position = {cid: i for i, cid in enumerate(self.ids)}
        self.retriever = retriever
        self.meta = meta
        _stem_fn(meta.stemmer)  # fail early on an unknown stemmer

    # --- build, save, load ------------------------------------------------------------

    @classmethod
    def build(
        cls,
        rows: Iterable[tuple[str, str]],
        *,
        index_version: str,
        stemmer: str = "",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> Bm25Index:
        """Index `(chunk_id, text)` rows. Token IDs are kept as int32 arrays per chunk,
        so the 296k-chunk corpus fits in about 1 GB while it is indexed."""
        import bm25s

        tok = _tokenizer(stemmer)
        ids: list[str] = []
        docs: list[np.ndarray] = []
        batch: list[str] = []

        def flush() -> None:
            out = tok.tokenize(batch, update_vocab=True, return_as="ids", show_progress=False)
            docs.extend(np.asarray(d, dtype=np.int32) for d in out)
            batch.clear()

        for cid, text in rows:
            ids.append(cid)
            batch.append(text or "")
            if len(batch) >= _BATCH:
                flush()
        if batch:
            flush()
        if not ids:
            raise ValueError("no chunks to index")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids")
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index((docs, tok.get_vocab_dict()), show_progress=False)
        meta = Bm25Meta(index_version=index_version, chunks=len(ids), stemmer=stemmer, k1=k1, b=b)
        return cls(ids, retriever, meta)

    def save(self, directory: Path | str) -> Path:
        """Write the index to `directory` (replaced whole, through a sibling temp dir)."""
        out = Path(directory)
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.retriever.save(str(tmp), show_progress=False)
        (tmp / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (tmp / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2), encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        return out

    @classmethod
    def load(cls, directory: Path | str, *, mmap: bool = True) -> Bm25Index:
        import bm25s

        d = Path(directory)
        meta = Bm25Meta(**json.loads((d / "meta.json").read_text(encoding="utf-8")))
        if meta.format != FORMAT:
            raise ValueError(f"BM25 index format {meta.format}, expected {FORMAT}")
        ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        retriever = bm25s.BM25.load(str(d), mmap=mmap, show_progress=False)
        return cls(ids, retriever, meta)

    # --- search -----------------------------------------------------------------------

    def query_tokens(self, question: str) -> list[int]:
        """Index token IDs of the question; words the index never saw are dropped.

        A fresh tokenizer per call: bm25s's tokenizer only returns words already
        in its own vocabulary unless it may add them, and a shared one would
        grow with every query (and is not thread-safe while it does).
        """
        tok = _tokenizer(self.meta.stemmer)
        words = tok.tokenize([question], update_vocab=True, return_as="string", show_progress=False)
        return self.retriever.get_tokens_ids([w for w in words[0] if w])

    def mask(self, chunk_ids: Iterable[str]) -> np.ndarray:
        """A boolean mask over the index for these chunk IDs (unknown IDs ignored)."""
        m = np.zeros(len(self.ids), dtype=bool)
        pos = [self.position[c] for c in chunk_ids if c in self.position]
        if pos:
            m[np.asarray(pos)] = True
        return m

    def search(
        self,
        question: str,
        k: int,
        *,
        allowed: np.ndarray | None = None,
        first: np.ndarray | None = None,
    ) -> list[str]:
        """Chunk IDs, best BM25 score first, at most `k`.

        `allowed` (a mask) limits the result to the filtered chunks. Chunks in
        `first` (a mask: the quoted-phrase matches) come before all others,
        in score order, even when they share no scored word with the question.
        Only chunks with a positive score (or in `first`) are returned; ties
        are broken by chunk ID order.
        """
        if k <= 0:
            return []
        tokens = self.query_tokens(question)
        n = len(self.ids)
        scores = (
            np.asarray(self.retriever.get_scores(tokens), dtype=np.float64)
            if tokens
            else np.zeros(n)
        )
        keep = scores > 0
        if first is not None:
            keep |= first
        if allowed is not None:
            keep &= allowed
        cand = np.flatnonzero(keep)
        if cand.size == 0:
            return []
        tier = first[cand] if first is not None else np.zeros(cand.size, dtype=bool)
        # lexsort: last key is primary -> phrase tier, then score, then position
        order = np.lexsort((cand, -scores[cand], ~tier))
        return [self.ids[i] for i in cand[order[:k]]]


# --- files under build/ -----------------------------------------------------------------


def index_dir(build_root: Path | str, index_version: str) -> Path:
    return Path(build_root) / "bm25" / index_version


def stream_chunks(conn: Any, index_version: str) -> Iterator[tuple[str, str]]:
    """`(chunk_id, header || ' ' || text)` for one version, by chunk_id, streamed."""
    with conn.transaction(), conn.cursor(name="nyaya_bm25_chunks") as cur:
        cur.itersize = _BATCH
        cur.execute(
            "SELECT chunk_id, header || ' ' || text FROM chunks"
            " WHERE index_version = %s ORDER BY chunk_id",
            (index_version,),
        )
        yield from cur


def chunk_count(conn: Any, index_version: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE index_version = %s", (index_version,)
    ).fetchone()
    return int(row[0])


def build_index(
    conn: Any, index_version: str, *, stemmer: str, directory: Path | None
) -> Bm25Index:
    """Build from the database; save to `directory` unless it is None."""
    index = Bm25Index.build(
        stream_chunks(conn, index_version), index_version=index_version, stemmer=stemmer
    )
    if directory is not None:
        index.save(directory)
        log.info(
            "bm25: indexed %d chunks of %s into %s", index.meta.chunks, index_version, directory
        )
    return index


def open_index(conn: Any, index_version: str, *, stemmer: str, directory: Path | None) -> Bm25Index:
    """The saved index for `index_version` if it is current, else a fresh build.

    "Current" means the same version, stemmer, format and chunk count as the
    database. `directory` None builds in memory only (temporary G4 schemas).
    """
    if directory is not None and (directory / "meta.json").exists():
        # the meta is checked before the (memory-mapped) load, so a stale
        # directory is never open while it is replaced
        try:
            meta = Bm25Meta(**json.loads((directory / "meta.json").read_text(encoding="utf-8")))
            current = (
                meta.format == FORMAT
                and meta.index_version == index_version
                and meta.stemmer == stemmer
                and meta.chunks == chunk_count(conn, index_version)
            )
            if current:
                return Bm25Index.load(directory)
            log.warning("bm25: %s is stale (%s); rebuilding", directory, meta)
        except Exception as exc:  # a torn or old index is rebuilt, never trusted
            log.warning("bm25: cannot load %s (%s); rebuilding", directory, exc)
    return build_index(conn, index_version, stemmer=stemmer, directory=directory)


def keyword_index(
    settings: Any, conn: Any, index_version: str, *, save: bool = True
) -> Bm25Index | None:
    """The BM25 index the engine should use, or None when `retrieval.keyword_engine`
    is "postgres". `save=False` keeps a fresh build in memory (temporary schemas)."""
    from nyaya.config import REPO_ROOT

    r = settings.retrieval
    if r.keyword_engine != "bm25":
        return None
    root = Path(settings.paths.build)
    if not root.is_absolute():
        root = REPO_ROOT / root
    directory = index_dir(root, index_version) if save else None
    return open_index(conn, index_version, stemmer=r.bm25_stemmer, directory=directory)
