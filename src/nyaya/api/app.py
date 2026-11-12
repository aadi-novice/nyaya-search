"""FastAPI application, `nyaya-api` (design file 37).

Endpoints:
    POST /ask                                AskRequest -> AskResponse
    GET  /judgment/{path}?highlight=chunk_id JudgmentRecord + the chunk's span
    GET  /health                             active version, models, fallback
    /                                        the static page in web/

Each /ask runs in a worker thread under `asyncio.wait_for` with the
`timeouts.request_s` limit. On a timeout or an exception the response has
status `error` and the API writes the trace itself; on success the pipeline
writes it. The server binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nyaya.config import Settings, load_settings
from nyaya.contracts import (
    AskRequest,
    AskResponse,
    ChunkRecord,
    JudgmentRecord,
    Route,
    Status,
    Versions,
)

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

ERROR_NOTICE_TIMEOUT = "The request took longer than {s} seconds and was stopped. Please try again."
ERROR_NOTICE_FAILURE = "The request failed because of an internal error. Please try again."


class Backend(Protocol):
    """What the API needs from the rest of the system."""

    def ask(self, request: AskRequest) -> AskResponse: ...
    def get_judgment(self, path: str) -> JudgmentRecord | None: ...
    def get_chunk(self, chunk_id: str) -> ChunkRecord | None: ...
    def health(self) -> dict[str, Any]: ...
    def versions(self) -> Versions: ...
    def write_trace(
        self, request: AskRequest, response: AskResponse, error: str | None
    ) -> None: ...


class PipelineBackend:
    """The real backend: `nyaya.pipeline.ask(request, deps)` and the search engine.

    Integration assumptions (see the pipeline and trace modules):
      * `nyaya.pipeline.build_deps(settings)` loads the models and the engine
        once and runs the startup checks;
      * `deps.engine` is the SearchEngine (`get_judgment`, `get_chunks`);
      * `nyaya.trace.write_trace(deps, request, response, error=...)`.
    """

    def __init__(self, settings: Settings) -> None:
        from nyaya import pipeline

        self._pipeline = pipeline
        self.settings = settings
        self.deps = pipeline.build_deps(settings)

    def ask(self, request: AskRequest) -> AskResponse:
        return self._pipeline.ask(request, self.deps)

    def get_judgment(self, path: str) -> JudgmentRecord | None:
        return self.deps.engine.get_judgment(path)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        chunks = self.deps.engine.get_chunks([chunk_id])
        return chunks[0] if chunks else None

    def health(self) -> dict[str, Any]:
        return dict(self.deps.health())

    def versions(self) -> Versions:
        return self.deps.versions()

    def write_trace(self, request: AskRequest, response: AskResponse, error: str | None) -> None:
        from nyaya import trace

        trace.write_trace(self.deps, request, response, error=error)


def build_backend(settings: Settings) -> Backend:
    return PipelineBackend(settings)


def _error_response(
    backend: Backend,
    settings: Settings,
    request_id: str,
    request: AskRequest,
    notice: str,
    elapsed_ms: float,
) -> AskResponse:
    try:
        versions = backend.versions()
    except Exception:
        log.exception("could not read versions for an error response")
        versions = Versions(
            index_version="unknown",
            config_version=settings.versions.config,
            prompt_version=settings.versions.prompt,
            llm_model=settings.secrets.llm_model or "unknown",
            reranker_model=settings.rerank.model,
            reranker_fallback_used=False,
            embed_model=settings.embed.model,
        )
    filters = request.filters.model_dump(exclude_none=True) if request.filters else {}
    return AskResponse(
        request_id=request_id,
        status=Status.ERROR,
        notices=[notice, *settings.notices],
        route=Route.HYBRID,
        filters_applied=filters,
        versions=versions,
        timings_ms={"total": elapsed_ms},
    )


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept.split(",")[0]


def create_app(
    backend: Backend,
    settings: Settings | None = None,
    request_timeout_s: float | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    timeout_s = float(request_timeout_s or settings.timeouts.request_s)

    app = FastAPI(title="Nyaya Search", docs_url=None, redoc_url=None)

    @app.post("/ask", response_model=AskResponse, response_model_exclude_none=False)
    async def ask(request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(asyncio.to_thread(backend.ask, request), timeout_s)
        except TimeoutError:
            notice = ERROR_NOTICE_TIMEOUT.format(s=f"{timeout_s:g}")
            error = f"timeout after {timeout_s:g} s"
        except Exception as exc:
            log.exception("pipeline failed")
            notice = ERROR_NOTICE_FAILURE
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        resp = _error_response(
            backend, settings, uuid.uuid4().hex, request, notice, round(elapsed_ms, 1)
        )
        try:
            await asyncio.to_thread(backend.write_trace, request, resp, error)
        except Exception:
            log.exception("trace write failed for request %s", resp.request_id)
        return resp

    @app.get("/judgment/{path:path}", response_model=None)
    def judgment(path: str, request: Request, highlight: str | None = None) -> Any:
        if _wants_html(request):
            # A citation link opened in the browser: the page renders the
            # judgment by fetching this same URL as JSON.
            return FileResponse(WEB_DIR / "index.html", media_type="text/html")
        record = backend.get_judgment(path)
        if record is None:
            raise HTTPException(404, f"judgment not found: {path}")
        span = None
        if highlight:
            chunk = backend.get_chunk(highlight)
            if chunk is None or chunk.judgment_id != path:
                raise HTTPException(404, f"chunk not found in {path}: {highlight}")
            span = {
                "chunk_id": chunk.chunk_id,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
        return {**record.model_dump(mode="json"), "highlight": span}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **backend.health(), "notices": settings.notices}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nyaya-api", description="Run the Nyaya Search API.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    app = create_app(build_backend(settings), settings=settings)
    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
