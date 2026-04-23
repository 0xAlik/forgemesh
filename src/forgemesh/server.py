"""FastAPI application — OpenAI-compatible proxy in front of llama-server.

Responsibilities:
  - API-key auth on the `/v1/*` endpoints
  - Transparent pass-through to the underlying llama-server (streaming too)
  - Rewrite the `model` field in responses so clients see a stable name
  - /healthz independent of the backend
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from forgemesh import __version__
from forgemesh.auth import make_auth_dependency
from forgemesh.config import Config
from forgemesh.llama_server import LlamaServer
from forgemesh.metrics import Metrics, MetricsMiddleware

log = logging.getLogger(__name__)


# Proxy config assembled in `create_app`; only one model served in v0.0.1.
class _AppState:
    config: Config
    llama: LlamaServer
    model_name: str
    api_key: str
    http: httpx.AsyncClient
    metrics: Metrics


def create_app(state: _AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.http = httpx.AsyncClient(
            base_url=state.llama.base_url,
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
        )
        try:
            yield
        finally:
            await state.http.aclose()

    app = FastAPI(
        title="ForgeMesh",
        version=__version__,
        description="Self-hosted, OpenAI-compatible LLM inference.",
        lifespan=lifespan,
    )
    app.add_middleware(MetricsMiddleware, metrics=state.metrics)

    require_auth = make_auth_dependency(state.api_key, enabled=state.config.auth.enabled)

    @app.get("/")
    async def root():
        return {"name": "forgemesh", "version": __version__, "model": state.model_name}

    @app.get("/healthz")
    async def healthz():
        upstream_ok = state.llama.is_running()
        return JSONResponse(
            {"ok": upstream_ok, "model": state.model_name, "version": __version__},
            status_code=200 if upstream_ok else 503,
        )

    @app.get("/metrics")
    async def metrics():
        return state.metrics.snapshot()

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "forgemesh",
                }
            ],
        }

    async def _proxy(request: Request, upstream_path: str) -> StreamingResponse | JSONResponse:
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "authorization", "content-length"}
        }
        try:
            upstream_req = state.http.build_request(
                method=request.method,
                url=upstream_path,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
            upstream = await state.http.send(upstream_req, stream=True)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"upstream unavailable: {e}") from e

        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }

        async def stream_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(request: Request):
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions", dependencies=[Depends(require_auth)])
    async def completions(request: Request):
        return await _proxy(request, "/v1/completions")

    @app.post("/v1/embeddings", dependencies=[Depends(require_auth)])
    async def embeddings(request: Request):
        return await _proxy(request, "/v1/embeddings")

    return app


def build_state(config: Config, model_path, model_name: str, api_key: str) -> _AppState:
    state = _AppState()
    state.config = config
    state.llama = LlamaServer(config=config, model_path=model_path)
    state.model_name = model_name
    state.api_key = api_key
    state.metrics = Metrics()
    return state
