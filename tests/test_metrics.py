"""Tests for the metrics middleware and /metrics endpoint.

Builds a minimal FastAPI app with the middleware attached and fakes
upstream chat-completion JSON; no llama-server needed.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from forgemesh.metrics import Metrics, MetricsMiddleware


def _build_app(metrics: Metrics) -> FastAPI:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    @app.get("/metrics")
    async def get_metrics():
        return metrics.snapshot()

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat():
        return {
            "id": "a",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    @app.get("/v1/boom")
    async def boom():
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="nope")

    return app


def test_metrics_counts_requests_and_tokens():
    m = Metrics()
    app = _build_app(m)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.post("/v1/chat/completions", json={"x": 1}).status_code == 200
        assert c.post("/v1/chat/completions", json={"x": 1}).status_code == 200

        snap = c.get("/metrics").json()

    # /metrics snapshot is computed before its own request is recorded,
    # so we expect 3 here (healthz + 2 chat) not 4.
    assert snap["total"]["requests"] == 3
    assert snap["total"]["prompt_tokens"] == 20
    assert snap["total"]["completion_tokens"] == 40

    chat_stats = snap["routes"]["/v1/chat/completions"]
    assert chat_stats["count"] == 2
    assert chat_stats["prompt_tokens"] == 20
    assert chat_stats["completion_tokens"] == 40
    assert chat_stats["avg_latency_ms"] > 0


def test_metrics_tracks_errors():
    import contextlib
    import warnings

    m = Metrics()
    app = _build_app(m)
    with TestClient(app, raise_server_exceptions=False) as c, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.suppress(Exception):
            c.get("/v1/boom")

    snap = m.snapshot()
    boom_stats = snap["routes"].get("/v1/boom", {})
    assert boom_stats.get("count", 0) >= 1


def test_metrics_snapshot_shape():
    m = Metrics()
    snap = m.snapshot()
    assert "uptime_s" in snap
    assert "total" in snap
    assert "routes" in snap
    assert set(snap["total"].keys()) == {
        "requests",
        "errors",
        "prompt_tokens",
        "completion_tokens",
    }


def test_metrics_middleware_ignores_non_json_bodies():
    """Responses without application/json should not error, and token counts stay zero."""
    m = Metrics()
    app = FastAPI()
    app.add_middleware(MetricsMiddleware, metrics=m)

    @app.post("/v1/chat/completions")
    async def chat():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("hello")

    with TestClient(app) as c:
        r = c.post("/v1/chat/completions")
        assert r.status_code == 200

    snap = m.snapshot()
    assert snap["routes"]["/v1/chat/completions"]["completion_tokens"] == 0


def test_metrics_endpoint_reports_snapshot_directly():
    m = Metrics()
    m.record(path="/v1/chat/completions", status=200, latency_ms=50.0,
             prompt_tokens=3, completion_tokens=7)
    snap = m.snapshot()
    assert snap["total"]["prompt_tokens"] == 3
    assert snap["total"]["completion_tokens"] == 7
    r = snap["routes"]["/v1/chat/completions"]
    assert r["count"] == 1
    assert r["avg_latency_ms"] == 50.0


def test_bench_summary_dict_shape():
    """Smoke: BenchSummary.as_dict without hitting a network."""
    from forgemesh.bench import BenchRunResult, BenchSummary

    results = [
        BenchRunResult(
            run=1, ok=True, status=200, wall_ms=1000.0,
            prompt_tokens=10, completion_tokens=50,
            wall_tps=50.0, server_predicted_tps=60.0,
        ),
        BenchRunResult(
            run=2, ok=False, status=500, wall_ms=500.0,
            prompt_tokens=0, completion_tokens=0,
            wall_tps=0.0, server_predicted_tps=None, error="boom",
        ),
    ]
    s = BenchSummary(
        endpoint="http://x", model="m", runs=2, ok_runs=1,
        max_tokens=256, prompt="p", results=results,
    )
    d = s.as_dict()
    assert d["runs"] == 2
    assert d["ok_runs"] == 1
    assert d["wall_tps"]["mean"] == 50.0
    assert len(d["per_run"]) == 2
    assert d["per_run"][1]["ok"] is False
    _ = json  # linter placation
