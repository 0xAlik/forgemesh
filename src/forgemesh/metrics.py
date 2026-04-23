"""In-memory request metrics.

Deliberately minimal for v0.0.1:
  - counts per route and HTTP status bucket
  - total observed latency (ms) per route, for computing simple averages
  - total prompt/completion tokens observed on /v1/chat/completions and
    /v1/completions responses (parsed out of the JSON body when present)
  - recent-latency ring buffer for rough p50/p95 without a full histogram

No persistence. Resets on process restart. A dashboard or Prometheus
exporter would read from here in a future version.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from starlette.types import ASGIApp, Receive, Scope, Send

_TOKEN_ENDPOINTS = {"/v1/chat/completions", "/v1/completions"}
_LATENCY_RING = 512  # number of recent samples per route


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * p)))
    return sorted_values[k]


@dataclass
class _RouteStats:
    count: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_RING))
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        samples = sorted(self.latencies_ms)
        return {
            "count": self.count,
            "errors": self.errors,
            "avg_latency_ms": round(self.total_latency_ms / self.count, 2) if self.count else 0.0,
            "p50_latency_ms": round(_pct(samples, 0.50), 2),
            "p95_latency_ms": round(_pct(samples, 0.95), 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._routes: dict[str, _RouteStats] = defaultdict(_RouteStats)

    def record(
        self,
        *,
        path: str,
        status: int,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with self._lock:
            stats = self._routes[path]
            stats.count += 1
            stats.total_latency_ms += latency_ms
            stats.latencies_ms.append(latency_ms)
            if status >= 500:
                stats.errors += 1
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens

    def snapshot(self) -> dict:
        with self._lock:
            routes = {path: stats.to_dict() for path, stats in self._routes.items()}
            total_prompt = sum(s["prompt_tokens"] for s in routes.values())
            total_completion = sum(s["completion_tokens"] for s in routes.values())
            total_req = sum(s["count"] for s in routes.values())
            total_err = sum(s["errors"] for s in routes.values())
            uptime_s = time.time() - self._started_at
        return {
            "uptime_s": round(uptime_s, 1),
            "total": {
                "requests": total_req,
                "errors": total_err,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
            },
            "routes": routes,
        }


class MetricsMiddleware:
    """Raw ASGI middleware (works with streaming responses).

    Captures the outgoing body on token-producing endpoints and parses
    `usage` out of the JSON when it's a non-streaming response. Streaming
    responses contribute latency + count but not token accounting (the
    usage block is the last SSE chunk and parsing SSE here isn't worth
    the complexity).
    """

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = 500
        captured = bytearray()
        content_type = ""
        should_capture = path in _TOKEN_ENDPOINTS

        async def send_wrapper(message):
            nonlocal status_code, content_type
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                for k, v in message.get("headers", []):
                    if k.lower() == b"content-type":
                        content_type = v.decode("latin-1", errors="replace").lower()
                        break
            elif message["type"] == "http.response.body" and should_capture:
                body = message.get("body", b"")
                if body and "application/json" in content_type and len(captured) < 1_000_000:
                    captured.extend(body)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            pt = ct = 0
            if should_capture and captured and "application/json" in content_type:
                try:
                    d = json.loads(bytes(captured))
                    usage = d.get("usage") or {}
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                except Exception:
                    pass
            self.metrics.record(
                path=path,
                status=status_code,
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
            )
