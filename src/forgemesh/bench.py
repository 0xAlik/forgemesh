"""Simple benchmark harness.

Hits an OpenAI-compatible /v1/chat/completions endpoint (ours or any
other), runs N sequential requests, and reports wall-clock and
server-reported tokens/sec. Non-streaming only for v0.0.1; a concurrent
/ streaming mode is a reasonable next thing to add.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class BenchRunResult:
    run: int
    ok: bool
    status: int
    wall_ms: float
    prompt_tokens: int
    completion_tokens: int
    wall_tps: float
    server_predicted_tps: float | None
    error: str | None = None
    snippet: str = ""


@dataclass
class BenchSummary:
    endpoint: str
    model: str
    runs: int
    ok_runs: int
    max_tokens: int
    prompt: str
    results: list[BenchRunResult]

    def _tps(self, field: str) -> list[float]:
        return [getattr(r, field) for r in self.results if r.ok and getattr(r, field)]

    def as_dict(self) -> dict:
        wall = self._tps("wall_tps")
        server = [r.server_predicted_tps for r in self.results if r.ok and r.server_predicted_tps]
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "runs": self.runs,
            "ok_runs": self.ok_runs,
            "max_tokens": self.max_tokens,
            "wall_tps": {
                "mean": round(statistics.mean(wall), 2) if wall else 0.0,
                "median": round(statistics.median(wall), 2) if wall else 0.0,
                "min": round(min(wall), 2) if wall else 0.0,
                "max": round(max(wall), 2) if wall else 0.0,
            },
            "server_predicted_tps": {
                "mean": round(statistics.mean(server), 2) if server else None,
                "median": round(statistics.median(server), 2) if server else None,
            },
            "per_run": [
                {
                    "run": r.run,
                    "ok": r.ok,
                    "status": r.status,
                    "wall_ms": round(r.wall_ms, 1),
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "wall_tps": round(r.wall_tps, 2),
                    "server_predicted_tps": (
                        round(r.server_predicted_tps, 2) if r.server_predicted_tps else None
                    ),
                    "error": r.error,
                }
                for r in self.results
            ],
        }


DEFAULT_PROMPT = (
    "Write a concise 250-word description of how distributed LLM inference "
    "across multiple GPUs works. Use prose paragraphs, not bullet lists."
)


def run_bench(
    *,
    endpoint: str,
    model: str,
    api_key: str | None = None,
    runs: int = 3,
    max_tokens: int = 256,
    prompt: str = DEFAULT_PROMPT,
    timeout_s: float = 600.0,
    warmup: bool = True,
) -> BenchSummary:
    """Run the benchmark synchronously. Caller is expected to display progress."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }

    with httpx.Client(timeout=timeout_s) as client:
        if warmup:
            with contextlib.suppress(httpx.RequestError):
                client.post(url, headers=headers, json={**body, "max_tokens": 8})

        results: list[BenchRunResult] = []
        for i in range(1, runs + 1):
            t0 = time.perf_counter()
            try:
                resp = client.post(url, headers=headers, json=body)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                status = resp.status_code
                if status != 200:
                    results.append(
                        BenchRunResult(
                            run=i, ok=False, status=status, wall_ms=wall_ms,
                            prompt_tokens=0, completion_tokens=0,
                            wall_tps=0.0, server_predicted_tps=None,
                            error=resp.text[:300],
                        )
                    )
                    continue
                data = resp.json()
            except httpx.RequestError as e:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                results.append(
                    BenchRunResult(
                        run=i, ok=False, status=0, wall_ms=wall_ms,
                        prompt_tokens=0, completion_tokens=0,
                        wall_tps=0.0, server_predicted_tps=None,
                        error=str(e),
                    )
                )
                continue

            usage = data.get("usage", {}) or {}
            timings = data.get("timings", {}) or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            snippet = ""
            with contextlib.suppress(Exception):
                snippet = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")[:120]
                    .replace("\n", " ")
                )
            results.append(
                BenchRunResult(
                    run=i,
                    ok=True,
                    status=status,
                    wall_ms=wall_ms,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    wall_tps=(ct * 1000.0 / wall_ms) if wall_ms and ct else 0.0,
                    server_predicted_tps=timings.get("predicted_per_second"),
                    snippet=snippet,
                )
            )

    return BenchSummary(
        endpoint=endpoint,
        model=model,
        runs=runs,
        ok_runs=sum(1 for r in results if r.ok),
        max_tokens=max_tokens,
        prompt=prompt,
        results=results,
    )
