"""Streaming chat completions against ForgeMesh.

Same OpenAI client, with `stream=True`. ForgeMesh forwards the
upstream `llama-server` SSE stream verbatim, so the official client
parses it with no adapter.

Run:

    pip install openai
    python examples/streaming.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from openai import OpenAI


def _read_api_key() -> str:
    env = os.environ.get("FORGEMESH_API_KEY")
    if env:
        return env
    key_file = Path.home() / ".forgemesh" / "api-key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit("set FORGEMESH_API_KEY or run `forgemesh serve` first")


def main() -> None:
    base_url = os.environ.get("FORGEMESH_BASE_URL", "http://127.0.0.1:8080")
    model = os.environ.get("FORGEMESH_MODEL", "Qwen3-0.6B-Q4_K_M")

    client = OpenAI(base_url=f"{base_url}/v1", api_key=_read_api_key())

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Write a short haiku about local LLMs."},
        ],
        max_tokens=128,
        temperature=0.4,
        stream=True,
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            completion_tokens += 1
    end = time.perf_counter()
    sys.stdout.write("\n")

    elapsed = end - start
    if first_token_at is not None:
        ttft = first_token_at - start
        gen = end - first_token_at
        rate = completion_tokens / gen if gen > 0 else 0.0
        print()
        print(
            f"# ttft={ttft*1000:.0f}ms "
            f"total={elapsed*1000:.0f}ms "
            f"completion_chunks={completion_tokens} "
            f"chunks/s={rate:.1f}"
        )


if __name__ == "__main__":
    main()
