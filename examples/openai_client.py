"""Drop-in OpenAI Python client against ForgeMesh.

ForgeMesh is OpenAI-API-compatible: point the official `openai` client
at your ForgeMesh endpoint, hand it the API key from
`~/.forgemesh/api-key`, and your existing code works unchanged.

Run:

    pip install openai
    python examples/openai_client.py

Override the endpoint or model via env vars if your ForgeMesh box isn't
on localhost:

    FORGEMESH_BASE_URL=http://10.0.0.5:8080 \\
    FORGEMESH_MODEL=Qwen3-8B-Q4_K_M \\
    python examples/openai_client.py
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


def _read_api_key() -> str:
    env = os.environ.get("FORGEMESH_API_KEY")
    if env:
        return env
    key_file = Path.home() / ".forgemesh" / "api-key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "Could not find an API key. Set FORGEMESH_API_KEY or run "
        "`forgemesh serve` once to generate ~/.forgemesh/api-key."
    )


def main() -> None:
    base_url = os.environ.get("FORGEMESH_BASE_URL", "http://127.0.0.1:8080")
    model = os.environ.get("FORGEMESH_MODEL", "Qwen3-0.6B-Q4_K_M")

    client = OpenAI(base_url=f"{base_url}/v1", api_key=_read_api_key())

    print(f"# {base_url}  model={model}")
    print()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Name three benefits of running inference on your own hardware."},
        ],
        max_tokens=128,
        temperature=0.2,
    )
    print(resp.choices[0].message.content)
    if resp.usage:
        print()
        print(
            f"# usage: prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )


if __name__ == "__main__":
    main()
