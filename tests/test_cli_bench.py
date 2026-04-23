"""Tests for `forgemesh bench --json`: stdout must be pure, parseable JSON.

We spin up a tiny fake OpenAI-compatible endpoint on a real TCP port
(localhost, ephemeral) in a background thread, then invoke the Typer CLI
and assert stdout is exactly one JSON object with the expected keys.

Regression coverage for the "rich-colored JSON breaks jq" bug we hit during
the first multi-GPU validation run.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from typer.testing import CliRunner

from forgemesh.cli import app


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # quiet
        return

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(n)  # discard body
        body = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "timings": {"predicted_per_second": 42.0},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_fake_server() -> tuple[str, HTTPServer, threading.Thread]:
    srv = HTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}", srv, t


def test_bench_json_stdout_is_parseable():
    endpoint, srv, _ = _start_fake_server()
    try:
        # click 8.3+ separates stdout/stderr by default.
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bench",
                "--endpoint", endpoint,
                "--model", "fake",
                "--api-key", "irrelevant",
                "--runs", "2",
                "--max-tokens", "8",
                "--no-warmup",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stderr
        # Stdout must be EXACTLY one JSON doc, nothing else.
        data = json.loads(result.stdout)
        assert data["runs"] == 2
        assert data["ok_runs"] == 2
        assert data["wall_tps"]["mean"] is not None
        # Pretty header + per-run lines went to stderr, not stdout.
        assert "Benchmarking" in result.stderr
        # And no rich ANSI escapes leaked into stdout.
        assert "\x1b[" not in result.stdout
    finally:
        srv.shutdown()


def test_bench_plain_output_has_table():
    endpoint, srv, _ = _start_fake_server()
    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bench",
                "--endpoint", endpoint,
                "--model", "fake",
                "--api-key", "irrelevant",
                "--runs", "2",
                "--max-tokens", "8",
                "--no-warmup",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Summary" in result.output
        assert "ok_runs" in result.output
    finally:
        srv.shutdown()
