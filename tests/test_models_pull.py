"""Tests for the direct-HTTP pull path that bypasses huggingface_hub Xet.

These don't hit the real Hugging Face — they stand up a tiny in-process
HTTP server and point httpx at it. The fallback path is exercised by
forcing the direct call to fail and asserting we land in
huggingface_hub. (We monkeypatch the import to avoid a real network
call.)
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path
from typing import ClassVar

import pytest

from forgemesh import models as models_mod
from forgemesh.models import ModelCatalog


@pytest.fixture
def fake_hf_server(tmp_path: Path):
    """Spin up an in-process HTTP server that serves files from a temp tree
    under the same `<repo_id>/resolve/<revision>/<filename>` path layout
    Hugging Face uses, so we can rebase _HF_RESOLVE_BASE at it."""
    repo_root = tmp_path / "fake-hf"
    repo_root.mkdir()

    handler = http.server.SimpleHTTPRequestHandler

    class _Handler(handler):
        def log_message(self, *args, **kwargs):
            pass

        def translate_path(self, path):
            return str(repo_root / path.lstrip("/"))

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield repo_root, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _stage_fake_file(repo_root: Path, repo_id: str, filename: str, content: bytes) -> Path:
    p = repo_root / repo_id / "resolve" / "main" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_direct_download_writes_file_atomically(
    tmp_path: Path, fake_hf_server, monkeypatch: pytest.MonkeyPatch
):
    repo_root, base = fake_hf_server
    payload = b"x" * (3 * 1024 * 1024 + 17)  # ~3 MiB + change, exercises chunking
    _stage_fake_file(repo_root, "owner/repo", "model.gguf", payload)
    monkeypatch.setattr(models_mod, "_HF_RESOLVE_BASE", base)

    catalog = ModelCatalog(tmp_path / "cache")
    out = catalog.pull("owner/repo", "model.gguf")

    assert out == (tmp_path / "cache" / "model.gguf").resolve()
    assert out.read_bytes() == payload
    assert not (tmp_path / "cache" / "model.gguf.part").exists()


def test_pull_skips_when_already_cached(
    tmp_path: Path, fake_hf_server, monkeypatch: pytest.MonkeyPatch
):
    _, base = fake_hf_server
    monkeypatch.setattr(models_mod, "_HF_RESOLVE_BASE", base)
    cache = tmp_path / "cache"
    cache.mkdir()
    existing = cache / "model.gguf"
    existing.write_bytes(b"already-here")

    catalog = ModelCatalog(cache)
    out = catalog.pull("owner/repo", "model.gguf")

    assert out.read_bytes() == b"already-here"


def test_direct_404_falls_back_to_hub(
    tmp_path: Path, fake_hf_server, monkeypatch: pytest.MonkeyPatch
):
    """If the direct URL 404s, we should land in the hf_hub_download fallback."""
    _, base = fake_hf_server  # repo intentionally empty -> direct GET will 404
    monkeypatch.setattr(models_mod, "_HF_RESOLVE_BASE", base)

    fake_blob = tmp_path / "blob_store" / "model.gguf"
    fake_blob.parent.mkdir()
    fake_blob.write_bytes(b"fallback-payload")

    calls: dict = {}

    def fake_hf_hub_download(repo_id: str, filename: str, revision: str = "main"):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["revision"] = revision
        return str(fake_blob)

    fake_module = type("M", (), {"hf_hub_download": staticmethod(fake_hf_hub_download)})
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    catalog = ModelCatalog(tmp_path / "cache")
    out = catalog.pull("owner/repo", "model.gguf")

    assert calls == {"repo_id": "owner/repo", "filename": "model.gguf", "revision": "main"}
    assert out.exists()
    assert out.read_bytes() == b"fallback-payload"


def test_no_direct_skips_direct_path(
    tmp_path: Path, fake_hf_server, monkeypatch: pytest.MonkeyPatch
):
    """direct=False must not even attempt the direct URL."""
    repo_root, base = fake_hf_server
    _stage_fake_file(repo_root, "owner/repo", "model.gguf", b"direct-payload")
    monkeypatch.setattr(models_mod, "_HF_RESOLVE_BASE", base)

    fake_blob = tmp_path / "blob_store" / "model.gguf"
    fake_blob.parent.mkdir()
    fake_blob.write_bytes(b"hub-payload")

    def fake_hf_hub_download(repo_id: str, filename: str, revision: str = "main"):
        return str(fake_blob)

    fake_module = type("M", (), {"hf_hub_download": staticmethod(fake_hf_hub_download)})
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    catalog = ModelCatalog(tmp_path / "cache")
    out = catalog.pull("owner/repo", "model.gguf", direct=False)

    # Hub path is symlink-or-copy from the fake blob, so payload should match
    # the hub blob, NOT the (faster) direct one we staged.
    assert out.read_bytes() == b"hub-payload"


def test_fallback_disables_xet_via_env(
    tmp_path: Path, fake_hf_server, monkeypatch: pytest.MonkeyPatch
):
    """The fallback path must export HF_HUB_DISABLE_XET=1 so the Hub client
    doesn't pick the slow Xet path that motivated this whole module."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    _, base = fake_hf_server  # 404 on direct
    monkeypatch.setattr(models_mod, "_HF_RESOLVE_BASE", base)

    fake_blob = tmp_path / "blob_store" / "model.gguf"
    fake_blob.parent.mkdir()
    fake_blob.write_bytes(b"x")

    captured: dict = {}

    def fake_hf_hub_download(repo_id, filename, revision="main"):
        import os

        captured["xet"] = os.environ.get("HF_HUB_DISABLE_XET")
        return str(fake_blob)

    fake_module = type("M", (), {"hf_hub_download": staticmethod(fake_hf_hub_download)})
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    catalog = ModelCatalog(tmp_path / "cache")
    catalog.pull("owner/repo", "model.gguf")

    assert captured["xet"] == "1"


def test_direct_download_honours_total_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If total_timeout elapses mid-stream we should raise instead of hanging."""

    class _SlowResp:
        headers: ClassVar[dict[str, str]] = {"content-length": "1000000"}

        def raise_for_status(self):
            return None

        def iter_bytes(self, n):
            import time as _time

            for _ in range(100):
                yield b"x" * 1024
                _time.sleep(0.05)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _SlowResp()

    monkeypatch.setattr(models_mod.httpx, "Client", _FakeClient)

    dst = tmp_path / "out.gguf"
    with pytest.raises(RuntimeError, match="deadline"):
        models_mod._direct_download(
            "http://example/x", dst, connect_timeout=1.0, read_timeout=1.0, total_timeout=0.1
        )
    assert not dst.exists()
