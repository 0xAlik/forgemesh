"""Smoke tests that don't require llama-server or a GPU.

These exercise: package import, config parsing, auth dependency behaviour,
and model-catalog listing/resolution over a temp directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forgemesh import __version__
from forgemesh.auth import ensure_api_key, make_auth_dependency
from forgemesh.config import Config
from forgemesh.models import ModelCatalog


def test_version_string():
    assert __version__ == "0.0.2"


def test_config_defaults():
    cfg = Config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8080
    assert cfg.auth.enabled is True
    assert cfg.engine.gpu_layers == -1


def test_config_from_yaml(tmp_path: Path):
    p = tmp_path / "forgemesh.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "host": "0.0.0.0",
                "port": 9999,
                "auth": {"enabled": False},
                "engine": {"context_size": 8192, "gpu_layers": 20, "extra_args": ["--flash-attn"]},
            }
        )
    )
    cfg = Config.load(p)
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9999
    assert cfg.auth.enabled is False
    assert cfg.engine.context_size == 8192
    assert cfg.engine.gpu_layers == 20
    assert cfg.engine.extra_args == ["--flash-attn"]


def test_ensure_api_key_generates_once(tmp_path: Path):
    f = tmp_path / "api-key"
    k1, new1 = ensure_api_key(f)
    k2, new2 = ensure_api_key(f)
    assert new1 is True
    assert new2 is False
    assert k1 == k2
    assert len(k1) >= 32


def _build_authed_app(api_key: str, *, enabled: bool = True) -> FastAPI:
    app = FastAPI()
    dep = make_auth_dependency(api_key, enabled=enabled)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    from fastapi import Depends

    @app.get("/v1/models", dependencies=[Depends(dep)])
    async def models():
        return {"data": []}

    return app


def test_auth_rejects_without_bearer():
    app = _build_authed_app("secret")
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        r = c.get("/v1/models")
        assert r.status_code == 401


def test_auth_rejects_wrong_key():
    app = _build_authed_app("secret")
    with TestClient(app) as c:
        r = c.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401


def test_auth_accepts_right_key():
    app = _build_authed_app("secret")
    with TestClient(app) as c:
        r = c.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200


def test_auth_disabled_lets_everything_through():
    app = _build_authed_app("secret", enabled=False)
    with TestClient(app) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200


def test_model_catalog_list_empty(tmp_path: Path):
    cat = ModelCatalog(tmp_path)
    assert cat.list() == []


def test_model_catalog_list_and_resolve(tmp_path: Path):
    fake = tmp_path / "fake-model.gguf"
    fake.write_bytes(b"not a real gguf, but pretend" * 10)
    cat = ModelCatalog(tmp_path)
    items = cat.list()
    assert len(items) == 1
    assert items[0].name == "fake-model"
    assert cat.resolve("fake-model") == fake.resolve()
    assert cat.resolve("fake-model.gguf") == fake.resolve()
    assert cat.resolve(str(fake)) == fake.resolve()


def test_model_catalog_resolve_missing(tmp_path: Path):
    cat = ModelCatalog(tmp_path)
    with pytest.raises(FileNotFoundError):
        cat.resolve("does-not-exist")
