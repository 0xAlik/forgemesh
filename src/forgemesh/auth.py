"""API-key authentication.

Single shared secret, read from a file. Generated on first run if absent.
Sent by clients as `Authorization: Bearer <key>`.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import HTTPException, Request, status


def ensure_api_key(path: Path) -> tuple[str, bool]:
    """Return (key, was_newly_generated). Creates parent dir and file if needed."""
    path = Path(path).expanduser()
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key, False
    key = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    path.chmod(0o600)
    return key, True


def make_auth_dependency(api_key: str, *, enabled: bool):
    """Return a FastAPI dependency that validates Bearer tokens."""

    async def require_auth(request: Request) -> None:
        if not enabled:
            return
        if request.url.path in {"/healthz", "/", "/metrics"}:
            return
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        presented = header.split(" ", 1)[1].strip()
        if not secrets.compare_digest(presented, api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid api key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_auth
