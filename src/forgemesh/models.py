"""Local GGUF model cache, sourced from HuggingFace.

Download strategy
-----------------
By default, `pull()` streams the file directly from
`https://huggingface.co/<repo_id>/resolve/<revision>/<filename>` using
``httpx``. This bypasses ``huggingface_hub``'s Xet transport, which has
been observed to stall at ~12 KB/s in regions where a plain HTTP GET
saturates the link at ~100 MB/s (see internal/artifacts/benchmarks/
install-time.md §v0.0.2-r1 and tasks.yaml T-0013).

If the direct path fails (network, 4xx other than 404, etc.) the puller
falls back to ``huggingface_hub.hf_hub_download``, which is the
known-correct path for private/gated repos and the only one that knows
about Xet at all. ``FORGEMESH_NO_XET=1`` is exported into the fallback
process so the Hub client picks the legacy LFS path even when Xet is
on offer.

Callers can opt out of the direct path entirely by passing
``direct=False`` to ``pull()`` (CLI: ``forgemesh models pull --no-direct
…``). That's mostly an escape hatch for repos whose direct URL doesn't
work without Hub-side authentication.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


# Hugging Face's CDN saturates ~100 MB/s on a fast link; on a slow link
# we still want to wait long enough for a multi-GiB GGUF to download.
# 30 minutes is the upper bound; if a transfer is genuinely sustained
# at <2 MB/s we'd rather fail and retry than hang an installer.
_HF_RESOLVE_BASE = "https://huggingface.co"
_DEFAULT_CONNECT_TIMEOUT_S = 15.0
_DEFAULT_READ_TIMEOUT_S = 60.0
_DEFAULT_TOTAL_TIMEOUT_S = 1800.0
_CHUNK_SIZE = 1024 * 1024  # 1 MiB stream chunks
_PROGRESS_LOG_EVERY_S = 5.0


@dataclass
class ModelInfo:
    name: str
    path: Path
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)


def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{_HF_RESOLVE_BASE}/{repo_id}/resolve/{revision}/{filename}"


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _direct_download(
    url: str,
    dst: Path,
    *,
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT_S,
    read_timeout: float = _DEFAULT_READ_TIMEOUT_S,
    total_timeout: float = _DEFAULT_TOTAL_TIMEOUT_S,
) -> None:
    """Stream `url` to `dst` via httpx with `.part` atomic rename.

    Raises ``httpx.HTTPError`` on transport failure, or
    ``RuntimeError`` if the deadline elapses mid-stream.
    """
    part = dst.with_suffix(dst.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=read_timeout,
        pool=connect_timeout,
    )
    deadline = time.monotonic() + total_timeout

    log.info("downloading %s", url)
    with (
        httpx.Client(follow_redirects=True, timeout=timeout) as client,
        client.stream("GET", url) as r,
    ):
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0")) or None
        received = 0
        last_log = time.monotonic()
        start = last_log

        with part.open("wb") as f:
            for chunk in r.iter_bytes(_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError(
                        f"download exceeded {total_timeout:.0f}s deadline "
                        f"after {_format_bytes(received)}"
                    )
                if now - last_log >= _PROGRESS_LOG_EVERY_S:
                    elapsed = max(now - start, 1e-3)
                    rate = received / elapsed
                    if total:
                        pct = 100.0 * received / total
                        log.info(
                            "  %s / %s (%.1f%%) @ %s/s",
                            _format_bytes(received),
                            _format_bytes(total),
                            pct,
                            _format_bytes(int(rate)),
                        )
                    else:
                        log.info(
                            "  %s @ %s/s",
                            _format_bytes(received),
                            _format_bytes(int(rate)),
                        )
                    last_log = now

    part.replace(dst)


def _hub_download(repo_id: str, filename: str, dst: Path, revision: str = "main") -> Path:
    """Fallback: use huggingface_hub. Disable Xet to avoid the slow path
    that motivated this whole module.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    log.info("falling back to huggingface_hub (xet disabled) for %s/%s", repo_id, filename)
    resolved = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    src = Path(resolved)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return dst
    try:
        dst.symlink_to(src)
    except OSError:
        import shutil

        shutil.copy2(src, dst)
    return dst


class ModelCatalog:
    """Flat directory of GGUF files. One file == one model. Name == stem."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for p in sorted(self.model_dir.glob("*.gguf")):
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out.append(ModelInfo(name=p.stem, path=p, size_bytes=size))
        return out

    def resolve(self, name_or_path: str) -> Path:
        """Resolve a user-provided model identifier to a concrete file path.

        Accepts:
          - Absolute or relative path to a .gguf file
          - Bare name (stem) of a model already in the cache
          - Filename with .gguf extension inside the cache

        Symlinks are intentionally NOT followed: the cache keeps friendly
        names (e.g. Qwen3-0.6B-Q4_K_M.gguf) as symlinks into HF's blob
        store, and we want to preserve the friendly name for the model
        identifier we report back to clients. llama-server itself opens
        the target through the kernel, so the actual blob still gets
        loaded; only the name we show the user differs.
        """
        p = Path(name_or_path).expanduser()
        if p.is_file():
            return p.absolute()

        candidates = [
            self.model_dir / name_or_path,
            self.model_dir / f"{name_or_path}.gguf",
        ]
        for c in candidates:
            if c.is_file():
                return c.absolute()

        known = ", ".join(m.name for m in self.list()) or "(cache empty)"
        raise FileNotFoundError(
            f"model '{name_or_path}' not found. "
            f"Locally cached models: {known}. "
            f"Download with: forgemesh models pull <hf-repo> <filename>"
        )

    def pull(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        direct: bool = True,
    ) -> Path:
        """Download `filename` from HuggingFace repo `repo_id` into the cache.

        Args:
            repo_id: e.g. ``unsloth/Qwen3-8B-GGUF``.
            filename: file inside the repo, e.g. ``Qwen3-8B-Q4_K_M.gguf``.
            revision: git revision/branch/tag on the HF side. Default ``main``.
            direct: when True (default), stream directly from
                ``huggingface.co/.../resolve/<revision>/<filename>`` and
                only fall back to ``huggingface_hub`` on failure. When
                False, go straight to the Hub client (still with Xet
                disabled).

        Returns:
            The path inside ``self.model_dir`` under which the file is
            available. Always uses the user-supplied ``filename`` for the
            cached entry (the Hub fallback symlinks back into HF's blob
            store; the direct path writes the file in place).
        """
        dst = self.model_dir / filename
        if dst.exists() or dst.is_symlink():
            log.info("already present in cache: %s", dst)
            return dst

        if direct:
            url = _resolve_url(repo_id, filename, revision)
            try:
                _direct_download(url, dst)
                return dst
            except (httpx.HTTPError, RuntimeError) as e:
                log.warning("direct download failed (%s); falling back to huggingface_hub", e)

        return _hub_download(repo_id, filename, dst, revision=revision)

    def remove(self, name: str) -> Path:
        path = self.resolve(name)
        path.unlink(missing_ok=False)
        return path
