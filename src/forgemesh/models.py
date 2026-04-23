"""Local GGUF model cache, sourced from HuggingFace."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    path: Path
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)


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

    def pull(self, repo_id: str, filename: str) -> Path:
        """Download `filename` from HuggingFace repo `repo_id` into the cache.

        hf_hub_download may return the blob path under HF's cache (whose
        filename is a content-hash, not the logical `filename` the user
        asked for). We always expose the file in our cache under the
        original requested `filename`, via a symlink when possible.
        """
        log.info("downloading %s/%s", repo_id, filename)
        resolved = hf_hub_download(repo_id=repo_id, filename=filename)
        src = Path(resolved)
        dst = self.model_dir / filename
        if dst.exists() or dst.is_symlink():
            log.info("already present in cache: %s", dst)
            return dst
        try:
            dst.symlink_to(src)
        except OSError:
            import shutil

            shutil.copy2(src, dst)
        return dst

    def remove(self, name: str) -> Path:
        path = self.resolve(name)
        path.unlink(missing_ok=False)
        return path
