"""Configuration model.

Resolved in this precedence, highest wins:
  1. CLI flags
  2. Environment variables (FORGEMESH_*)
  3. Config file (YAML)
  4. Defaults defined here
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_HOME = Path.home() / ".forgemesh"

SplitMode = Literal["layer", "row", "none"]


class AuthConfig(BaseModel):
    enabled: bool = True
    api_key_file: Path = DEFAULT_HOME / "api-key"


class EngineConfig(BaseModel):
    """Options forwarded to the llama.cpp llama-server subprocess."""

    context_size: int = 4096
    gpu_layers: int = -1
    threads: int | None = None

    # --- Multi-GPU (same host). These map to llama-server flags. ---
    # How to split a model across multiple GPUs.
    #   "layer" (llama.cpp default): each GPU owns a contiguous block of layers.
    #   "row":  tensor-parallel; faster with fast interconnect, slower on PCIe.
    #   "none": single-GPU (use main_gpu only).
    split_mode: SplitMode | None = None

    # Proportional split across GPUs, e.g. [3, 2] -> ~60% on GPU 0, ~40% on GPU 1.
    # None means "equal split" (llama.cpp default for layer/row modes).
    tensor_split: list[float] | None = None

    # Index of the GPU used for small ops (scratch buffers, KV cache for some layouts).
    # None means llama.cpp picks (typically 0).
    main_gpu: int | None = None

    # Escape hatch for any llama-server flag we don't model explicitly.
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("tensor_split")
    @classmethod
    def _validate_tensor_split(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) < 1:
            raise ValueError("tensor_split must have at least one entry")
        if any(x < 0 for x in v):
            raise ValueError("tensor_split entries must be non-negative")
        if all(x == 0 for x in v):
            raise ValueError("tensor_split cannot be all zeros")
        return v

    @field_validator("main_gpu")
    @classmethod
    def _validate_main_gpu(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("main_gpu must be >= 0")
        return v


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080

    model_dir: Path = DEFAULT_HOME / "models"
    llama_server_path: str = "llama-server"
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int = 9337

    model: str | None = None

    auth: AuthConfig = Field(default_factory=AuthConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load from a YAML file if given; otherwise return defaults."""
        if path is None:
            return cls()
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def resolve_paths(self) -> Config:
        """Expand ~ and make relative paths absolute."""
        self.model_dir = Path(self.model_dir).expanduser().resolve()
        self.auth.api_key_file = Path(self.auth.api_key_file).expanduser().resolve()
        return self
