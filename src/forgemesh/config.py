"""Configuration model.

Resolved in this precedence, highest wins:
  1. CLI flags
  2. Environment variables (FORGEMESH_*)
  3. Config file (YAML)
  4. Defaults defined here
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_HOME = Path.home() / ".forgemesh"


class AuthConfig(BaseModel):
    enabled: bool = True
    api_key_file: Path = DEFAULT_HOME / "api-key"


class EngineConfig(BaseModel):
    """Options forwarded to the llama.cpp llama-server subprocess."""

    context_size: int = 4096
    gpu_layers: int = -1
    threads: int | None = None
    extra_args: list[str] = Field(default_factory=list)


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
