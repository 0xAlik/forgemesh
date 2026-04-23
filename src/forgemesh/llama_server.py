"""Manage a single `llama-server` subprocess.

Launch, wait-for-ready, graceful shutdown. We shell out to the upstream
`llama.cpp` binary rather than linking bindings. One less compile-time
dependency for users; also matches how the reference CLI tools work.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from forgemesh.config import Config

log = logging.getLogger(__name__)


class LlamaServerError(RuntimeError):
    pass


def _fmt_num(x: float) -> str:
    """Render a number without a trailing .0 for integer-valued floats.

    Keeps `--tensor-split 3,2` clean instead of `3.0,2.0`, which llama-server
    accepts but looks noisy in logs.
    """
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


@dataclass
class LlamaServer:
    config: Config
    model_path: Path
    _proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.llama_server_host}:{self.config.llama_server_port}"

    def _build_argv(self) -> list[str]:
        cfg = self.config
        eng = cfg.engine
        argv = [
            cfg.llama_server_path,
            "--model", str(self.model_path),
            "--host", cfg.llama_server_host,
            "--port", str(cfg.llama_server_port),
            "--ctx-size", str(eng.context_size),
            "--n-gpu-layers", str(eng.gpu_layers),
        ]
        if eng.threads is not None:
            argv.extend(["--threads", str(eng.threads)])
        if eng.split_mode is not None:
            argv.extend(["--split-mode", eng.split_mode])
        if eng.tensor_split is not None:
            # llama-server expects a comma-separated list, e.g. "3,2".
            argv.extend(["--tensor-split", ",".join(_fmt_num(x) for x in eng.tensor_split)])
        if eng.main_gpu is not None:
            argv.extend(["--main-gpu", str(eng.main_gpu)])
        argv.extend(eng.extra_args)
        return argv

    def start(self, *, ready_timeout_s: float = 120.0) -> None:
        if self._proc is not None:
            raise LlamaServerError("already started")
        if not self.model_path.exists():
            raise LlamaServerError(f"model file not found: {self.model_path}")

        argv = self._build_argv()
        log.info("launching llama-server: %s", " ".join(argv))
        try:
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise LlamaServerError(
                f"llama-server not found at '{self.config.llama_server_path}'. "
                "Install llama.cpp and put `llama-server` on your PATH, or set "
                "`llama_server_path` in forgemesh.yaml."
            ) from e

        self._wait_for_ready(ready_timeout_s)

    def _wait_for_ready(self, timeout_s: float) -> None:
        assert self._proc is not None
        deadline = time.time() + timeout_s
        health = f"{self.base_url}/health"
        while time.time() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                out = self._drain_output_nonblocking()
                raise LlamaServerError(
                    f"llama-server exited with rc={rc} before becoming ready. "
                    f"last output:\n{out}"
                )
            try:
                r = httpx.get(health, timeout=1.0)
                if r.status_code == 200:
                    log.info("llama-server is ready at %s", self.base_url)
                    return
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        raise LlamaServerError(f"llama-server did not become ready within {timeout_s}s")

    def _drain_output_nonblocking(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            return ""
        try:
            data = self._proc.stdout.read()
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)

    def stop(self, *, timeout_s: float = 10.0) -> None:
        if self._proc is None:
            return
        pgid = None
        with contextlib.suppress(ProcessLookupError):
            pgid = os.getpgid(self._proc.pid)

        with contextlib.suppress(ProcessLookupError):
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self._proc.terminate()

        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("llama-server did not exit on SIGTERM; sending SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self._proc.kill()
            self._proc.wait(timeout=5.0)
        finally:
            self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
