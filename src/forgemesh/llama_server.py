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
    _log_fh = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.llama_server_host}:{self.config.llama_server_port}"

    @property
    def log_path(self) -> Path:
        """Where llama-server's stdout/stderr is appended.

        Lives next to the API-key file under FORGEMESH_HOME so the
        location is predictable for users grepping "why is generation
        running on CPU?". Backend identity (CUDA / Vulkan / Metal /
        CPU), per-layer offload, and llama.cpp's startup banner all
        end up here.
        """
        return self.config.auth.api_key_file.parent / "llama-server.log"

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

        log_path = self.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append-binary so llama-server's chunked output never blocks
        # on a PIPE the parent isn't draining. line buffering lets `tail -f`
        # behave nicely.
        self._log_fh = log_path.open("ab", buffering=0)
        log.info("llama-server stdout/stderr -> %s", log_path)

        try:
            self._proc = subprocess.Popen(
                argv,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            self._close_log()
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
                tail = self._tail_log(40)
                raise LlamaServerError(
                    f"llama-server exited with rc={rc} before becoming ready. "
                    f"see {self.log_path} (last lines):\n{tail}"
                )
            try:
                r = httpx.get(health, timeout=1.0)
                if r.status_code == 200:
                    log.info("llama-server is ready at %s", self.base_url)
                    return
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        raise LlamaServerError(
            f"llama-server did not become ready within {timeout_s}s. "
            f"see {self.log_path} (last lines):\n{self._tail_log(40)}"
        )

    def _tail_log(self, n: int) -> str:
        try:
            with self.log_path.open("rb") as f:
                data = f.read()
        except OSError:
            return ""
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-n:])

    def _close_log(self) -> None:
        fh = self._log_fh
        self._log_fh = None
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()

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
            self._close_log()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
