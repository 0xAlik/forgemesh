"""Tests for multi-GPU config + llama-server argv construction.

These don't start a llama-server; we just inspect the argv we'd launch it with.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from forgemesh.config import Config, EngineConfig
from forgemesh.llama_server import LlamaServer, _fmt_num


def _server_for(cfg: Config, tmp_path: Path) -> LlamaServer:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    return LlamaServer(config=cfg, model_path=model)


def test_default_argv_has_no_multigpu_flags(tmp_path: Path):
    srv = _server_for(Config(), tmp_path)
    argv = srv._build_argv()
    assert "--split-mode" not in argv
    assert "--tensor-split" not in argv
    assert "--main-gpu" not in argv


def test_split_mode_layer(tmp_path: Path):
    cfg = Config(engine=EngineConfig(split_mode="layer"))
    argv = _server_for(cfg, tmp_path)._build_argv()
    i = argv.index("--split-mode")
    assert argv[i + 1] == "layer"


def test_tensor_split_serialised_as_csv(tmp_path: Path):
    cfg = Config(engine=EngineConfig(tensor_split=[3, 2]))
    argv = _server_for(cfg, tmp_path)._build_argv()
    i = argv.index("--tensor-split")
    assert argv[i + 1] == "3,2"


def test_tensor_split_fractional(tmp_path: Path):
    cfg = Config(engine=EngineConfig(tensor_split=[0.6, 0.4]))
    argv = _server_for(cfg, tmp_path)._build_argv()
    i = argv.index("--tensor-split")
    assert argv[i + 1] == "0.6,0.4"


def test_main_gpu(tmp_path: Path):
    cfg = Config(engine=EngineConfig(main_gpu=1))
    argv = _server_for(cfg, tmp_path)._build_argv()
    i = argv.index("--main-gpu")
    assert argv[i + 1] == "1"


def test_combined_multigpu_args(tmp_path: Path):
    cfg = Config(
        engine=EngineConfig(
            split_mode="row",
            tensor_split=[1, 1],
            main_gpu=0,
            gpu_layers=-1,
        )
    )
    argv = _server_for(cfg, tmp_path)._build_argv()
    # All three flags must be present, order within each pair preserved.
    assert argv[argv.index("--split-mode") + 1] == "row"
    assert argv[argv.index("--tensor-split") + 1] == "1,1"
    assert argv[argv.index("--main-gpu") + 1] == "0"


def test_extra_args_still_appended_after_multigpu(tmp_path: Path):
    cfg = Config(
        engine=EngineConfig(
            split_mode="layer",
            tensor_split=[1, 1],
            extra_args=["--flash-attn"],
        )
    )
    argv = _server_for(cfg, tmp_path)._build_argv()
    assert argv[-1] == "--flash-attn"
    # Sanity: extra_args comes after our structured flags.
    assert argv.index("--flash-attn") > argv.index("--split-mode")
    assert argv.index("--flash-attn") > argv.index("--tensor-split")


def test_invalid_split_mode_rejected():
    with pytest.raises(ValidationError):
        EngineConfig(split_mode="wrong")  # type: ignore[arg-type]


def test_tensor_split_all_zeros_rejected():
    with pytest.raises(ValidationError):
        EngineConfig(tensor_split=[0, 0])


def test_tensor_split_negative_rejected():
    with pytest.raises(ValidationError):
        EngineConfig(tensor_split=[3, -1])


def test_tensor_split_empty_rejected():
    with pytest.raises(ValidationError):
        EngineConfig(tensor_split=[])


def test_main_gpu_negative_rejected():
    with pytest.raises(ValidationError):
        EngineConfig(main_gpu=-1)


def test_fmt_num_strips_trailing_zero():
    assert _fmt_num(3) == "3"
    assert _fmt_num(3.0) == "3"
    assert _fmt_num(0.6) == "0.6"


def test_yaml_multigpu_round_trip(tmp_path: Path):
    p = tmp_path / "forgemesh.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "engine": {
                    "split_mode": "layer",
                    "tensor_split": [3, 2],
                    "main_gpu": 0,
                }
            }
        )
    )
    cfg = Config.load(p)
    assert cfg.engine.split_mode == "layer"
    assert cfg.engine.tensor_split == [3, 2]
    assert cfg.engine.main_gpu == 0
