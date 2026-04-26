# ForgeMesh

Self-hosted, OpenAI-compatible LLM inference for small teams.

Install one binary on each machine you want to serve from, point clients at the endpoint, and get a private `/v1/chat/completions` API backed by your own hardware. No cloud inference API, no per-token billing, no data leaving your network.

> **Status: pre-alpha.** `v0.0.3` runs a single model on a single machine — on one or multiple GPUs on that machine — behind an API-key-protected OpenAI-compatible endpoint. Multi-machine support is next on the roadmap — see below. Expect breaking changes.

## Why

If you're a small team with your own GPU or a rented box, your options today are:

- Pay per-token to OpenAI / Anthropic / Together / Fireworks — fine, but your data leaves your network and costs scale with usage.
- Run `llama.cpp` or `vllm` directly — works, but you're writing the auth, the model management, the process supervision, and the deployment glue yourself.
- Stand up a Kubernetes cluster — overkill for one or two boxes.

ForgeMesh is the thin layer in between: `curl | bash`, point it at a GGUF model, and you have a production-shaped inference endpoint with API-key auth and a model catalog. When you add a second machine to your LAN, it joins the same endpoint.

## Install

Requires Python 3.11+, `git`, `curl`, and `tar`. The installer auto-installs a prebuilt [`llama.cpp`](https://github.com/ggml-org/llama.cpp) binary if one isn't already on your `PATH` — Vulkan on Linux x86_64, Metal on Apple Silicon, CPU-only on Linux ARM64.

```bash
curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/install.sh | bash
```

The installer creates an isolated venv at `~/.forgemesh/venv`, installs ForgeMesh from the tagged GitHub release (ForgeMesh is pre-alpha and deliberately not on PyPI yet), and drops a `forgemesh` shim into `~/.local/bin`. If `llama-server` isn't already on your `PATH`, it also downloads the matching prebuilt `llama.cpp` release into `~/.forgemesh/llama.cpp` and shims `llama-server` next to `forgemesh`. Total install time on a clean box is typically well under a minute (model download not included).

Environment overrides:

- `FORGEMESH_VERSION=0.0.3` — pin a different ForgeMesh release tag.
- `FORGEMESH_REF=main` — track a branch for hacking (unpinned; not recommended for pilot use).
- `LLAMA_CPP_VERSION=bXXXX` — pin a specific upstream `llama.cpp` release; default is `latest` resolved via the GitHub API with a fallback to a known-good tag if the API is unreachable.
- `FORGEMESH_SKIP_LLAMA=1` — skip the `llama.cpp` auto-install (bring your own binary). Useful when you want a hand-tuned CUDA build for sustained NVIDIA workloads — build `llama-server` from source as documented upstream, drop it on `PATH`, and the installer leaves it alone.

Or from source:

```bash
git clone https://github.com/0xAlik/forgemesh
cd forgemesh
pip install -e .
```

## Quick start

```bash
forgemesh models pull unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf
forgemesh serve --model Qwen3-8B-Q4_K_M.gguf
```

`models pull` streams the file directly from `huggingface.co/<repo>/resolve/main/<filename>` (bypasses the `huggingface_hub` Xet transport, which has stalled at ~12 KB/s in some regions). Use `--no-direct` to force the Hub client (e.g. for private/gated repos that need auth), or `--revision` to fetch from a non-default branch/tag.

In another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $FORGEMESH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-Q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

`FORGEMESH_API_KEY` is generated on first run and printed once; it's stored at `~/.forgemesh/api-key`.

## Got a GPU? Smoke-test the whole thing in one shot

If you just want to verify ForgeMesh works on your hardware end-to-end (install → pull a small model → serve → chat → tiny benchmark), run:

```bash
curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/scripts/smoke-test.sh | bash
```

Defaults are sized to be friendly to any machine: `unsloth/Qwen3-0.6B-GGUF` (378 MiB) so the download is seconds, not minutes. The script prints per-phase timings (install, pull, start, first chat, bench) and writes both a JSON and a Markdown summary under `~/.forgemesh/smoke-test/`. To exercise an 8B model on a real GPU instead:

```bash
FORGEMESH_TEST_MODEL_REPO=unsloth/Qwen3-8B-GGUF \
FORGEMESH_TEST_MODEL_FILE=Qwen3-8B-Q4_K_M.gguf \
  bash <(curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/scripts/smoke-test.sh)
```

If you're test-driving this for me, please send back the contents of `~/.forgemesh/smoke-test/summary.json` (and `summary.md` if you want it pretty).

## Configuration

Minimal `forgemesh.yaml`:

```yaml
host: 0.0.0.0
port: 8080
model_dir: ~/.forgemesh/models
llama_server_path: llama-server
auth:
  enabled: true
engine:
  context_size: 4096
  gpu_layers: -1
```

Run with `forgemesh serve --config forgemesh.yaml`.

### Multiple GPUs on one machine

If the box has more than one GPU, ForgeMesh can split a single model across them:

```yaml
engine:
  gpu_layers: -1
  split_mode: layer      # 'layer' (default), 'row', or 'none'
  tensor_split: [3, 2]   # ~60% of layers on GPU 0, ~40% on GPU 1
  main_gpu: 0
```

Or inline:

```bash
forgemesh serve --model Qwen3-8B-Q4_K_M --tensor-split 3,2 --split-mode layer
```

`layer` is the right default on PCIe-only systems. `row` is tensor-parallel and only faster when you have a fast interconnect like NVLink.

## Commands

| Command | What it does |
|---|---|
| `forgemesh serve` | Start the API server |
| `forgemesh bench -m <model>` | Benchmark any OpenAI-compatible endpoint |
| `forgemesh models list` | List locally cached models |
| `forgemesh models pull <hf-repo> <filename>` | Download a GGUF from HuggingFace into the cache |
| `forgemesh models rm <name>` | Remove a cached model |
| `forgemesh version` | Print version |

## OpenAI API compatibility

Endpoints supported in `v0.0.3`:

- `GET  /v1/models`
- `POST /v1/chat/completions` (streaming and non-streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET  /healthz`
- `GET  /metrics` — JSON: request counts, average/p50/p95 latency, total prompt/completion tokens (in-memory, resets on restart)

Anything the upstream `llama-server` accepts, we forward.

## Roadmap

`v0.0.3` is still deliberately narrow. Things we intend to add, roughly in order:

- Same-LAN multi-machine: serve one model sharded across GPUs on multiple machines over the LAN, one endpoint. This is the "mesh" in the name.
- Model-catalog improvements: auto-resume downloads, checksumming, per-model default prompt templates.
- Web dashboard: server health, token usage per API key, model catalog.
- Per-API-key rate limits and usage accounting.

## License

Dual-licensed under [Apache 2.0](./LICENSE-APACHE) or [MIT](./LICENSE-MIT) at your option.
