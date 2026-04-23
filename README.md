# ForgeMesh

Self-hosted, OpenAI-compatible LLM inference for small teams.

Install one binary on each machine you want to serve from, point clients at the endpoint, and get a private `/v1/chat/completions` API backed by your own hardware. No cloud inference API, no per-token billing, no data leaving your network.

> **Status: pre-alpha.** `v0.0.2` runs a single model on a single machine — on one or multiple GPUs on that machine — behind an API-key-protected OpenAI-compatible endpoint. Multi-machine support is next on the roadmap — see below. Expect breaking changes.

## Why

If you're a small team with your own GPU or a rented box, your options today are:

- Pay per-token to OpenAI / Anthropic / Together / Fireworks — fine, but your data leaves your network and costs scale with usage.
- Run `llama.cpp` or `vllm` directly — works, but you're writing the auth, the model management, the process supervision, and the deployment glue yourself.
- Stand up a Kubernetes cluster — overkill for one or two boxes.

ForgeMesh is the thin layer in between: `curl | bash`, point it at a GGUF model, and you have a production-shaped inference endpoint with API-key auth and a model catalog. When you add a second machine to your LAN, it joins the same endpoint.

## Install

Requires Python 3.11+ and a working [`llama.cpp`](https://github.com/ggerganov/llama.cpp) build (`llama-server` on your `PATH`). GPU acceleration is whatever `llama.cpp` was compiled with — CUDA, Metal, CPU, etc.

```bash
curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/install.sh | bash
```

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

In another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $FORGEMESH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-Q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

`FORGEMESH_API_KEY` is generated on first run and printed once; it's stored at `~/.forgemesh/api-key`.

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

Endpoints supported in `v0.0.2`:

- `GET  /v1/models`
- `POST /v1/chat/completions` (streaming and non-streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET  /healthz`
- `GET  /metrics` — JSON: request counts, average/p50/p95 latency, total prompt/completion tokens (in-memory, resets on restart)

Anything the upstream `llama-server` accepts, we forward.

## Roadmap

`v0.0.2` is still deliberately narrow. Things we intend to add, roughly in order:

- Same-LAN multi-machine: serve one model sharded across GPUs on multiple machines over the LAN, one endpoint. This is the "mesh" in the name.
- Model-catalog improvements: auto-resume downloads, checksumming, per-model default prompt templates.
- Web dashboard: server health, token usage per API key, model catalog.
- Per-API-key rate limits and usage accounting.

## License

Dual-licensed under [Apache 2.0](./LICENSE-APACHE) or [MIT](./LICENSE-MIT) at your option.
