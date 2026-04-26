#!/usr/bin/env bash
# ForgeMesh end-to-end smoke test.
#
# Runs the published install.sh, pulls a small GGUF, starts `forgemesh
# serve` in the background, waits for /healthz, hits /v1/chat/completions,
# runs a tiny `forgemesh bench`, then shuts everything down. Times each
# phase and prints a single PASS/FAIL summary at the end.
#
# Doubles as the canonical T-0009 install-time measurement script —
# the per-phase timings printed at the end are the data we care about.
#
# Defaults are sized for "any GPU box, any region": Qwen3-0.6B-Q4_K_M
# is 378 MiB, fits in any GPU, downloads in seconds on a fast link.
# Override via env vars below for the "real" 8B benchmark.
#
# Usage on a clean box:
#
#   curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/scripts/smoke-test.sh | bash
#
# With a specific model:
#
#   FORGEMESH_TEST_MODEL_REPO=unsloth/Qwen3-8B-GGUF \
#   FORGEMESH_TEST_MODEL_FILE=Qwen3-8B-Q4_K_M.gguf \
#     bash <(curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/scripts/smoke-test.sh)
#
# Requires: bash, python3.11+, git, curl, tar. The install.sh leg will
# install everything else (forgemesh + a prebuilt llama.cpp).
#
# This script does not touch anything outside $FORGEMESH_HOME (default
# ~/.forgemesh) and $INSTALL_BIN (default ~/.local/bin) — same surface
# install.sh manages.

set -euo pipefail

# Note: this script runs against the canonical install footprint
# (~/.forgemesh + ~/.local/bin). install.sh accepts FORGEMESH_HOME and
# INSTALL_BIN overrides for isolation, but the runtime `forgemesh`
# binary always reads/writes ~/.forgemesh/{models,api-key} from
# Config defaults, so trying to relocate one without the other just
# causes mismatches. Keep them locked to the defaults here; if you
# need true isolation use a container or a fresh user account.
FORGEMESH_HOME="$HOME/.forgemesh"
INSTALL_BIN="$HOME/.local/bin"
FORGEMESH_REPO_RAW="${FORGEMESH_REPO_RAW:-https://raw.githubusercontent.com/0xAlik/forgemesh/main}"

TEST_MODEL_REPO="${FORGEMESH_TEST_MODEL_REPO:-unsloth/Qwen3-0.6B-GGUF}"
TEST_MODEL_FILE="${FORGEMESH_TEST_MODEL_FILE:-Qwen3-0.6B-Q4_K_M.gguf}"
TEST_PORT="${FORGEMESH_TEST_PORT:-8090}"
TEST_HOST="${FORGEMESH_TEST_HOST:-127.0.0.1}"

# Where we drop logs and the JSON summary.
RESULTS_DIR="${FORGEMESH_TEST_RESULTS_DIR:-$FORGEMESH_HOME/smoke-test}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/serve.log"
SUMMARY_JSON="$RESULTS_DIR/summary.json"
SUMMARY_MD="$RESULTS_DIR/summary.md"
# llama-server.log is forgemesh's canonical location for the upstream
# llama.cpp banner (backend, layer offload, etc.). We grep this to
# detect CPU-only fallback on a machine that has a GPU.
LLAMA_SERVER_LOG="$FORGEMESH_HOME/llama-server.log"

export PATH="$INSTALL_BIN:$PATH"

# Phase timing helpers -------------------------------------------------------
# Per-phase millis live in scalar variables (PHASE_MS_<name>) so we don't
# need bash 4 associative arrays — macOS ships bash 3.2.
phase_start_ts=0
current_phase=""

ts_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

phase_ms_var() { echo "PHASE_MS_$1"; }

start_phase() {
  current_phase="$1"
  phase_start_ts=$(ts_ms)
  echo
  echo "==> [phase] $current_phase"
}

end_phase() {
  local end_ts elapsed var
  end_ts=$(ts_ms)
  elapsed=$(( end_ts - phase_start_ts ))
  var="$(phase_ms_var "$current_phase")"
  eval "$var=$elapsed"
  printf '     %s: %.2fs\n' "$current_phase" "$(echo "$elapsed/1000" | bc -l)"
}

# Cleanup --------------------------------------------------------------------
SERVE_PID=""
cleanup() {
  if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# 0. Box info ----------------------------------------------------------------
echo "==> ForgeMesh smoke test"
echo "    host:          $(uname -a)"
echo "    python3:       $(python3 -V 2>&1)"
echo "    model:         ${TEST_MODEL_REPO}/${TEST_MODEL_FILE}"
echo "    bind:          ${TEST_HOST}:${TEST_PORT}"
echo "    forgemesh_home:$FORGEMESH_HOME"
echo "    install_bin:   $INSTALL_BIN"
echo "    results_dir:   $RESULTS_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    gpu (nvidia):"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/      /'
elif command -v rocm-smi >/dev/null 2>&1; then
  echo "    gpu (rocm):    rocm-smi present"
else
  case "$(uname -s):$(uname -m)" in
    Darwin:arm64) echo "    gpu:           Apple Silicon (Metal)" ;;
    *)            echo "    gpu:           none detected (CPU inference)" ;;
  esac
fi

T_TOTAL_START=$(ts_ms)

# 1. Install -----------------------------------------------------------------
start_phase install
if command -v forgemesh >/dev/null 2>&1 && command -v llama-server >/dev/null 2>&1; then
  echo "     forgemesh + llama-server already on PATH; skipping install.sh"
else
  curl -fsSL "${FORGEMESH_REPO_RAW}/install.sh" | bash
fi
hash -r
forgemesh version
end_phase

# 2. Pull model --------------------------------------------------------------
start_phase pull_model
forgemesh models pull "$TEST_MODEL_REPO" "$TEST_MODEL_FILE"
end_phase

# 3. Start serve in the background ------------------------------------------
start_phase start_serve
forgemesh serve \
  --model "$TEST_MODEL_FILE" \
  --host "$TEST_HOST" \
  --port "$TEST_PORT" \
  --no-auth \
  > "$LOG" 2>&1 &
SERVE_PID=$!

# 4. Wait for /healthz to return 200 ----------------------------------------
healthz_url="http://${TEST_HOST}:${TEST_PORT}/healthz"
deadline=$(( $(ts_ms) + 120000 ))   # 120s wall budget for cold model load
while true; do
  if [ "$(ts_ms)" -ge "$deadline" ]; then
    echo "     /healthz did not respond within 120s — last 50 lines of serve log:"
    tail -n 50 "$LOG" || true
    end_phase
    echo
    echo "FAIL: serve never became healthy"
    exit 1
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "     serve process exited unexpectedly — last 50 lines of serve log:"
    tail -n 50 "$LOG" || true
    echo
    echo "FAIL: serve crashed before /healthz"
    exit 1
  fi
  if curl -fsS "$healthz_url" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
end_phase

# 5. Smoke chat completion ---------------------------------------------------
# Server is in --no-auth mode for the duration of the smoke test, so we
# don't have to round-trip an API-key file. Real deployments should run
# WITHOUT --no-auth.
start_phase smoke_chat
model_stem="${TEST_MODEL_FILE%.gguf}"
chat_url="http://${TEST_HOST}:${TEST_PORT}/v1/chat/completions"
chat_body=$(cat <<JSON
{"model":"${model_stem}","messages":[{"role":"user","content":"Reply with exactly the word: OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
JSON
)
chat_response=$(curl -fsS -X POST "$chat_url" \
  -H "Content-Type: application/json" \
  --data "$chat_body")
echo "     response: $chat_response"
end_phase

# 6. GPU-fallback assertion --------------------------------------------------
# Read the llama-server log forgemesh writes at FORGEMESH_HOME/llama-server.log.
# llama.cpp prints its backend identity (CUDA/Vulkan/Metal/CPU) and per-layer
# offload counts on startup. If the host has nvidia-smi but the log doesn't
# mention CUDA or Vulkan, we're silently running on CPU — fail loudly.
echo
echo "==> [check] GPU offload"
GPU_BACKEND="cpu"
GPU_OFFLOAD_LINE=""
if [ -f "$LLAMA_SERVER_LOG" ]; then
  if grep -qiE 'CUDA|cuBLAS' "$LLAMA_SERVER_LOG"; then
    GPU_BACKEND="cuda"
  elif grep -qiE 'Vulkan' "$LLAMA_SERVER_LOG"; then
    GPU_BACKEND="vulkan"
  elif grep -qiE 'Metal|MPS' "$LLAMA_SERVER_LOG"; then
    GPU_BACKEND="metal"
  elif grep -qiE 'ROCm|HIP' "$LLAMA_SERVER_LOG"; then
    GPU_BACKEND="rocm"
  fi
  GPU_OFFLOAD_LINE="$(grep -iE 'offloaded.*layers? to (GPU|device)|load_tensors:.*GPU' "$LLAMA_SERVER_LOG" | head -1 || true)"
fi
echo "     backend: $GPU_BACKEND"
if [ -n "$GPU_OFFLOAD_LINE" ]; then
  echo "     offload: $GPU_OFFLOAD_LINE"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$GPU_BACKEND" = "cpu" ] || [ "$GPU_BACKEND" = "metal" ]; then
    echo "     nvidia-smi present but llama-server reports backend=$GPU_BACKEND"
    echo "     last 30 lines of $LLAMA_SERVER_LOG:"
    tail -n 30 "$LLAMA_SERVER_LOG" 2>/dev/null || echo "     (log not found)"
    echo
    echo "FAIL: this box has an NVIDIA GPU but llama-server isn't using it."
    echo "      The prebuilt llama.cpp Vulkan binary may not be picking up"
    echo "      the NVIDIA Vulkan ICD. Workarounds:"
    echo "       - install the NVIDIA Vulkan driver / ICD"
    echo "       - or build llama.cpp from source with -DGGML_CUDA=ON and put"
    echo "         the resulting llama-server on \$PATH before re-running."
    exit 2
  fi
fi

# 7. Streaming probe ---------------------------------------------------------
# Confirm SSE streaming actually flows through ForgeMesh end-to-end. We use
# `--no-buffer` so curl flushes chunks and read the SSE stream line by line,
# scoring TTFT (time-to-first-token) and chunk count. Streaming is the path
# any real chat app will use, so silently shipping a broken stream would
# undermine the friend-test entirely.
start_phase stream_chat
stream_body=$(cat <<JSON
{"model":"${model_stem}","messages":[{"role":"user","content":"Count from one to ten in English. One per line."}],"max_tokens":64,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}
JSON
)
STREAM_LOG="$RESULTS_DIR/stream.log"
STREAM_RESULT_FILE="$RESULTS_DIR/stream-result.json"
python3 - "$chat_url" "$stream_body" "$STREAM_LOG" > "$STREAM_RESULT_FILE" <<'PY'
import json, sys, time
import urllib.request

url, body, out_path = sys.argv[1:4]
req = urllib.request.Request(url, data=body.encode("utf-8"),
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
start = time.perf_counter()
ttft_ms = None
chunks = 0
text_chars = 0
with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "w") as out:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        out.write(line + "\n")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        for ch in d.get("choices", []):
            delta_obj = ch.get("delta") or {}
            # Count both content and reasoning_content as "chunk arrived":
            # Qwen3-style thinking models route the chain-of-thought to
            # reasoning_content; the smoke test cares about whether the
            # stream is actually flowing, not which channel the bytes
            # ended up on.
            piece = delta_obj.get("content") or delta_obj.get("reasoning_content")
            if piece:
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - start) * 1000)
                chunks += 1
                text_chars += len(piece)
end = time.perf_counter()
total_ms = int((end - start) * 1000)
print(json.dumps({"ok": chunks > 0, "ttft_ms": ttft_ms, "total_ms": total_ms,
                  "chunks": chunks, "text_chars": text_chars}))
PY
echo "     stream: $(cat "$STREAM_RESULT_FILE")"
if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('ok') else 1)" "$STREAM_RESULT_FILE"; then
  echo "FAIL: streaming response yielded zero content chunks"
  exit 1
fi
end_phase

# 7b. Tiny bench --------------------------------------------------------------
start_phase bench
bench_json="$RESULTS_DIR/bench.json"
if ! forgemesh bench \
      --endpoint "http://${TEST_HOST}:${TEST_PORT}" \
      --api-key "smoke-no-auth" \
      --model "${model_stem}" \
      --max-tokens 64 \
      --runs 2 \
      --json > "$bench_json" 2> "$RESULTS_DIR/bench.stderr.log"; then
  echo "     forgemesh bench failed; stderr below:"
  tail -n 40 "$RESULTS_DIR/bench.stderr.log"
  end_phase
  echo "FAIL: bench did not complete"
  exit 1
fi
echo "     bench JSON written to: $bench_json"
end_phase

# 8. Snapshot /metrics (server-internal, not the bench harness) -------------
metrics_url="http://${TEST_HOST}:${TEST_PORT}/metrics"
METRICS_FILE="$RESULTS_DIR/metrics.json"
if curl -fsS "$metrics_url" > "$METRICS_FILE" 2>/dev/null; then
  echo "     metrics snapshot: $METRICS_FILE"
else
  echo "     warn: /metrics did not respond"
  echo '{}' > "$METRICS_FILE"
fi

T_TOTAL_END=$(ts_ms)
TOTAL_MS=$(( T_TOTAL_END - T_TOTAL_START ))

# 8. Summary -----------------------------------------------------------------
phase_ms() { eval "echo \${$(phase_ms_var "$1"):-0}"; }

echo
echo "==> PASS"
printf '     total wall: %.2fs\n' "$(echo "$TOTAL_MS/1000" | bc -l)"
for phase in install pull_model start_serve smoke_chat stream_chat bench; do
  ms="$(phase_ms "$phase")"
  printf '     %-12s %.2fs\n' "$phase" "$(echo "$ms/1000" | bc -l)"
done

# JSON summary the user can email back ---------------------------------------
python3 - "$SUMMARY_JSON" "$TOTAL_MS" "$(phase_ms install)" "$(phase_ms pull_model)" \
  "$(phase_ms start_serve)" "$(phase_ms smoke_chat)" "$(phase_ms stream_chat)" \
  "$(phase_ms bench)" \
  "$TEST_MODEL_REPO" "$TEST_MODEL_FILE" "$bench_json" \
  "$STREAM_RESULT_FILE" "$METRICS_FILE" "$GPU_BACKEND" "$LLAMA_SERVER_LOG" <<'PY'
import json, platform, subprocess, sys
(out, total, t_install, t_pull, t_start, t_chat, t_stream, t_bench, repo, fname,
 bench_json, stream_json, metrics_json, gpu_backend, llama_log) = sys.argv[1:16]
gpu = "unknown"
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0 and r.stdout.strip():
        gpu = r.stdout.strip().splitlines()[0].strip()
except Exception:
    pass
if gpu == "unknown" and platform.machine() == "arm64" and platform.system() == "Darwin":
    gpu = "Apple Silicon (Metal)"

def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

bench_data = _read_json(bench_json)
stream_data = _read_json(stream_json)
metrics_data = _read_json(metrics_json, default={})

# Pull just the offload line(s) from the llama-server log so the friend's
# summary.json is self-contained for triage.
offload_lines = []
try:
    with open(llama_log) as f:
        for line in f:
            low = line.lower()
            if ("offloaded" in low and "gpu" in low) or "load_tensors" in low:
                offload_lines.append(line.rstrip())
                if len(offload_lines) >= 5:
                    break
except Exception:
    pass

data = {
    "schema": 2,
    "host": {
        "uname": " ".join(platform.uname()),
        "python": platform.python_version(),
        "gpu": gpu,
    },
    "model": {"repo": repo, "file": fname},
    "backend": {
        "name": gpu_backend,
        "offload_lines": offload_lines,
    },
    "phases_ms": {
        "install": int(t_install),
        "pull_model": int(t_pull),
        "start_serve": int(t_start),
        "smoke_chat": int(t_chat),
        "stream_chat": int(t_stream),
        "bench": int(t_bench),
        "total": int(total),
    },
    "stream": stream_data,
    "bench": bench_data,
    "metrics": metrics_data,
}
with open(out, "w") as f:
    json.dump(data, f, indent=2)
print(f"\n     summary JSON: {out}")
PY

# Markdown summary the user can paste into an email --------------------------
{
  echo "# ForgeMesh smoke-test summary"
  echo
  echo "- host: \`$(uname -srm)\`"
  echo "- python: $(python3 -V 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "- gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  fi
  echo "- backend: \`$GPU_BACKEND\`"
  echo "- model: ${TEST_MODEL_REPO}/${TEST_MODEL_FILE}"
  echo
  echo "| phase | wall (s) |"
  echo "|---|---:|"
  for phase in install pull_model start_serve smoke_chat stream_chat bench; do
    ms="$(phase_ms "$phase")"
    printf '| %s | %.2f |\n' "$phase" "$(echo "$ms/1000" | bc -l)"
  done
  printf '| **total** | **%.2f** |\n' "$(echo "$TOTAL_MS/1000" | bc -l)"
  echo
  echo "Streaming: \`$STREAM_RESULT_FILE\`"
  echo "Raw bench: \`$bench_json\`"
  echo "Metrics:   \`$METRICS_FILE\`"
} > "$SUMMARY_MD"
echo "     summary MD:   $SUMMARY_MD"
echo
echo "If you're sharing this with a friend: please send back $SUMMARY_JSON and $SUMMARY_MD."
