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

FORGEMESH_HOME="${FORGEMESH_HOME:-$HOME/.forgemesh}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
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

# 5. Read API key (first start-up writes it once) ---------------------------
API_KEY_FILE="${FORGEMESH_HOME}/api-key"
if [ -f "$API_KEY_FILE" ]; then
  API_KEY="$(cat "$API_KEY_FILE")"
else
  API_KEY=""
fi

# 6. Smoke chat completion ---------------------------------------------------
start_phase smoke_chat
model_stem="${TEST_MODEL_FILE%.gguf}"
chat_url="http://${TEST_HOST}:${TEST_PORT}/v1/chat/completions"
chat_body=$(cat <<JSON
{"model":"${model_stem}","messages":[{"role":"user","content":"Reply with exactly the word: OK."}],"max_tokens":8,"temperature":0}
JSON
)
chat_response=$(curl -fsS -X POST "$chat_url" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  --data "$chat_body")
echo "     response: $chat_response"
end_phase

# 7. Tiny bench --------------------------------------------------------------
start_phase bench
bench_json="$RESULTS_DIR/bench.json"
if ! forgemesh bench \
      --endpoint "http://${TEST_HOST}:${TEST_PORT}" \
      --api-key "${API_KEY}" \
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

T_TOTAL_END=$(ts_ms)
TOTAL_MS=$(( T_TOTAL_END - T_TOTAL_START ))

# 8. Summary -----------------------------------------------------------------
phase_ms() { eval "echo \${$(phase_ms_var "$1"):-0}"; }

echo
echo "==> PASS"
printf '     total wall: %.2fs\n' "$(echo "$TOTAL_MS/1000" | bc -l)"
for phase in install pull_model start_serve smoke_chat bench; do
  ms="$(phase_ms "$phase")"
  printf '     %-12s %.2fs\n' "$phase" "$(echo "$ms/1000" | bc -l)"
done

# JSON summary the user can email back ---------------------------------------
python3 - "$SUMMARY_JSON" "$TOTAL_MS" "$(phase_ms install)" "$(phase_ms pull_model)" \
  "$(phase_ms start_serve)" "$(phase_ms smoke_chat)" "$(phase_ms bench)" \
  "$TEST_MODEL_REPO" "$TEST_MODEL_FILE" "$bench_json" <<'PY'
import json, os, platform, subprocess, sys
out, total, t_install, t_pull, t_start, t_chat, t_bench, repo, fname, bench_json = sys.argv[1:11]
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
try:
    with open(bench_json) as f:
        bench_data = json.load(f)
except Exception:
    bench_data = None
data = {
    "schema": 1,
    "host": {
        "uname": " ".join(platform.uname()),
        "python": platform.python_version(),
        "gpu": gpu,
    },
    "model": {"repo": repo, "file": fname},
    "phases_ms": {
        "install": int(t_install),
        "pull_model": int(t_pull),
        "start_serve": int(t_start),
        "smoke_chat": int(t_chat),
        "bench": int(t_bench),
        "total": int(total),
    },
    "bench": bench_data,
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
  echo "- model: ${TEST_MODEL_REPO}/${TEST_MODEL_FILE}"
  echo
  echo "| phase | wall (s) |"
  echo "|---|---:|"
  for phase in install pull_model start_serve smoke_chat bench; do
    ms="$(phase_ms "$phase")"
    printf '| %s | %.2f |\n' "$phase" "$(echo "$ms/1000" | bc -l)"
  done
  printf '| **total** | **%.2f** |\n' "$(echo "$TOTAL_MS/1000" | bc -l)"
  echo
  echo "Raw bench: \`$bench_json\`"
} > "$SUMMARY_MD"
echo "     summary MD:   $SUMMARY_MD"
echo
echo "If you're sharing this with a friend: please send back $SUMMARY_JSON and $SUMMARY_MD."
