#!/usr/bin/env bash
# ForgeMesh installer — pip installs the package from the tagged GitHub
# release into an isolated venv at $FORGEMESH_HOME (default ~/.forgemesh)
# and links a `forgemesh` shim into $INSTALL_BIN (default ~/.local/bin).
# Also installs a prebuilt `llama-server` into $FORGEMESH_HOME/llama.cpp
# when one isn't already on PATH (Linux x64 Vulkan, macOS Metal, Linux
# arm64 CPU-only).
#
# ForgeMesh is deliberately NOT on PyPI during pre-alpha. Installs pull
# from the git tag so every box gets the exact same wheel-equivalent.
#
# Requires: python >= 3.11, git, curl, tar.
#
#   curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/install.sh | bash
#
# Pin a different forgemesh version:
#   FORGEMESH_VERSION=0.0.3 curl -fsSL .../install.sh | bash
# Track a branch (unpinned; not recommended):
#   FORGEMESH_REF=main curl -fsSL .../install.sh | bash
# Pin a specific upstream llama.cpp release:
#   LLAMA_CPP_VERSION=b8920 curl -fsSL .../install.sh | bash
# Skip the llama.cpp auto-install (bring your own):
#   FORGEMESH_SKIP_LLAMA=1 curl -fsSL .../install.sh | bash

set -euo pipefail

FORGEMESH_HOME="${FORGEMESH_HOME:-$HOME/.forgemesh}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
FORGEMESH_VERSION="${FORGEMESH_VERSION:-0.0.2}"
FORGEMESH_REF="${FORGEMESH_REF:-v$FORGEMESH_VERSION}"
FORGEMESH_REPO="${FORGEMESH_REPO:-https://github.com/0xAlik/forgemesh.git}"
FORGEMESH_SKIP_LLAMA="${FORGEMESH_SKIP_LLAMA:-0}"
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-latest}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-ggml-org/llama.cpp}"
# Fallback tag if the GitHub API can't be reached (rate-limit, network, etc.).
# Bumped manually when we want to track a newer upstream by default.
LLAMA_CPP_FALLBACK_TAG="${LLAMA_CPP_FALLBACK_TAG:-b8920}"

have() { command -v "$1" >/dev/null 2>&1; }

for tool in python3 git curl tar; do
  if ! have "$tool"; then
    echo "error: $tool not found on PATH." >&2
    exit 1
  fi
done

PY_MAJOR_MINOR=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PY_MAJOR_MINOR" in
  3.11|3.12|3.13)
    ;;
  *)
    echo "error: need Python 3.11+, found $PY_MAJOR_MINOR" >&2
    exit 1
    ;;
esac

mkdir -p "$FORGEMESH_HOME" "$INSTALL_BIN"
VENV="$FORGEMESH_HOME/venv"

echo "==> creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null

echo "==> installing forgemesh from ${FORGEMESH_REPO}@${FORGEMESH_REF}"
"$VENV/bin/pip" install "git+${FORGEMESH_REPO}@${FORGEMESH_REF}"

SHIM="$INSTALL_BIN/forgemesh"
cat > "$SHIM" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/forgemesh" "\$@"
EOF
chmod +x "$SHIM"

# ---------------------------------------------------------------------------
# llama.cpp auto-install (prebuilt binary, no source build)
# ---------------------------------------------------------------------------
# Picks a prebuilt asset from upstream llama.cpp releases. NVIDIA users get
# the Vulkan build — llama.cpp does not ship Linux CUDA prebuilts, and Vulkan
# works on NVIDIA, AMD, and Intel GPUs out of the box (with the vendor driver
# installed). For maximum NVIDIA throughput, see the README for the
# source-build escape hatch — build once, drop `llama-server` on PATH, and
# this step skips itself next time.

install_llama_cpp() {
  local py="$VENV/bin/python"
  local os arch asset_name
  os=$(uname -s)
  arch=$(uname -m)

  case "$os:$arch" in
    Linux:x86_64)   asset_name="bin-ubuntu-vulkan-x64.tar.gz" ;;
    Linux:aarch64)  asset_name="bin-ubuntu-arm64.tar.gz" ;;
    Darwin:arm64)   asset_name="bin-macos-arm64.tar.gz" ;;
    Darwin:x86_64)  asset_name="bin-macos-x64.tar.gz" ;;
    *)
      echo "note: no prebuilt llama.cpp for $os/$arch."
      echo "      ForgeMesh will not start until \`llama-server\` is on PATH."
      echo "      See: https://github.com/${LLAMA_CPP_REPO}"
      return 0
      ;;
  esac

  local tag
  if [ "$LLAMA_CPP_VERSION" = "latest" ]; then
    echo "==> resolving latest llama.cpp release tag (5s connect / 15s total)"
    tag=$(curl -fsSL --connect-timeout 5 --max-time 15 \
            "https://api.github.com/repos/${LLAMA_CPP_REPO}/releases/latest" 2>/dev/null \
          | "$py" -c 'import json,sys
try:
    print(json.load(sys.stdin)["tag_name"])
except Exception:
    pass' 2>/dev/null) || tag=""
    if [ -z "$tag" ]; then
      echo "     GitHub API unreachable or rate-limited — falling back to pinned tag $LLAMA_CPP_FALLBACK_TAG"
      tag="$LLAMA_CPP_FALLBACK_TAG"
    else
      echo "     resolved: $tag"
    fi
  else
    tag="$LLAMA_CPP_VERSION"
  fi

  local url="https://github.com/${LLAMA_CPP_REPO}/releases/download/${tag}/llama-${tag}-${asset_name}"
  local llama_home="$FORGEMESH_HOME/llama.cpp"
  local tmp
  tmp=$(mktemp -d)

  echo "==> downloading llama.cpp ${tag} ($asset_name)"
  echo "    $url"
  curl -fL --connect-timeout 15 --max-time 900 --retry 2 \
       -o "$tmp/llama.tar.gz" "$url"
  echo "     downloaded $(du -h "$tmp/llama.tar.gz" | cut -f1)"

  echo "==> extracting to $llama_home"
  tar -xzf "$tmp/llama.tar.gz" -C "$tmp"
  local inner
  inner=$(find "$tmp" -maxdepth 1 -mindepth 1 -type d -name "llama-*" | head -n1)
  if [ -z "$inner" ] || [ ! -x "$inner/llama-server" ]; then
    echo "error: extracted archive did not contain 'llama-server'." >&2
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$llama_home"
  mv "$inner" "$llama_home"
  rm -rf "$tmp"

  local llama_shim="$INSTALL_BIN/llama-server"
  cat > "$llama_shim" <<EOF
#!/usr/bin/env bash
exec "$llama_home/llama-server" "\$@"
EOF
  chmod +x "$llama_shim"

  echo "     llama.cpp: $llama_home"
  echo "     llama-server shim: $llama_shim"
}

if [ "$FORGEMESH_SKIP_LLAMA" = "1" ]; then
  echo "==> skipping llama.cpp auto-install (FORGEMESH_SKIP_LLAMA=1)"
  if ! have llama-server; then
    echo "     note: \`llama-server\` is not on PATH; ForgeMesh will not start until you install llama.cpp."
  fi
elif have llama-server; then
  echo "==> \`llama-server\` already on PATH — skipping llama.cpp auto-install"
else
  install_llama_cpp
fi

echo
echo "==> installed:"
echo "     venv:   $VENV"
echo "     shim:   $SHIM"
echo
if ! echo ":$PATH:" | grep -q ":$INSTALL_BIN:"; then
  echo "note: $INSTALL_BIN is not on your PATH."
  echo "      add it with: echo 'export PATH=\"$INSTALL_BIN:\$PATH\"' >> ~/.bashrc"
fi
echo "run: forgemesh version"
