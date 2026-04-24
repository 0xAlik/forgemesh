#!/usr/bin/env bash
# ForgeMesh installer — pip installs the package from the tagged GitHub
# release into an isolated venv at $FORGEMESH_HOME (default ~/.forgemesh)
# and links a `forgemesh` shim into $INSTALL_BIN (default ~/.local/bin).
#
# ForgeMesh is deliberately NOT on PyPI during pre-alpha. Installs pull
# from the git tag so every box gets the exact same wheel-equivalent.
#
# Requires: python >= 3.11, git, and `llama-server` from llama.cpp on PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/install.sh | bash
#
# Pin a different version:
#   FORGEMESH_VERSION=0.0.3 curl -fsSL .../install.sh | bash
# Track a branch (unpinned; not recommended):
#   FORGEMESH_REF=main curl -fsSL .../install.sh | bash

set -euo pipefail

FORGEMESH_HOME="${FORGEMESH_HOME:-$HOME/.forgemesh}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
FORGEMESH_VERSION="${FORGEMESH_VERSION:-0.0.2}"
FORGEMESH_REF="${FORGEMESH_REF:-v$FORGEMESH_VERSION}"
FORGEMESH_REPO="${FORGEMESH_REPO:-https://github.com/0xAlik/forgemesh.git}"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have python3; then
  echo "error: python3 not found on PATH." >&2
  exit 1
fi

if ! have git; then
  echo "error: git not found on PATH (pip needs it to fetch the repo)." >&2
  exit 1
fi

PY_MAJOR_MINOR=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PY_MAJOR_MINOR" in
  3.11|3.12|3.13)
    ;;
  *)
    echo "error: need Python 3.11+, found $PY_MAJOR_MINOR" >&2
    exit 1
    ;;
esac

if ! have llama-server; then
  echo "warning: 'llama-server' not found on PATH." >&2
  echo "  ForgeMesh will not start until you install llama.cpp." >&2
  echo "  See: https://github.com/ggerganov/llama.cpp" >&2
fi

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
