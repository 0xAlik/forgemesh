#!/usr/bin/env bash
# ForgeMesh installer — pip installs the package into an isolated venv
# at $FORGEMESH_HOME (default ~/.forgemesh) and links a `forgemesh`
# shim into $INSTALL_BIN (default ~/.local/bin).
#
# Requires: python >= 3.11, `llama-server` from llama.cpp on PATH (see README).
#
#   curl -fsSL https://raw.githubusercontent.com/0xAlik/forgemesh/main/install.sh | bash

set -euo pipefail

FORGEMESH_HOME="${FORGEMESH_HOME:-$HOME/.forgemesh}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
FORGEMESH_VERSION="${FORGEMESH_VERSION:-0.0.2}"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have python3; then
  echo "error: python3 not found on PATH." >&2
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

echo "==> installing forgemesh==$FORGEMESH_VERSION"
"$VENV/bin/pip" install "forgemesh==$FORGEMESH_VERSION" || {
  echo "  pip install from PyPI failed; falling back to source install from the current directory" >&2
  "$VENV/bin/pip" install .
}

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
