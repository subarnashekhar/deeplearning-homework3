#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q2_Transfer_Learning_Freeze_vs_Fine_Tune.py"
VENV_DIR="$WORKDIR/.venv-q2"
VENV_PY="$VENV_DIR/bin/python"
IS_INTEL_MAC=false

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
  IS_INTEL_MAC=true
fi

if [[ ! -x "$VENV_PY" ]]; then
  if [[ "$IS_INTEL_MAC" == true ]]; then
    if [[ -x "/usr/local/bin/python3.11" ]]; then
      BASE_PY="/usr/local/bin/python3.11"
    elif command -v python3.11 >/dev/null 2>&1; then
      BASE_PY="$(command -v python3.11)"
    else
      echo "Intel Macs need Python 3.11 for the available PyTorch wheels. Install Python 3.11 and rerun." >&2
      exit 1
    fi
  elif [[ -x "$WORKDIR/.venv/bin/python" ]]; then
    BASE_PY="$WORKDIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PY="$(command -v python3)"
  else
    echo "Python 3 not found in PATH." >&2
    exit 1
  fi
  "$BASE_PY" -m venv "$VENV_DIR"
fi

"$VENV_PY" -m pip install --upgrade "numpy<2" >/dev/null 2>&1 || true

if ! "$VENV_PY" -c "import torch, torchvision, matplotlib" >/dev/null 2>&1; then
  echo "Installing PyTorch, torchvision, matplotlib, and a NumPy version compatible with the local PyTorch build in $VENV_DIR..."
  if [[ "$IS_INTEL_MAC" == true ]]; then
    "$VENV_PY" -m pip install "numpy<2" "torch==2.2.2" "torchvision==0.17.2" matplotlib
  else
    "$VENV_PY" -m pip install "numpy<2" torch torchvision matplotlib
  fi
fi

exec "$VENV_PY" "$SCRIPT" "$@"