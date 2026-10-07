#!/usr/bin/env bash
set -euo pipefail

# Run the Q1 2-D convolution solution.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q1_Implement_2D_Convolution.py"

if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  echo "NumPy is not installed. Installing it now..."
  "$PY" -m pip install numpy
fi

exec "$PY" "$SCRIPT" "$@"