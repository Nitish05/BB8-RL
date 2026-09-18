#!/bin/bash
set -euo pipefail
BB8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_ROOT="${GENESIS_STUDIO_ROOT:-$(dirname -- "$BB8_ROOT")/Genesis-Studio}"
if [[ -n "${BB8_PYTHON:-}" ]]; then
  BB8_RUNTIME="$BB8_PYTHON"
elif [[ -x "$BB8_ROOT/.venv/bin/python" ]]; then
  BB8_RUNTIME="$BB8_ROOT/.venv/bin/python"
else
  BB8_RUNTIME="$STUDIO_ROOT/.venv-genesis/bin/python"
fi
if [[ ! -x "$BB8_RUNTIME" ]]; then
  echo "Python runtime not found. Install BB8-RL in .venv or set BB8_PYTHON; see README.md." >&2
  exit 1
fi
export PYTHONPATH="$BB8_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$BB8_ROOT/work/matplotlib-cache}"
cd -- "$BB8_ROOT"
exec "$BB8_RUNTIME" "$@"
