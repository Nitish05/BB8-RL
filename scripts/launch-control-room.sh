#!/bin/bash
set -euo pipefail
BB8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${BB8_PYTHON:-}" && -x "$BB8_ROOT/.venv-dreamer/bin/python" ]]; then
  export BB8_PYTHON="$BB8_ROOT/.venv-dreamer/bin/python"
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$BB8_ROOT/work/numba-cache}"
exec "$BB8_ROOT/scripts/python.sh" -m bb8_rl.interactive "$@"
