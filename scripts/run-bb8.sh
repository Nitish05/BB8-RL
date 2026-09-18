#!/bin/bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/python.sh" -m bb8_rl.cli "$@"
