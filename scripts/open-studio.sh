#!/bin/bash
set -euo pipefail
BB8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_ROOT="${GENESIS_STUDIO_ROOT:-$(dirname -- "$BB8_ROOT")/Genesis-Studio}"
exec "$STUDIO_ROOT/scripts/run-genesis-desktop.sh" --project "$BB8_ROOT/projects/bb8/empty-floor.genesis.json" "$@"
