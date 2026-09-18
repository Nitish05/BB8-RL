#!/bin/bash
set -euo pipefail
BB8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$BB8_ROOT/scripts/launch-control-room.sh" "$@"
