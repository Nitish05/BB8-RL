#!/bin/bash
# Isolated CPU dependency overlay; never installs into the Studio runtime.
set -euo pipefail
BB8_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$BB8_ROOT"
BB8_DREAMER_REV="e3f02248693a79dc8b0ebd62c93683888ddaccfe"
BB8_DREAMER_SRC="$BB8_ROOT/work/m5/dreamerv3-upstream"
mkdir -p work/m5
if [[ ! -d "$BB8_DREAMER_SRC/.git" ]]; then
  git clone --no-checkout https://github.com/danijar/dreamerv3.git "$BB8_DREAMER_SRC"
  git -C "$BB8_DREAMER_SRC" checkout "$BB8_DREAMER_REV"
fi
if [[ "$(git -C "$BB8_DREAMER_SRC" rev-parse HEAD)" != "$BB8_DREAMER_REV" ]]; then
  echo 'Dreamer checkout differs from the pinned revision; preserve it and choose a fresh checkout.' >&2
  exit 1
fi
if [[ ! -x .venv-dreamer/bin/python ]]; then
  ./scripts/python.sh -m venv .venv-dreamer
fi
./scripts/python.sh - <<'PY'
import site
from pathlib import Path
overlay = next(Path('.venv-dreamer/lib').glob('python*/site-packages'))
# Process editable-install .pth files as well as ordinary native dependencies.
line = 'import site; site.addsitedir(' + repr(site.getsitepackages()[0]) + ')\n'
(overlay / 'bb8_native.pth').write_text(line)
PY
.venv-dreamer/bin/python -m pip install -r configs/training/requirements-dreamer-cpu.txt
echo 'Use BB8_PYTHON="$PWD/.venv-dreamer/bin/python" with scripts/python.sh or scripts/run-bb8.sh.'
