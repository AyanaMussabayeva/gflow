#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[build_frozen_notebooks.sh] Root: $ROOT_DIR"
echo "[build_frozen_notebooks.sh] Generating frozen notebooks from saved artifacts"

python -m pkg.cli.build_frozen_notebooks

echo "[build_frozen_notebooks.sh] Done."
