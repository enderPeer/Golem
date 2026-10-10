#!/usr/bin/env bash
# Read-only checks of this project's filesystem and Linux GPU metadata.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec python3 "$PROJECT_ROOT/ops/worker_guard.py" --preflight "$PROJECT_ROOT"
