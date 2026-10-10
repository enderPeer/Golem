#!/usr/bin/env bash
# RUNTIME and device visibility are selected explicitly after host inventory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
CONFIG="${1:?config required}"
RUN_DIR="${2:?fresh run directory required}"
RUNTIME="${GOLEM_PYTHON:-$ROOT/.venv-neural/bin/python}"
THREADS="${GOLEM_THREADS:-2}"
RESERVE="${GOLEM_MIN_FREE_GIB:-50}"
MEMORY_MIB="${GOLEM_MEMORY_MIB:-2048}"
test -x "$RUNTIME"
[[ "$THREADS" =~ ^[1-9][0-9]?$ ]] && (( THREADS <= 64 ))
[[ "$MEMORY_MIB" =~ ^[0-9]+$ ]] && (( MEMORY_MIB >= 512 && MEMORY_MIB <= 8192 ))
UNIT="golem-chat-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$ROOT/cache" "$ROOT/runs/launches"
systemd-run --user --collect --unit="$UNIT" \
    --description="Golem isolated chat-training island" \
    --property="WorkingDirectory=$ROOT" --property="CPUQuota=$((THREADS * 100))%" \
    --property="MemoryMax=${MEMORY_MIB}M" --property=TasksMax=96 --property=Nice=19 \
    --property=IOSchedulingClass=idle --property=RuntimeMaxSec=3630 \
    --property=TimeoutStopSec=25 --property=KillMode=control-group --property=NoNewPrivileges=yes \
    --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
    --setenv="OMP_NUM_THREADS=$THREADS" --setenv="MKL_NUM_THREADS=$THREADS" \
    --setenv="OPENBLAS_NUM_THREADS=$THREADS" --setenv="XDG_CACHE_HOME=$ROOT/cache" \
    --setenv="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --setenv="HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" \
    "$RUNTIME" "$ROOT/ops/chat_guard.py" --config "$CONFIG" --run-dir "$RUN_DIR" \
        --minimum-free-gib "$RESERVE"
printf '%s.service\n' "$UNIT" | tee "$ROOT/runs/launches/$(basename -- "$RUN_DIR").unit"
