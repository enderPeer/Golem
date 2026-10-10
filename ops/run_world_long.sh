#!/usr/bin/env bash
# The unit timer is independent of both the supervisor and the desktop app.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
CONFIG="${1:?config required}"
RUN_DIR="${2:?dedicated supervisor directory required}"
DEADLINE="${3:?explicit UTC deadline required}"
RUNTIME="${GOLEM_PYTHON:-$ROOT/.venv-neural/bin/python}"
THREADS="${GOLEM_THREADS:-2}"
MEMORY_MIB="${GOLEM_MEMORY_MIB:-2048}"
RESERVE="${GOLEM_MIN_FREE_GIB:-50}"
RESUME_ARGS=()
case "${GOLEM_RESUME:-0}" in
    0) ;;
    1) RESUME_ARGS=(--resume) ;;
    *) printf 'GOLEM_RESUME must be 0 or 1.\n' >&2; exit 2 ;;
esac
test -x "$RUNTIME"
[[ "$THREADS" =~ ^[1-9][0-9]?$ ]] && (( THREADS <= 64 ))
[[ "$MEMORY_MIB" =~ ^[0-9]+$ ]] && (( MEMORY_MIB >= 512 && MEMORY_MIB <= 8192 ))
# Start shutdown 30 seconds early; TimeoutStopSec limits any final save to 20s.
LIMIT="$(python3 -c 'from datetime import datetime,timezone; import math,sys; d=datetime.fromisoformat(sys.argv[1].replace("Z","+00:00")); assert d.tzinfo is not None and d.utcoffset().total_seconds()==0; print(math.floor(d.timestamp()-datetime.now(timezone.utc).timestamp())-30)' "$DEADLINE")"
(( LIMIT > 60 && LIMIT <= 172800 )) || { printf 'Deadline must be 90 seconds to two days ahead.\n' >&2; exit 2; }
UNIT="golem-world-long-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$ROOT/cache" "$ROOT/runs/world-long-launches"
systemd-run --user --collect --unit="$UNIT" \
    --description="Golem conversation worlds until $DEADLINE" \
    --property="WorkingDirectory=$ROOT" --property="CPUQuota=$((THREADS * 100))%" \
    --property="MemoryMax=${MEMORY_MIB}M" --property=TasksMax=128 --property=Nice=19 \
    --property=IOSchedulingClass=idle --property="RuntimeMaxSec=$LIMIT" \
    --property=TimeoutStopSec=20 --property=KillMode=control-group --property=NoNewPrivileges=yes \
    --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
    --setenv="OMP_NUM_THREADS=$THREADS" --setenv="MKL_NUM_THREADS=$THREADS" \
    --setenv="OPENBLAS_NUM_THREADS=$THREADS" --setenv="XDG_CACHE_HOME=$ROOT/cache" \
    --setenv="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --setenv="HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" \
    "$RUNTIME" "$ROOT/ops/world_supervisor.py" --config-base "$CONFIG" \
        --data data/conversation-world/dataset.json --run-dir "$RUN_DIR" --deadline "$DEADLINE" \
        --stage-seconds 1800 --shutdown-grace 15 --memory-limit-mib "$MEMORY_MIB" \
        --min-free-gib "$RESERVE" --run-budget-gib 1 --project-budget-gib 20 --self-replay "${RESUME_ARGS[@]}"
printf '%s.service\n' "$UNIT" | tee "$ROOT/runs/world-long-launches/$(basename -- "$RUN_DIR").unit"
