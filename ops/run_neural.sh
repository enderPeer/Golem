#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
MODE="${1:-compare}"
WORKER_MODE="$MODE"
case "$MODE" in
    memory) CONFIG=configs/neural-memory.toml; LIMIT=270 ;;
    benchmark) CONFIG=configs/neural-pilot.toml; LIMIT=180 ;;
    compare) CONFIG=configs/neural-pilot.toml; LIMIT=1230 ;;
    language) CONFIG=configs/neural-language.toml; LIMIT=1230; WORKER_MODE=compare ;;
    qat) CONFIG=configs/neural-qat.toml; LIMIT=210 ;;
    *) printf 'Usage: bash ops/run_neural.sh [benchmark|compare|language|memory|qat] [RUN_DIR] [QAT_CHECKPOINT]\n' >&2; exit 2 ;;
esac
RUN_DIR="${2:-runs/neural-$MODE}"
EXTRA=()
if [[ "$MODE" == qat ]]; then
    EXTRA=(--checkpoint "${3:-runs/neural-compare/ordinary_champion.pt}")
fi
test -x .venv-neural/bin/python || { printf 'Create .venv-neural and install requirements-neural.txt first.\n' >&2; exit 1; }
UNIT="golem-neural-$MODE-$(date -u +%Y%m%dT%H%M%S)-$$"
systemd-run --user --collect --unit="$UNIT" \
    --description="Project Golem bounded neural $MODE" \
    --property="WorkingDirectory=$PROJECT_ROOT" \
    --property=CPUQuota=200% --property=MemoryMax=8G --property=TasksMax=64 \
    --property=Nice=19 --property=IOSchedulingClass=idle \
    --property="RuntimeMaxSec=$LIMIT" --property=TimeoutStopSec=20 \
    --property=KillMode=control-group --property=NoNewPrivileges=yes \
    --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
    --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
    --setenv=OPENBLAS_NUM_THREADS=2 --setenv=CUDA_VISIBLE_DEVICES= \
    --setenv=HIP_VISIBLE_DEVICES= --setenv=ROCR_VISIBLE_DEVICES= \
    "$PROJECT_ROOT/.venv-neural/bin/python" "$PROJECT_ROOT/ops/neural_guard.py" "$WORKER_MODE" \
        --config "$CONFIG" --data data/neural-dialogue/dataset.json --run-dir "$RUN_DIR" "${EXTRA[@]}"
printf 'Started: %s.service\nStatus: ./golem neural status --run-dir %s\n' "$UNIT" "$RUN_DIR"
