#!/usr/bin/env bash
# Resume one isolated English checkpoint for one additional training hour.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
if (( $# > 1 )); then
    printf 'Usage: bash ops/run_language_hour.sh [RUN_DIR]\n' >&2
    exit 2
fi
RUN_DIR="${1:-runs/language-1h}"
CONFIG="configs/language-hour.toml"
DATA="data/english/dataset.json"
python3 "$PROJECT_ROOT/ops/worker_guard.py" --resume --validate "$PROJECT_ROOT" "$CONFIG" "$DATA" "$RUN_DIR"
bash "$PROJECT_ROOT/ops/preflight.sh"
UNIT="golem-language-hour-$(date -u +%Y%m%dT%H%M%S)-$$"
systemd-run --user --collect --unit="$UNIT" \
    --description="Project Golem: one-hour English language evolution" \
    --property="WorkingDirectory=$PROJECT_ROOT" \
    --property=CPUQuota=400% --property=MemoryMax=8G --property=TasksMax=64 \
    --property=Nice=19 --property=IOSchedulingClass=idle \
    --property=RuntimeMaxSec=3660 --property=TimeoutStopSec=20 \
    --property=KillMode=control-group --property=NoNewPrivileges=yes \
    --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
    --setenv=PYTHONHASHSEED=0 --setenv=OMP_NUM_THREADS=4 \
    --setenv=OPENBLAS_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
    --setenv=NUMEXPR_NUM_THREADS=4 --setenv=CUDA_VISIBLE_DEVICES= \
    --setenv=HIP_VISIBLE_DEVICES= --setenv=ROCR_VISIBLE_DEVICES= \
    python3 "$PROJECT_ROOT/ops/worker_guard.py" --resume --run "$PROJECT_ROOT" \
        --config "$CONFIG" --data "$DATA" --run-dir "$RUN_DIR"
printf 'Started unit: %s.service\n' "$UNIT"
printf 'Training budget: 3600 seconds; supervisor allows 60 additional seconds for checkpoint shutdown.\n'
printf 'Status: ./golem status --run-dir %q\n' "$RUN_DIR"
printf 'Chat: ./golem chat --checkpoint %q/champion.json\n' "$RUN_DIR"
printf 'Stop only this run: systemctl --user stop %q.service\n' "$UNIT"
