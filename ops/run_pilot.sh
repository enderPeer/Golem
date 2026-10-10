#!/usr/bin/env bash
# Start only this project's bounded transient CPU job. No installation/autostart.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if (( $# > 3 )); then
    printf 'Usage: bash ops/run_pilot.sh [--dry-run] [CONFIG [DATA [RUN_DIR]]]\n' >&2
    exit 2
fi
CONFIG="${1:-configs/pilot.toml}"
DATA="${2:-data/english/dataset.json}"
RUN_DIR="${3:-runs/pilot}"
python3 "$PROJECT_ROOT/ops/worker_guard.py" --validate "$PROJECT_ROOT" "$CONFIG" "$DATA" "$RUN_DIR"
bash "$PROJECT_ROOT/ops/preflight.sh"
UNIT="golem-pilot-$(date -u +%Y%m%dT%H%M%S)-$$"
COMMAND=(systemd-run --user --collect --unit="$UNIT"
    --description="Bounded Project Golem CPU pilot"
    --property="WorkingDirectory=$PROJECT_ROOT"
    --property=CPUQuota=400% --property=MemoryMax=8G --property=TasksMax=64
    --property=Nice=19 --property=RuntimeMaxSec=1800
    --property=IOSchedulingClass=idle
    --property=TimeoutStopSec=20 --property=KillMode=control-group
    --property=NoNewPrivileges=yes
    --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1
    --setenv=PYTHONHASHSEED=0 --setenv=OMP_NUM_THREADS=4
    --setenv=OPENBLAS_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4
    --setenv=NUMEXPR_NUM_THREADS=4 --setenv=CUDA_VISIBLE_DEVICES=
    --setenv=HIP_VISIBLE_DEVICES= --setenv=ROCR_VISIBLE_DEVICES=
    python3 "$PROJECT_ROOT/ops/worker_guard.py" --run "$PROJECT_ROOT"
    --config "$CONFIG" --data "$DATA" --run-dir "$RUN_DIR")
if (( DRY_RUN )); then
    printf 'Dry run; no job started. Command:\n'
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
command -v systemd-run >/dev/null || { printf 'systemd-run is required.\n' >&2; exit 1; }
"${COMMAND[@]}"
printf 'Started unit: %s.service\n' "$UNIT"
printf 'Status: systemctl --user status %q.service\n' "$UNIT"
printf 'Logs: journalctl --user -u %q.service\n' "$UNIT"
printf 'Stop only this job: systemctl --user stop %q.service\n' "$UNIT"
