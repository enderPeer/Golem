"""Run only this project's chat worker, with storage and memory guardrails.

CPU priority and quota belong to the launching systemd unit. This process never
signals an unrelated job. Temporary-memory deployments use their own small disk
reserve; persistent deployments retain the existing 50 GiB reserve.
"""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.worker_guard import contained, project_bytes, stop_worker

GIB = 2 ** 30


def check_storage(root, minimum_free_gib):
    free = shutil.disk_usage(root).free
    size = project_bytes(root)
    if free < minimum_free_gib * GIB:
        raise RuntimeError(f'Golem stopped: free storage below {minimum_free_gib} GiB')
    if size > 20 * GIB:
        raise RuntimeError('Golem stopped: dedicated project exceeds 20 GiB')
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    if memory.get('MemAvailable', 0) < 2 * GIB:
        raise RuntimeError('Golem stopped: less than 2 GiB available host memory')
    return {'free_storage_gib': free / GIB, 'project_gib': size / GIB,
            'available_memory_gib': memory['MemAvailable'] / GIB}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='data/chat/dataset.json')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--minimum-free-gib', type=float, default=50.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not 2 <= args.minimum_free_gib <= 10000:
        parser.error('Storage reserve must be at least 2 GiB')
    # A smaller reserve is permitted only for our explicitly temporary checkout.
    if args.minimum_free_gib < 50 and not root.is_relative_to(Path('/tmp')):
        parser.error('Persistent checkouts require the 50 GiB storage reserve')
    config = contained(root, args.config, must_exist=True)
    data = contained(root, args.data, must_exist=True)
    run = contained(root, args.run_dir)
    cfg = tomllib.loads(config.read_text())
    if cfg.get('init_checkpoint'):
        contained(root, cfg['init_checkpoint'], must_exist=True)
    if run.exists() and any(run.iterdir()) and not args.resume:
        parser.error('Use a fresh run directory or explicit resume')
    print(check_storage(root, args.minimum_free_gib), flush=True)
    command = [sys.executable, '-B', '-m', 'english_world', 'neural', 'chat-train',
               '--config', str(config), '--data', str(data), '--run-dir', str(run)]
    if args.resume:
        command.append('--resume')
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    def stop(signum, _frame):
        stop_worker(child)
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        while child.poll() is None:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                check_storage(root, args.minimum_free_gib)
        return child.returncode
    finally:
        stop_worker(child)


if __name__ == '__main__':
    raise SystemExit(main())
