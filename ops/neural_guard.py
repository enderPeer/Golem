"""Run a bounded neural worker with the existing project storage guard."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.worker_guard import budget_check, contained, preflight, stop_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "compare", "memory", "qat"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/neural-dialogue/dataset.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = contained(root, args.config, must_exist=True)
    run = contained(root, args.run_dir)
    if args.resume:
        if args.mode != "compare":
            parser.error("Only comparison runs support resume")
        contained(root, run / "training.pt", must_exist=True)
    elif run.exists() and (not run.is_dir() or any(run.iterdir())):
        parser.error("Run directory must be new or empty")
    command = [sys.executable, "-B", "-m", "english_world", "neural", args.mode,
               "--config", str(config), "--run-dir", str(run), "--threads", "2"]
    if args.mode != "memory":
        data = contained(root, args.data, must_exist=True)
        command += ["--data", str(data)]
    if args.resume:
        command.append("--resume")
    if args.mode == "qat":
        if not args.checkpoint:
            parser.error("QAT requires --checkpoint")
        checkpoint = contained(root, args.checkpoint, must_exist=True)
        command += ["--checkpoint", str(checkpoint)]
    preflight(root)
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    def stop(signum, frame):
        stop_worker(child)
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        while child.poll() is None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                budget_check(root)
        return child.returncode
    finally:
        stop_worker(child)


if __name__ == "__main__":
    raise SystemExit(main())
