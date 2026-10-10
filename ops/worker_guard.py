"""Read-only preflight and a disk-budget guard around this project's CPU worker.

The launcher supplies CPU/memory/process/time limits through a transient user
systemd unit. This guard never discovers or terminates unrelated processes.
"""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


GIB = 1024 ** 3
MIN_FREE = 50 * GIB
PROJECT_BUDGET = 20 * GIB
CHECK_SECONDS = 5


def contained(root, raw, *, must_exist=False):
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"Path must be inside this project: {raw}")
    if must_exist and not candidate.is_file():
        raise ValueError(f"Required file is missing: {candidate}")
    return candidate


def validate(root, config, data, run_dir):
    config_path = contained(root, config, must_exist=True)
    data_path = contained(root, data, must_exist=True)
    run_path = contained(root, run_dir)
    if run_path.exists():
        if not run_path.is_dir() or any(run_path.iterdir()):
            raise ValueError(f"Run directory must be new or empty: {run_path}")
    return config_path, data_path, run_path


def project_bytes(root):
    total = 0
    # Never follow links to files or directories outside this dedicated project.
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in files:
            path = Path(base) / name
            try:
                stat = path.lstat()
                if not path.is_symlink():
                    # Count logical bytes conservatively, including sparse files.
                    total += stat.st_size
            except FileNotFoundError:
                # A concurrently committed/rotated file disappeared.
                continue
    return total


def budget_check(root):
    free = shutil.disk_usage(root).free
    size = project_bytes(root)
    if free <= MIN_FREE:
        raise RuntimeError(f"Free disk {free / GIB:.2f} GiB must exceed 50 GiB")
    if size >= PROJECT_BUDGET:
        raise RuntimeError(f"Project size {size / GIB:.2f} GiB reached 20 GiB budget")
    return free, size


def preflight(root):
    free, size = budget_check(root)
    print(f"Project: {root}")
    print(f"Disk free: {free / GIB:.2f} GiB; project: {size / GIB:.3f} / 20 GiB")
    devices = []
    pci = Path("/sys/bus/pci/devices")
    if pci.is_dir():
        for device in sorted(pci.iterdir()):
            try:
                klass = (device / "class").read_text().strip()
                if klass.startswith("0x03"):
                    vendor = (device / "vendor").read_text().strip()
                    devices.append(f"{device.name} vendor={vendor} class={klass}")
            except (FileNotFoundError, PermissionError):
                continue
        print("GPU/display PCI metadata (read only): " + ("; ".join(devices) or "none"))
    else:
        print("Linux PCI metadata unavailable on this host")
    print("Worker mode: CPU only; GPU visibility disabled by the launcher")


def stop_worker(child):
    if child.poll() is not None:
        return
    # start_new_session makes this exactly the launched worker's process group.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_worker(root, config, data, run_dir):
    if os.name != "posix":
        raise RuntimeError("The guarded deployment worker requires Linux/POSIX")
    config_path, data_path, run_path = validate(root, config, data, run_dir)
    preflight(root)
    command = [sys.executable, "-m", "english_world", "evolve", "--config", str(config_path),
               "--data", str(data_path), "--run-dir", str(run_path)]
    child = subprocess.Popen(command, cwd=root, start_new_session=True)
    previous_handlers = {}

    def interrupted(signum, _frame):
        stop_worker(child)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        while child.poll() is None:
            try:
                child.wait(timeout=CHECK_SECONDS)
            except subprocess.TimeoutExpired:
                budget_check(root)
        return child.returncode
    finally:
        stop_worker(child)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", metavar="PROJECT_ROOT")
    group.add_argument("--validate", nargs=4, metavar=("PROJECT_ROOT", "CONFIG", "DATA", "RUN_DIR"))
    group.add_argument("--run", metavar="PROJECT_ROOT")
    parser.add_argument("--config")
    parser.add_argument("--data")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    try:
        if args.preflight:
            preflight(Path(args.preflight).resolve(strict=True))
        elif args.validate:
            root, config, data, run_dir = args.validate
            validate(Path(root).resolve(strict=True), config, data, run_dir)
        else:
            if not all((args.config, args.data, args.run_dir)):
                parser.error("--run requires --config, --data and --run-dir")
            return run_worker(Path(args.run).resolve(strict=True), args.config, args.data, args.run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Worker guard: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
