"""Launch the already-reviewed jobs for this host; never discovers new hosts."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess


def launch_modes(manifest):
    if manifest.get('schema') not in ('golem-chat-cluster-sweep-v1', 'golem-semantic-world-sweep-v1'):
        raise ValueError('Unsupported reviewed allocation manifest')
    world = manifest['schema'] == 'golem-semantic-world-sweep-v1'
    return world, world and manifest.get('mode') == 'supervised_until_deadline'


def device_visibility(job):
    """Expose only the GPU explicitly admitted for this job, if any."""
    visibility = {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': ''}
    if job['kind'] not in ('cpu', 'gpu'):
        raise ValueError('Invalid job device kind')
    if job['kind'] == 'gpu':
        backend = job.get('gpu_backend', 'hip')
        if backend not in ('cuda', 'hip'):
            raise ValueError('gpu_backend must be cuda or hip')
        physical_gpu = job.get('physical_gpu')
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError('physical_gpu must be a nonnegative integer for GPU jobs')
        device = str(physical_gpu)
        if 'gpu_uuid' in job:
            if backend != 'cuda':
                raise ValueError('gpu_uuid is allowed only for CUDA GPU jobs')
            uuid = job['gpu_uuid']
            if not isinstance(uuid, str) or not re.fullmatch(
                    r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', uuid):
                raise ValueError('gpu_uuid must be a complete NVIDIA GPU UUID')
            device = uuid
        visibility['CUDA_VISIBLE_DEVICES' if backend == 'cuda' else 'HIP_VISIBLE_DEVICES'] = device
    elif 'gpu_uuid' in job:
        raise ValueError('gpu_uuid is allowed only for CUDA GPU jobs')
    return visibility


def job_environment(job, ambient=None):
    """Replace ambient visibility; retain unrelated runtime environment fields."""
    env = dict(os.environ if ambient is None else ambient)
    # These lower-level AMD filters can remap the reviewed physical index.
    for key in ('ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env.update(GOLEM_PYTHON=job['python'], GOLEM_THREADS=str(job['threads']),
               GOLEM_MEMORY_MIB=str(job['memory_mib']),
               GOLEM_MIN_FREE_GIB=str(job['minimum_free_gib']),
               GOLEM_STAGE_SECONDS=str(job.get('stage_seconds', 1800)))
    env.update(device_visibility(job))
    return env


def inspect_receipt(root, job, receipt):
    """A receipt is an identity record, not proof its service remains healthy."""
    unit = receipt.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'golem-[A-Za-z0-9_.@-]+\.service', unit):
        raise ValueError(f"Invalid service receipt for {job['id']}: {receipt}")
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=LoadState,ActiveState,SubState,Result,ExecMainStatus'],
                            capture_output=True, text=True, timeout=15, check=False)
    service = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    status_path = root / job['run_dir'] / 'status.json'
    if not status_path.is_file():
        status_path = root / job['run_dir'] / 'report.json'
    recorded_status = None
    if status_path.is_file():
        if not status_path.resolve().is_relative_to(root):
            raise ValueError('A service status path leaves the dedicated project')
        recorded_status = json.loads(status_path.read_text(encoding='utf-8')).get('status')
    live = (result.returncode == 0 and service.get('LoadState') == 'loaded' and
            service.get('ActiveState') in ('active', 'activating', 'reloading') and
            service.get('SubState') not in ('dead', 'failed', 'exited') and
            service.get('Result', 'success') == 'success')
    if not live or recorded_status not in (None, 'running', 'pending', 'starting'):
        raise RuntimeError(
            f"Existing receipt for {job['id']} is not a live training run: unit={unit}, "
            f"active={service.get('ActiveState', 'unknown')}, sub={service.get('SubState', 'unknown')}, "
            f"result={service.get('Result', 'unknown')}, report_status={recorded_status!r}. "
            'No automatic restart or duplicate launch was attempted. Inspect the existing run and explicitly choose recovery or a fresh run.')
    return {'id': job['id'], 'already_launched': unit, 'service_state': service['ActiveState'],
            'report_status': recorded_status, 'receipt_verified': True}


def plan_launch(manifest, root, hostname, kind='cpu', *, receipt_inspector=None):
    root = Path(root).resolve()
    world, long_run = launch_modes(manifest)
    jobs = [job for job in manifest['jobs'] if job['host'] == hostname]
    if not jobs or any(Path(job['root']).resolve() != root for job in jobs):
        raise ValueError('Manifest host/project root mismatch')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Manifest contains duplicate job identities')
    pending, existing = [], []
    inspect = receipt_inspector or inspect_receipt
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', job['id']):
            raise ValueError('Invalid job identity')
        device_visibility(job)
        if type(job['threads']) is not int or not 1 <= job['threads'] <= 64:
            raise ValueError('Invalid job thread quota')
        if type(job['memory_mib']) is not int or not 512 <= job['memory_mib'] <= 8192:
            raise ValueError('Invalid job memory quota')
        stage_seconds = job.get('stage_seconds', 1800)
        if type(stage_seconds) is not int or not 30 <= stage_seconds <= 7200:
            raise ValueError('stage_seconds must be an integer in 30..7200')
        for field in ('config', 'run_dir'):
            path = (root / job[field]).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError('A launch path leaves the dedicated project')
        if kind != 'all' and job['kind'] != kind:
            continue
        receipts = 'runs/world-long-launches' if long_run else 'runs/world-launches' if world else 'runs/launches'
        receipt = root / receipts / (job['id'] + '.unit')
        if receipt.is_symlink():
            raise ValueError('Service receipts must not be symlinks')
        if receipt.exists():
            existing.append(inspect(root, job, receipt))
        else:
            run_dir = root / job['run_dir']
            if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
                raise RuntimeError(f"Unreceipted existing run for {job['id']}; explicit recovery is required before any duplicate launch")
            pending.append(job)
    return jobs, pending, existing


def check_admission(host_jobs, pending_jobs, logical_cpus, available_memory):
    if sum(job['threads'] for job in host_jobs) > logical_cpus:
        raise ValueError('Allocation exceeds this host logical CPU count')
    # MemAvailable already reflects existing workers; count only new workers of
    # the selected device kind, not the full allocation for a second time.
    required = sum(job['memory_mib'] for job in pending_jobs) * 2**20
    if pending_jobs and required > available_memory - 2 * 2**30:
        raise ValueError('New selected jobs would leave less than 2 GiB available; recheck allocation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='runs/chat-manifest.json')
    parser.add_argument('--kind', choices=('cpu', 'gpu', 'all'), default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    world, long_run = launch_modes(manifest)
    jobs, pending, existing = plan_launch(manifest, root, socket.gethostname(), args.kind)
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    check_admission(jobs, pending, os.cpu_count(), memory['MemAvailable'])
    for item in existing:
        print(json.dumps(item), flush=True)
    for job in pending:
        env = job_environment(job)
        command = ['bash', 'ops/run_world_long.sh' if long_run else 'ops/run_world.sh' if world else 'ops/run_chat.sh',
                   job['config'], job['run_dir']]
        if long_run:
            command.append(manifest['deadline_utc'])
        subprocess.run(command, cwd=root, env=env, check=True)
        print(json.dumps({'id': job['id'], 'launched': True}), flush=True)


if __name__ == '__main__':
    main()
