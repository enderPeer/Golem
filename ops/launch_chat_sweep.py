"""Launch the already-reviewed jobs for this host; never discovers new hosts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='runs/chat-manifest.json')
    parser.add_argument('--kind', choices=('cpu', 'gpu', 'all'), default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get('schema') != 'golem-chat-cluster-sweep-v1':
        raise ValueError('Unsupported reviewed allocation manifest')
    jobs = [job for job in manifest['jobs'] if job['host'] == socket.gethostname()]
    if not jobs or any(Path(job['root']).resolve() != root for job in jobs):
        raise ValueError('Manifest host/project root mismatch')
    if sum(job['threads'] for job in jobs) > os.cpu_count():
        raise ValueError('Allocation exceeds this host logical CPU count')
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    if sum(job['memory_mib'] for job in jobs) * 2**20 > memory['MemAvailable'] - 2 * 2**30:
        raise ValueError('Aggregate maximum memory would leave less than 2 GiB available; recheck allocation')
    for job in jobs:
        if args.kind != 'all' and job['kind'] != args.kind:
            continue
        receipt = root / 'runs/launches' / (job['id'] + '.unit')
        if receipt.exists():
            print(json.dumps({'id': job['id'], 'already_launched': receipt.read_text().strip()}), flush=True)
            continue
        for field in ('config', 'run_dir'):
            path = (root / job[field]).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError('A launch path leaves the dedicated project')
        env = dict(os.environ)
        env.update(GOLEM_PYTHON=job['python'], GOLEM_THREADS=str(job['threads']),
                   GOLEM_MEMORY_MIB=str(job['memory_mib']),
                   GOLEM_MIN_FREE_GIB=str(job['minimum_free_gib']),
                   CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='')
        if job['kind'] == 'gpu':
            # Only AMD devices were admitted by this recorded cluster inventory.
            env['HIP_VISIBLE_DEVICES'] = str(job['physical_gpu'])
        subprocess.run(['bash', 'ops/run_chat.sh', job['config'], job['run_dir']],
                       cwd=root, env=env, check=True)
        print(json.dumps({'id': job['id'], 'launched': True}), flush=True)


if __name__ == '__main__':
    main()
