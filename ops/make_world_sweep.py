"""Create finite, independent semantic-world experiments after idle-host review."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.make_chat_sweep import ALLOCATIONS, toml_text


def make_world_sweep(output, deadline=None):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh sweep directory')
    base = tomllib.loads((root / 'configs/neural-world.toml').read_text())
    jobs = []
    serial = 0
    for host, allocation in ALLOCATIONS.items():
        cpu_jobs = 3 if deadline and host == 'falke64' else allocation['cpu_jobs']
        workers = [('cpu', None, i) for i in range(cpu_jobs)]
        workers += [('gpu', device, i) for device in allocation['gpu_ids'] for i in range(2)]
        for kind, physical, index in workers:
            identity = f'{host}-world-cpu-{index}' if kind == 'cpu' else f'{host}-world-gpu{physical}-{index}'
            cfg = {**base, 'model': dict(base['model'])}
            cfg.update(seed=97 + serial * 103,
                learning_rate=(.0003, .0007, .0015)[serial % 3],
                intent_loss_weight=(.5, 1., 2.)[(serial // 3) % 3],
                first_response_weight=(2., 4., 6.)[(serial // 2) % 3],
                dropout=(0., .1)[serial % 2],
                device='cpu' if kind == 'cpu' else 'cuda:0', threads=2 if kind == 'cpu' else 1,
                batch_size=8 if kind == 'cpu' else 16,
                max_total_updates=1600 if kind == 'cpu' else 800,
                init_checkpoint='runs/seeds/world-' + ('chat' if serial % 2 else 'language') + '-base.pt')
            cfg['model']['dropout'] = cfg['dropout']
            if deadline:
                cfg['evaluate_test_on_complete'] = False
                if host == 'falke64' and kind == 'cpu':
                    cfg['threads'] = 3
            if serial % 2:
                cfg['strategies'] = ['evolution', 'ordinary']
            folder = output / host
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (identity + '.toml')
            path.write_text(toml_text(cfg), encoding='utf-8')
            if tomllib.loads(path.read_text()) != cfg:
                raise ValueError('World config serialization mismatch')
            jobs.append({'id': identity, 'host': host, 'root': allocation['root'], 'kind': kind,
                'physical_gpu': physical, 'python': allocation[kind + '_python'], 'threads': cfg['threads'],
                'memory_mib': allocation['cpu_memory_mib'] if kind == 'cpu' else (4096 if deadline else 3072),
                'minimum_free_gib': 2 if allocation['root'].startswith('/tmp/') else 50,
                'config': 'runs/world-long-configs/' + identity + '.toml' if deadline else 'runs/world-configs/' + identity + '.toml',
                'run_dir': ('runs/world-long/' if deadline else 'runs/world-sweep/') + identity,
                'local_config': str(path.resolve()), 'seed': cfg['seed']})
            serial += 1
    result = {'schema': 'golem-semantic-world-sweep-v1', 'jobs': jobs,
        'world_policy': 'Independent scenarios and user facts; peers never become authoritative teachers.',
        'learning': 'Joint request-meaning and final-reply losses, verified USER-only memory, free generated rollout selection.',
        'budgets': 'Equal optimizer updates per strategy within an island; token lengths and hardware costs can differ.',
        'allocation_note': 'Inherited explicit host limits, rechecked idle on 2026-10-11 Europe/Berlin; occupied GPUs excluded.'}
    if deadline:
        result.update(deadline_utc=deadline, mode='supervised_until_deadline',
                      resource_note='GPU host memory measured up to 2.93 GiB during probes; allow 4 GiB per GPU worker. Falke uses three 3-thread CPU islands to fit memory headroom.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', help='Explicit UTC deadline enables bounded continuous stages')
    args = parser.parse_args()
    result = make_world_sweep(args.output, args.deadline)
    print(json.dumps({'world_experiments': len(result['jobs']), 'output': args.output}))
