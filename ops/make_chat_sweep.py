"""Materialize a reviewed, finite cluster sweep; this command launches nothing.

The host allocation is the explicitly checked 2026-10-10 allocation. Recheck
current workloads before reusing it. Occupied GPUs are deliberately absent.
"""
import argparse
import json
from pathlib import Path
import tomllib


ALLOCATIONS = {
    'specht32': {'root': '/home/ender/golem', 'cpu_jobs': 4, 'gpu_ids': [0, 1],
                 'cpu_python': '/home/ender/golem/.venv-neural/bin/python',
                 'gpu_python': '/home/ender/golem/.venv-rocm/bin/python', 'cpu_memory_mib': 2048},
    'falke64': {'root': '/home/ender/golem', 'cpu_jobs': 4, 'gpu_ids': [0],
                'cpu_python': '/home/ender/homunculi/.venv/bin/python',
                'gpu_python': '/home/ender/golem/.venv-rocm/bin/python', 'cpu_memory_mib': 1536},
    'adler40': {'root': '/home/ender/golem', 'cpu_jobs': 10, 'gpu_ids': [],
                'cpu_python': '/home/ender/homunculi/.venv/bin/python', 'cpu_memory_mib': 2048},
    'knecht24': {'root': '/tmp/golem-chat-20261010', 'cpu_jobs': 9, 'gpu_ids': [],
                 'cpu_python': '/home/ender/homunculi/.venv/bin/python', 'cpu_memory_mib': 2048},
}


def toml_text(config):
    lines = []
    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f'{key} = {json.dumps(value)}')
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f'\n[{section}]')
            lines.extend(f'{key} = {json.dumps(value)}' for key, value in values.items())
    return '\n'.join(lines) + '\n'


def make_sweep(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; existing sweep records are immutable')
    base = tomllib.loads((root / 'configs/neural-chat.toml').read_text())
    jobs = []
    serial = 0
    for host, allocation in ALLOCATIONS.items():
        variants = [('cpu', None, i) for i in range(allocation['cpu_jobs'])]
        variants += [('gpu', device, replica) for device in allocation['gpu_ids'] for replica in range(2)]
        for kind, physical, index in variants:
            identity = f'{host}-{kind}-{index}' if kind == 'cpu' else f'{host}-gpu{physical}-{index}'
            config = {**base, 'model': dict(base['model'])}
            config.update(seed=61 + serial * 101, learning_rate=(.0005, .0015, .003)[serial % 3],
                          weight_mutation_probability=(.25, .5, .8)[serial % 3],
                          weight_noise_rms_fraction=(.001, .003, .01)[serial % 3],
                          crossover_probability=(0., .25, .5)[serial % 3],
                          validation_cases=26,
                          init_checkpoint='runs/seeds/chat-' + ('language' if serial % 2 == 0 else 'relations') + '.pt')
            config['model']['dropout'] = (.0, .1)[serial % 2]
            config.update(device='cpu' if kind == 'cpu' else 'cuda:0', threads=2 if kind == 'cpu' else 1,
                          batch_size=8 if kind == 'cpu' else 32,
                          max_total_updates=2400 if kind == 'cpu' else 800)
            if host == 'falke64' and kind == 'gpu':
                config['strategies'] = ['evolution', 'ordinary', 'independent']
            directory = output / host
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / (identity + '.toml')
            config_path.write_text(toml_text(config), encoding='utf-8')
            # Parse our emitted TOML before creating any launch specification.
            if tomllib.loads(config_path.read_text()) != config:
                raise ValueError('Config serialization did not round-trip')
            jobs.append({'id': identity, 'host': host, 'root': allocation['root'], 'kind': kind,
                'physical_gpu': physical, 'python': allocation[kind + '_python'],
                'threads': config['threads'], 'memory_mib': allocation['cpu_memory_mib'] if kind == 'cpu' else 3072,
                'minimum_free_gib': 2 if allocation['root'].startswith('/tmp/') else 50,
                'config': 'runs/chat-configs/' + identity + '.toml',
                'run_dir': 'runs/chat-sweep/' + identity, 'seed': config['seed'],
                'local_config': str(config_path.resolve())})
            serial += 1
    manifest = {'schema': 'golem-chat-cluster-sweep-v1', 'jobs': jobs,
        'selection': 'Validation-generated score and validation NLL only; test is never used to pick an island.',
        'budget': 'Matched comparisons within an island; hardware, batch sizes and budgets differ across islands.',
        'excluded_gpus': 'Falke occupied R9700; Adler active DNA/loaded chat GPUs; all Knecht GPUs running DNA.',
        'cpu_policy': 'Nice 19, explicit thread quotas, no termination of unrelated workloads.',
        'runtime_note': 'Borrowed homunculi runtimes are used read-only with bytecode disabled; their files are not changed.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = make_sweep(args.output)
    print(json.dumps({'jobs': len(manifest['jobs']), 'output': args.output}, indent=2))
