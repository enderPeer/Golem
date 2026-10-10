"""Deterministic CPU evolutionary laboratory; no gradients or pretrained weights.

Selection sees TRAIN examples only. Validation selects a published champion, not
parents. Energy is a generation resource grant: base + positive task fitness times
reward scale. A birth consumes its entire parent's serialized byte count. A
tournament selects among parents that can afford a birth; this explicit resource
and selection model replaces spatial DNA-2 physics in English-world v1.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time

from .model import Genome, evaluate, reproduce, FORMAT

DEFAULTS = {'population_size': 64, 'generations': 300, 'seed': 1, 'max_order': 5,
            'max_contexts': 512, 'mutation_rate': .95, 'guided_probability': .7,
            'mutation_events': 4, 'eval_examples': 256, 'validation_examples': 512,
            'tournament_size': 4, 'elite_count': 2, 'checkpoint_every': 1,
            'checkpoint_seconds': 60., 'max_seconds': 1800., 'generated_chars': 48,
            'energy_base': 8192., 'energy_reward_scale': 16384., 'selection': 'tournament'}
SCHEMA = 'english-world-training-v1'


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _tuple_state(value):
    return tuple(_tuple_state(v) for v in value) if isinstance(value, list) else value


def _fingerprint(data: dict) -> str:
    # Canonical samples, rather than an unchecked caller-supplied label, determine
    # resume identity. The held-out test set is hashed but never evaluated here.
    fields = {k: data.get(k, []) for k in ('train', 'validation', 'test', 'dictionary')}
    if isinstance(fields['dictionary'], (set, frozenset)):
        fields['dictionary'] = sorted(fields['dictionary'])
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _configuration(config: dict) -> dict:
    c = dict(DEFAULTS)
    c.update(config)
    # Accept convenient CLI aliases without allowing them to change semantics.
    if 'population' in config and 'population_size' not in config:
        c['population_size'] = config['population']
    if 'training_examples' in config and 'eval_examples' not in config:
        c['eval_examples'] = config['training_examples']
    for key in ('population_size', 'generations', 'seed', 'max_order', 'max_contexts',
                'mutation_events', 'eval_examples', 'validation_examples', 'tournament_size',
                'elite_count', 'checkpoint_every', 'generated_chars'):
        try:
            c[key] = int(c[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{key} must be a finite integer') from exc
    for key in ('mutation_rate', 'guided_probability', 'checkpoint_seconds', 'max_seconds',
                'energy_base', 'energy_reward_scale'):
        try:
            c[key] = float(c[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{key} must be a finite number') from exc
        if not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
    if not 2 <= c['population_size'] <= 4096 or not 0 <= c['generations']:
        raise ValueError('population must be 2..4096 and generations nonnegative')
    if not 1 <= c['max_order'] <= 5 or not 1 <= c['max_contexts'] <= 4096:
        raise ValueError('invalid genome size bounds')
    if not 0 <= c['elite_count'] < c['population_size']:
        raise ValueError('elite_count must be smaller than population')
    if min(c['eval_examples'], c['validation_examples'], c['tournament_size'], c['checkpoint_every']) < 1:
        raise ValueError('evaluation, tournament and checkpoint sizes must be positive')
    if not 0 <= c['mutation_rate'] <= 1 or not 0 <= c['guided_probability'] <= 1:
        raise ValueError('mutation probabilities must be 0..1')
    if not 1 <= c['mutation_events'] <= 256 or not 0 <= c['generated_chars'] <= 1024:
        raise ValueError('invalid mutation or generation budget')
    if c['selection'] not in ('tournament', 'random'):
        raise ValueError('selection must be tournament or random')
    if c['max_seconds'] < 0 or c['checkpoint_seconds'] <= 0 or c['energy_base'] < 0 or c['energy_reward_scale'] < 0:
        raise ValueError('invalid time or energy limit')
    return c


class Trainer:
    def __init__(self, config: dict, data: dict, run_dir: str | Path, resume: bool = False):
        self.config = _configuration(config)
        self.data = data
        if not data.get('train') or not data.get('validation'):
            raise ValueError('nonempty train and validation splits are required')
        self.run_dir = Path(run_dir)
        if not resume and self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise ValueError('run directory is not empty; use resume or a new directory')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = _fingerprint(data)
        self.dictionary = frozenset(data.get('dictionary', []))
        self.validation = data['validation'][:self.config['validation_examples']]
        self.rng = random.Random(self.config['seed'])
        self.stop_requested = False
        self.started = time.monotonic()
        self.last_checkpoint = self.started
        self.elapsed_before = 0.
        self.history_cursors = {'metrics.jsonl': 0, 'ancestry.jsonl': 0}
        checkpoint = self.run_dir / 'checkpoint.json'
        if resume:
            if not checkpoint.exists():
                raise ValueError(f'checkpoint not found: {checkpoint}')
            self.state = json.loads(checkpoint.read_text(encoding='utf-8'))
            if self.state.get('schema') != SCHEMA or self.state['data_fingerprint'] != self.fingerprint:
                raise ValueError('checkpoint schema or dataset fingerprint mismatch')
            allowed_changes = {'generations', 'max_seconds', 'checkpoint_seconds', 'checkpoint_every'}
            for key in DEFAULTS:
                if key not in allowed_changes and self.state['config'].get(key) != self.config[key]:
                    raise ValueError(f'cannot change {key} while resuming')
            self.rng.setstate(_tuple_state(self.state['rng_state']))
            self.elapsed_before = self.state.get('elapsed_seconds', 0.)
            self.state['config'] = self.config
            self.state['status'] = 'running'
            self._write_history(rewrite=True)
        else:
            if checkpoint.exists():
                raise ValueError('run already has a checkpoint; use resume or a new directory')
            population = []
            for i in range(self.config['population_size']):
                genome = Genome(max_order=self.rng.randint(1, self.config['max_order']),
                                max_contexts=self.config['max_contexts'])
                population.append({'id': i, 'parent_id': None, 'generation_born': 0,
                                   'genome': genome.to_dict(), 'energy': 0., 'metrics': None})
            self.state = {'schema': SCHEMA, 'genome_schema': FORMAT, 'config': self.config,
                          'data_fingerprint': self.fingerprint, 'generation': 0, 'stage': 'start',
                          'population': population, 'next_id': len(population), 'pending': None,
                          'history': [], 'ancestry': [], 'champion': None, 'baseline': None,
                          'status': 'running', 'total_births': 0, 'faithful_births': 0,
                          'guided_events': 0, 'unguided_events': 0, 'inflow_founders': 0,
                          'bytes_copied': 0, 'elapsed_seconds': 0.}
            _atomic_json(self.run_dir / 'config.json', {'config': self.config,
                         'data_fingerprint': self.fingerprint, 'manifest': data.get('manifest', {})})
        self.checkpoint()

    def checkpoint(self):
        self.state['rng_state'] = self.rng.getstate()
        self.state['elapsed_seconds'] = self.elapsed_before + time.monotonic() - self.started
        _atomic_json(self.run_dir / 'checkpoint.json', self.state)
        if self.state['champion']:
            _atomic_json(self.run_dir / 'champion.json', self.state['champion'])
        self.status()
        self.last_checkpoint = time.monotonic()

    def status(self):
        pending = self.state.get('pending') or {}
        latest = self.state['history'][-1] if self.state['history'] else None
        champion = self.state.get('champion')
        status = {'schema': SCHEMA, 'status': self.state['status'], 'generation': self.state['generation'],
                  'target_generations': self.config['generations'], 'stage': self.state['stage'],
                  'evaluated_parents': pending.get('eval_cursor', 0),
                  'children_in_generation': len(pending.get('children', [])),
                  'population_size': self.config['population_size'],
                  'total_births': self.state['total_births'], 'faithful_births': self.state['faithful_births'],
                  'guided_events': self.state['guided_events'], 'unguided_events': self.state['unguided_events'],
                  'bytes_copied': self.state['bytes_copied'],
                  'elapsed_seconds': self.elapsed_before + time.monotonic() - self.started,
                  'latest': latest, 'champion_metrics': champion['metrics'] if champion else None}
        _atomic_json(self.run_dir / 'status.json', status)

    def _write_history(self, rewrite=False):
        # Append during a run. On resume, the authoritative checkpoint truncates
        # any records emitted after its last committed state.
        for name, records in (('metrics.jsonl', self.state['history']), ('ancestry.jsonl', self.state['ancestry'])):
            start = 0 if rewrite else self.history_cursors[name]
            if start == len(records) and not rewrite:
                continue
            path = self.run_dir / (name + '.tmp' if rewrite else name)
            with path.open('w' if rewrite else 'a', encoding='utf-8') as f:
                for record in records[start:]:
                    f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            if rewrite:
                os.replace(path, self.run_dir / name)
            self.history_cursors[name] = len(records)

    def _evaluate(self, genome: Genome, examples: list[dict]) -> dict:
        return evaluate(genome, examples, self.dictionary, generated_chars=self.config['generated_chars'])

    def _start_generation(self):
        count = min(self.config['eval_examples'], len(self.data['train']))
        indices = self.rng.sample(range(len(self.data['train'])), count)
        self.state['pending'] = {'batch_indices': indices, 'eval_cursor': 0, 'children': [],
                                 'selected_parent_fitness': [], 'failed_parents': [],
                                 'generation_guided': 0, 'generation_unguided': 0}
        self.state['stage'] = 'evaluate'

    def _step(self):
        c, state = self.config, self.state
        if state['stage'] == 'start':
            self._start_generation()
            return
        pending = state['pending']
        batch = [self.data['train'][i] for i in pending['batch_indices']]
        if state['stage'] == 'evaluate':
            cursor = pending['eval_cursor']
            if cursor < len(state['population']):
                individual = state['population'][cursor]
                genome = Genome.from_dict(individual['genome'])
                individual['metrics'] = self._evaluate(genome, batch)
                individual['energy'] = c['energy_base'] + c['energy_reward_scale'] * max(0., individual['metrics']['fitness'])
                pending['eval_cursor'] += 1
                return
            if state['baseline'] is None:
                state['baseline'] = self._evaluate(Genome(), self.validation)
            state['stage'] = 'reproduce'
            return
        if state['stage'] == 'reproduce':
            target_children = c['population_size'] - c['elite_count']
            if len(pending['children']) >= target_children:
                state['stage'] = 'finish'
                return
            eligible = [p for p in state['population'] if p['id'] not in pending['failed_parents']
                        and p['energy'] >= p['metrics']['genome_bytes'] + 1]
            replication = None
            if eligible:
                if c['selection'] == 'random':
                    parent = self.rng.choice(eligible)
                else:
                    tournament = self.rng.sample(eligible, min(c['tournament_size'], len(eligible)))
                    parent = max(tournament, key=lambda p: (p['metrics']['fitness'], -p['id']))
                try:
                    child_genome, replication = reproduce(Genome.from_dict(parent['genome']),
                        mutation_rate=c['mutation_rate'], rng=self.rng, train_examples=self.data['train'],
                        guided_probability=c['guided_probability'], mutation_events=c['mutation_events'])
                except ValueError:
                    pending['failed_parents'].append(parent['id'])
                    return
                parent['energy'] -= replication['offspring_copy_cost']
                parent_id = parent['id']
                parent_hash = Genome.from_dict(parent['genome']).digest()
                pending['selected_parent_fitness'].append(parent['metrics']['fitness'])
                state['total_births'] += 1
                state['faithful_births'] += int(replication['faithful_after_mutation'])
                # The VM verified two births; only one is admitted to the population.
                state['bytes_copied'] += replication['bytes_copied']
                for key in ('guided_events', 'unguided_events'):
                    state[key] += replication[key]
                    pending['generation_' + ('guided' if key == 'guided_events' else 'unguided')] += replication[key]
            else:
                # Explicit resource-starvation inflow, never an implicit learned model.
                child_genome = Genome(max_order=self.rng.randint(1, c['max_order']), max_contexts=c['max_contexts'])
                parent_id = parent_hash = None
                state['inflow_founders'] += 1
            individual = {'id': state['next_id'], 'parent_id': parent_id,
                          'generation_born': state['generation'] + 1, 'genome': child_genome.to_dict(),
                          'metrics': self._evaluate(child_genome, batch), 'energy': 0.}
            state['next_id'] += 1
            pending['children'].append(individual)
            state['ancestry'].append({'id': individual['id'], 'parent_id': parent_id,
                'generation': individual['generation_born'], 'parent_hash': parent_hash,
                'genome_hash': child_genome.digest(), 'replication': replication})
            return
        if state['stage'] == 'finish':
            old_population = state['population']
            elite = sorted(old_population, key=lambda p: (-p['metrics']['fitness'], p['id']))[:c['elite_count']]
            state['population'] = elite + pending['children']
            best = max(state['population'], key=lambda p: (p['metrics']['fitness'], -p['id']))
            genome = Genome.from_dict(best['genome'])
            validation = self._evaluate(genome, self.validation)
            if state['champion'] is None or validation['fitness'] > state['champion']['metrics']['fitness']:
                state['champion'] = {'schema': 'english-world-champion-v1', 'genome': genome.to_dict(),
                     'genome_hash': genome.digest(), 'generation': state['generation'] + 1,
                     'individual_id': best['id'], 'metrics': validation, 'evaluation_split': 'validation',
                     'data_fingerprint': self.fingerprint,
                     'capability': 'experimental character continuation; no general dialogue capability established'}
                _atomic_json(self.run_dir / 'champion.json', state['champion'])
            old_mean = sum(p['metrics']['fitness'] for p in old_population) / len(old_population)
            selected = pending['selected_parent_fitness']
            selected_mean = sum(selected) / len(selected) if selected else old_mean
            state['generation'] += 1
            state['history'].append({'generation': state['generation'],
                'mean_fitness': sum(p['metrics']['fitness'] for p in state['population']) / len(state['population']),
                'best_training_fitness': best['metrics']['fitness'], 'best_training_loss': best['metrics']['loss'],
                'candidate_validation': validation, 'champion_validation': state['champion']['metrics'],
                'uniform_validation': state['baseline'], 'selected_parent_mean_fitness': selected_mean,
                'parent_population_mean_fitness': old_mean, 'selection_differential': selected_mean - old_mean,
                'births': len(selected), 'guided_events': pending['generation_guided'],
                'unguided_events': pending['generation_unguided'],
                'unique_genomes': len({Genome.from_dict(p['genome']).digest() for p in state['population']}),
                'inflow_founders_total': state['inflow_founders']})
            state['pending'] = None
            state['stage'] = 'start'
            self._write_history()
            self.status()
            if state['generation'] % c['checkpoint_every'] == 0:
                self.checkpoint()

    def run(self) -> dict:
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            def request_stop(signum, frame):
                self.stop_requested = True
            for name in ('SIGINT', 'SIGTERM'):
                sig = getattr(signal, name, None)
                if sig is not None:
                    old_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, request_stop)
        try:
            while self.state['generation'] < self.config['generations']:
                elapsed = time.monotonic() - self.started
                if self.stop_requested or (self.config['max_seconds'] and elapsed >= self.config['max_seconds']):
                    self.state['status'] = 'interrupted' if self.stop_requested else 'time_limit'
                    break
                self._step()
                if time.monotonic() - self.last_checkpoint >= self.config['checkpoint_seconds']:
                    self.checkpoint()
            else:
                self.state['status'] = 'complete'
        except KeyboardInterrupt:
            self.state['status'] = 'interrupted'
        finally:
            self.checkpoint()
            self._write_history()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        champion = self.state['champion']
        return {'status': self.state['status'], 'generation': self.state['generation'],
                'generations': self.state['generation'], 'elapsed_seconds': self.state['elapsed_seconds'],
                'total_births': self.state['total_births'], 'guided_events': self.state['guided_events'],
                'unguided_events': self.state['unguided_events'], 'data_fingerprint': self.fingerprint,
                'run_dir': str(self.run_dir.resolve()), 'checkpoint': str((self.run_dir / 'checkpoint.json').resolve()),
                'champion': str((self.run_dir / 'champion.json').resolve()) if champion else None,
                'champion_metrics': champion['metrics'] if champion else None,
                'baseline_metrics': self.state['baseline']}


def train(config: dict, data: dict, run_dir: str | Path, resume: bool = False) -> dict:
    return Trainer(config, data, run_dir, resume=resume).run()


def benchmark(config: dict, data: dict, run_dir: str | Path) -> dict:
    cfg = dict(config)
    cfg['generations'] = min(3, int(cfg.get('generations', 3)))
    result = train(cfg, data, run_dir)
    result['generations_per_second'] = result['generation'] / max(result['elapsed_seconds'], 1e-9)
    return result
