"""Bounded neural experiments with an operation-matched four-member PBT control.

These are gradient-trained tensor models. Population evolution copies complete
model/AdamW state and mutates hyperparameters; it does not execute DNA-2 COPY.
Training uses only train documents. Validation selects checkpoints and parents;
test documents are evaluated only after both arms finish. Equal update/token
budgets are not a claim of equal total CPU time: all measured overhead is reported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import threading
import time

import torch
from torch import nn

from .neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint
from .neural_data import prepare_documents, sample_batch, make_recall_documents, SequenceDocument


SCHEMA = 'golem-neural-comparison-v1'
ARM_NAMES = ('ordinary', 'evolution')


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _hash_weights(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _config(config):
    defaults = {'max_total_updates': 20, 'population_size': 4, 'rounds': 1,
                'batch_size': 4, 'sequence_length': 64, 'seed': 7, 'threads': 2,
                'learning_rate': .003, 'weight_decay': .01, 'gradient_clip': 1.,
                'validation_batches': 2, 'test_batches': 4,
                'checkpoint_every': 10, 'max_seconds': 300., 'budget_mode': 'updates',
                'model': {'embedding_dim': 32, 'hidden_size': 64, 'num_layers': 1, 'dropout': .1}}
    result = copy.deepcopy(defaults)
    result.update(copy.deepcopy(config))
    result['model'] = {**defaults['model'], **config.get('model', {})}
    for name, low, high in (('max_total_updates', 4, 1000000), ('population_size', 4, 4),
                            ('rounds', 1, 10000), ('batch_size', 1, 128),
                            ('sequence_length', 2, 4096), ('threads', 1, 2),
                            ('validation_batches', 1, 1024), ('test_batches', 1, 1024),
                            ('checkpoint_every', 1, 1000000)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    for name in ('learning_rate', 'gradient_clip', 'max_seconds', 'weight_decay'):
        result[name] = float(result[name])
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if result['learning_rate'] == 0 or result['gradient_clip'] == 0:
        raise ValueError('learning rate and gradient clip must be positive')
    if result['max_total_updates'] % (4 * result['rounds']):
        raise ValueError('max_total_updates must be divisible by 4 * rounds')
    if result['budget_mode'] != 'updates':
        raise ValueError('Only operation-matched updates mode is implemented; total CPU is measured, not matched')
    NeuralConfig(**result['model'])
    return result


def _documents(data):
    documents = {split: prepare_documents(data, split) for split in ('train', 'validation', 'test')}
    if any(not documents[split] for split in documents):
        raise ValueError('nonempty document-disjoint train, validation and test splits are required')
    digest = hashlib.sha256()
    for split, docs in documents.items():
        digest.update(split.encode())
        for doc in docs:
            digest.update(struct.pack('<Q', len(doc.tokens)))
            digest.update(struct.pack('<' + 'H' * len(doc.tokens), *doc.tokens))
            digest.update(bytes(doc.loss_mask))
            digest.update(doc.kind.encode())
    return documents, digest.hexdigest()


def _loss(model, batch):
    logits, _ = model(batch['inputs'])
    mask = batch['loss_mask'].reshape(-1)
    count = int(mask.sum())
    if count == 0:
        raise ValueError('batch has no supervised target tokens')
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask],
                                       batch['targets'].reshape(-1)[mask])
    return loss, count, logits


def _evaluate(model, docs, config, batches, seed):
    model.eval()
    rng = random.Random(seed)
    nll = tokens = correct = processed = 0
    with torch.no_grad():
        for _ in range(batches):
            batch = sample_batch(docs, config['batch_size'], config['sequence_length'], rng)
            loss, count, logits = _loss(model, batch)
            mask = batch['loss_mask']
            nll += float(loss) * count
            tokens += count
            correct += int((logits.argmax(-1)[mask] == batch['targets'][mask]).sum())
            processed += batch['inputs'].numel()
    return {'loss': nll / tokens, 'bits_per_token': nll / tokens / math.log(2),
            'accuracy': correct / tokens, 'supervised_tokens': tokens,
            'processed_tokens': processed, 'forward_batches': batches}


def _new_arm(name, config):
    candidates = []
    count = 1 if name == 'ordinary' else 4
    for index in range(count):
        torch.manual_seed(config['seed'] + index)
        model = ByteGRULM(NeuralConfig(**config['model']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        candidates.append({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'learning_rate': config['learning_rate'], 'dropout': config['model']['dropout'],
                           'ancestral_updates': 0, 'initial_seed': config['seed'] + index})
    torch.manual_seed(config['seed'] + 9000)
    return {'candidates': candidates, 'updates': 0, 'next_batch_index': 0,
            'processed_training_tokens': 0, 'supervised_training_tokens': 0,
            'validation_forward_batches': 0, 'validation_processed_tokens': 0,
            'validation_supervised_tokens': 0, 'validation_history': [],
            'round_losses': [], 'inheritance': [], 'best': None, 'test': None,
            'torch_rng_state': torch.get_rng_state(),
            'evolution_rng_state': random.Random(config['seed'] + 7000).getstate(),
            'cpu_seconds': 0., 'wall_seconds': 0., 'status': 'pending'}


def _restore(arm, config):
    models, optimizers = [], []
    for candidate in arm['candidates']:
        model_config = {**config['model'], 'dropout': candidate['dropout']}
        model = ByteGRULM(NeuralConfig(**model_config))
        model.load_state_dict(candidate['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'], weight_decay=config['weight_decay'])
        optimizer.load_state_dict(candidate['optimizer'])
        models.append(model)
        optimizers.append(optimizer)
    torch.set_rng_state(arm['torch_rng_state'])
    return models, optimizers


def _sync(arm, models, optimizers):
    for candidate, model, optimizer in zip(arm['candidates'], models, optimizers):
        candidate['model'] = model.state_dict()
        candidate['optimizer'] = optimizer.state_dict()
    arm['torch_rng_state'] = torch.get_rng_state()


def _export_best(arm, config, path):
    best = arm['best']
    if best is None:
        return
    with torch.random.fork_rng(devices=[]):
        model = ByteGRULM(NeuralConfig(**{**config['model'], 'dropout': best['dropout']}))
        model.load_state_dict(best['model'])
        save_neural_checkpoint(path, model, metadata={'training_schema': SCHEMA,
            'selection_split': 'validation', 'validation': best['validation'],
            'global_updates': best['global_updates'], 'candidate': best['candidate'],
            'weight_sha256': best['weight_sha256'], 'supervised_gradient_training': True})


def _status(state, run_dir):
    payload = {'schema': SCHEMA, 'status': state['status'], 'active_arm': state['active_arm'],
               'data_fingerprint': state['data_fingerprint'], 'arms': {}}
    for name, arm in state['arms'].items():
        payload['arms'][name] = {key: arm[key] for key in ('status', 'updates', 'processed_training_tokens',
             'supervised_training_tokens', 'validation_forward_batches', 'cpu_seconds', 'wall_seconds')}
        payload['arms'][name]['best_validation'] = arm['best']['validation'] if arm['best'] else None
    if 'error' in state:
        payload['error'] = state['error']
    _json(Path(run_dir) / 'status.json', payload)


def _persist(state, run_dir):
    cpu, wall = time.process_time(), time.monotonic()
    _save(Path(run_dir) / 'training.pt', state)
    _status(state, run_dir)
    # The following checkpoint carries this write's timing. Final report includes
    # it immediately; timings deliberately do not participate in replay identity.
    active = state['active_arm']
    if active in state['arms']:
        state['arms'][active]['cpu_seconds'] += time.process_time() - cpu
        state['arms'][active]['wall_seconds'] += time.monotonic() - wall


def _step_arm(name, arm, models, optimizers, documents, config):
    interval = config['max_total_updates'] // (4 * config['rounds'])
    block = arm['updates'] // interval
    index = 0 if name == 'ordinary' else block % 4
    model, optimizer = models[index], optimizers[index]
    # Exactly the same global update index produces the same training batch in
    # both arms, including padding and supervision masks. No inherited sampler.
    rng = random.Random(config['seed'] + 100000 + arm['next_batch_index'])
    batch = sample_batch(documents['train'], config['batch_size'], config['sequence_length'], rng)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, count, _ = _loss(model, batch)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite training loss')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
    optimizer.step()
    arm['updates'] += 1
    arm['next_batch_index'] += 1
    arm['processed_training_tokens'] += batch['inputs'].numel()
    arm['supervised_training_tokens'] += count
    arm['candidates'][index]['ancestral_updates'] += 1
    if arm['updates'] % interval:
        return
    validation = _evaluate(model, documents['validation'], config, config['validation_batches'], config['seed'] + 200000)
    arm['validation_forward_batches'] += validation['forward_batches']
    arm['validation_processed_tokens'] += validation['processed_tokens']
    arm['validation_supervised_tokens'] += validation['supervised_tokens']
    arm['validation_history'].append({'global_updates': arm['updates'], 'candidate': index, **validation})
    if arm['best'] is None or validation['loss'] < arm['best']['validation']['loss']:
        weights = copy.deepcopy(model.state_dict())
        arm['best'] = {'model': weights, 'validation': validation, 'global_updates': arm['updates'],
                       'candidate': index, 'dropout': arm['candidates'][index]['dropout'],
                       'weight_sha256': _hash_weights(weights)}
    if name == 'evolution':
        arm['round_losses'].append(validation['loss'])
        if len(arm['round_losses']) == 4:
            if arm['updates'] < config['max_total_updates']:
                best = min(range(4), key=lambda i: (arm['round_losses'][i], i))
                worst = max(range(4), key=lambda i: (arm['round_losses'][i], i))
                mutation = random.Random()
                mutation.setstate(arm['evolution_rng_state'])
                parent = arm['candidates'][best]
                before = _hash_weights(models[worst].state_dict())
                models[worst].load_state_dict(copy.deepcopy(models[best].state_dict()))
                optimizers[worst].load_state_dict(copy.deepcopy(optimizers[best].state_dict()))
                lr = max(1e-5, min(.03, parent['learning_rate'] * mutation.choice((.8, 1.2))))
                dropout = max(0., min(.5, parent['dropout'] + mutation.choice((-.05, .05))))
                models[worst].set_dropout(dropout)
                for group in optimizers[worst].param_groups:
                    group['lr'] = lr
                child = arm['candidates'][worst]
                child.update(learning_rate=lr, dropout=dropout, ancestral_updates=parent['ancestral_updates'])
                arm['inheritance'].append({'round': (block + 1) // 4, 'parent': best, 'child': worst,
                    'global_updates': arm['updates'], 'parent_ancestral_updates': parent['ancestral_updates'],
                    'child_previous_weight_sha256': before, 'inherited_weight_sha256': _hash_weights(models[best].state_dict()),
                    'learning_rate': lr, 'dropout': dropout, 'copied_optimizer_state': True})
                arm['evolution_rng_state'] = mutation.getstate()
            arm['round_losses'] = []


def _report(state, directory):
    arms = {}
    for name, arm in state['arms'].items():
        arms[name] = {key: arm[key] for key in ('updates', 'status', 'processed_training_tokens',
            'supervised_training_tokens', 'validation_forward_batches', 'validation_processed_tokens',
            'validation_supervised_tokens', 'cpu_seconds', 'wall_seconds', 'validation_history', 'inheritance', 'test')}
        arms[name]['best_validation'] = arm['best']['validation'] if arm['best'] else None
        arms[name]['champion'] = str(Path(directory) / (name + '_champion.pt')) if arm['best'] else None
        arms[name]['candidate_ancestral_updates'] = [candidate['ancestral_updates'] for candidate in arm['candidates']]
        arms[name]['parameters_per_model'] = sum(tensor.numel() for tensor in arm['candidates'][0]['model'].values())
        arms[name]['population_size'] = len(arm['candidates'])
    a, b = state['arms']['ordinary'], state['arms']['evolution']
    matched = {key: a[key] == b[key] for key in ('updates', 'processed_training_tokens', 'supervised_training_tokens',
               'validation_forward_batches', 'validation_processed_tokens', 'validation_supervised_tokens')}
    return {'schema': SCHEMA, 'status': state['status'], 'config': state['config'],
            'data_fingerprint': state['data_fingerprint'], 'arms': arms, 'matched_operation_counts': matched,
            'shared_preparation_cpu_seconds': state['shared_preparation_cpu_seconds'],
            'shared_preparation_wall_seconds': state['shared_preparation_wall_seconds'],
            'compute_claim': 'Operation-matched only. Total CPU and wall time are measured, not equalized.',
            'timing_scope': 'Per-arm initialization/restore, training, validation, inheritance, export and checkpoints; test separately included after selection. Shared document preparation is separate.',
            'control_limitations': 'PBT has four initial seeds; ordinary training uses the same seed as candidate zero. A best-of-four independent-training control is not included.',
            'test_policy': 'Each selected arm champion evaluated once, only after both arms finish.',
            'validation_loss_difference_evolution_minus_ordinary':
                b['best']['validation']['loss'] - a['best']['validation']['loss'] if a['best'] and b['best'] else None}


def compare(config, data, run_dir, resume=False):
    """Run/resume ordinary and four-member PBT arms with matched operation budgets."""
    config = _config(config)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    shared_cpu, shared_wall = time.process_time(), time.monotonic()
    documents, fingerprint = _documents(data)
    shared_cpu, shared_wall = time.process_time() - shared_cpu, time.monotonic() - shared_wall
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'training.pt'
    if resume:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('schema') != SCHEMA or state['data_fingerprint'] != fingerprint:
            raise ValueError('comparison checkpoint schema or dataset fingerprint mismatch')
        for key in config:
            if key not in ('max_seconds', 'checkpoint_every') and config[key] != state['config'][key]:
                raise ValueError(f'cannot change {key} while resuming comparison')
        state['config'] = config
        if state['status'] == 'complete':
            # Recover a missing/stale report after a crash between the final
            # checkpoint commit and report publication, without reusing test.
            report = _report(state, directory)
            _json(directory / 'report.json', report)
            return report
    else:
        if any(directory.iterdir()):
            raise ValueError('comparison run directory must be empty; use resume or a new directory')
        state = {'schema': SCHEMA, 'config': config, 'data_fingerprint': fingerprint,
                 'arms': {}, 'active_arm': 'ordinary', 'status': 'running',
                 'shared_preparation_cpu_seconds': shared_cpu, 'shared_preparation_wall_seconds': shared_wall}
        for name in ARM_NAMES:
            cpu, wall = time.process_time(), time.monotonic()
            state['arms'][name] = _new_arm(name, config)
            state['arms'][name]['cpu_seconds'] += time.process_time() - cpu
            state['arms'][name]['wall_seconds'] += time.monotonic() - wall
        _persist(state, directory)
    state['status'] = 'running'
    stop = [False]
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()
    try:
        for name in ARM_NAMES:
            arm = state['arms'][name]
            if arm['updates'] >= config['max_total_updates']:
                continue
            state['active_arm'] = name
            arm['status'] = 'running'
            cpu, wall = time.process_time(), time.monotonic()
            models, optimizers = _restore(arm, config)
            arm['cpu_seconds'] += time.process_time() - cpu
            arm['wall_seconds'] += time.monotonic() - wall
            while arm['updates'] < config['max_total_updates']:
                if stop[0] or (config['max_seconds'] and time.monotonic() - started >= config['max_seconds']):
                    state['status'] = 'interrupted' if stop[0] else 'time_limit'
                    _sync(arm, models, optimizers)
                    cpu, wall = time.process_time(), time.monotonic()
                    _export_best(arm, config, directory / (name + '_champion.pt'))
                    arm['cpu_seconds'] += time.process_time() - cpu
                    arm['wall_seconds'] += time.monotonic() - wall
                    _persist(state, directory)
                    report = _report(state, directory)
                    _json(directory / 'report.json', report)
                    return report
                cpu, wall = time.process_time(), time.monotonic()
                _step_arm(name, arm, models, optimizers, documents, config)
                _sync(arm, models, optimizers)
                arm['cpu_seconds'] += time.process_time() - cpu
                arm['wall_seconds'] += time.monotonic() - wall
                if arm['updates'] % config['checkpoint_every'] == 0:
                    _persist(state, directory)
            cpu, wall = time.process_time(), time.monotonic()
            _export_best(arm, config, directory / (name + '_champion.pt'))
            arm['status'] = 'trained'
            arm['cpu_seconds'] += time.process_time() - cpu
            arm['wall_seconds'] += time.monotonic() - wall
            _persist(state, directory)
        # Test has no influence on any training/selection decision. Checkpoint
        # each result so an interrupted finalization never repeats completed tests.
        for name in ARM_NAMES:
            state['active_arm'] = name
            arm = state['arms'][name]
            if arm['test'] is None:
                cpu, wall = time.process_time(), time.monotonic()
                with torch.random.fork_rng(devices=[]):
                    model = ByteGRULM(NeuralConfig(**{**config['model'], 'dropout': arm['best']['dropout']}))
                    model.load_state_dict(arm['best']['model'])
                    arm['test'] = _evaluate(model, documents['test'], config, config['test_batches'], config['seed'] + 300000)
                _export_best(arm, config, directory / (name + '_champion.pt'))
                arm['cpu_seconds'] += time.process_time() - cpu
                arm['wall_seconds'] += time.monotonic() - wall
                arm['status'] = 'complete'
                _persist(state, directory)
        state['status'] = 'complete'
        _persist(state, directory)
        report = _report(state, directory)
        _json(directory / 'report.json', report)
        return report
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        _persist(state, directory)
        raise
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)


def benchmark(config, data, run_dir):
    """Small complete pipeline benchmark, with quality explicitly unestablished."""
    cfg = dict(config)
    cfg.update(max_total_updates=4, rounds=1, validation_batches=1, test_batches=1)
    started = time.monotonic()
    result = compare(cfg, data, run_dir)
    result['benchmark_wall_seconds'] = time.monotonic() - started
    result['quality_claim'] = 'None: four updates per arm are a pipeline/throughput test.'
    _json(Path(run_dir) / 'benchmark.json', result)
    return result


def train_memory(config, run_dir):
    """Train a bounded synthetic digit-recall curriculum; report all measured results.

    Source cues are ablated only in diagnostic evaluation, never in training.
    A 95% threshold is a success criterion, not a promised outcome.
    """
    cfg = {'seed': 11, 'threads': 2, 'batch_size': 32, 'max_updates': 2000,
           'max_seconds': 240., 'learning_rate': .003, 'test_documents': 200,
           'model': {'embedding_dim': 32, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.}}
    cfg.update(copy.deepcopy(config))
    if not 1 <= int(cfg['threads']) <= 2 or not 1 <= int(cfg['max_updates']) <= 20000:
        raise ValueError('memory threads must be 1..2 and max_updates 1..20000')
    if not math.isfinite(float(cfg['max_seconds'])) or not 0 < float(cfg['max_seconds']) <= 300:
        raise ValueError('memory max_seconds must be in (0, 300]')
    if not 1 <= int(cfg['batch_size']) <= 128 or not 10 <= int(cfg['test_documents']) <= 10000:
        raise ValueError('invalid memory batch or evaluation size')
    if int(cfg['test_documents']) % 10:
        raise ValueError('test_documents must be divisible by ten for an exactly balanced chance baseline')
    if not math.isfinite(float(cfg['learning_rate'])) or not 0 < float(cfg['learning_rate']) <= .1:
        raise ValueError('memory learning_rate must be finite and in (0, .1]')
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('memory run directory must be empty')
    torch.set_num_threads(int(cfg['threads']))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(cfg['seed']))
    model = ByteGRULM(NeuralConfig(**cfg['model']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))
    rng = random.Random(int(cfg['seed']))
    delays = [0, 8, 16, 32, 64]
    curriculum = {delay: make_recall_documents(400, delay=delay, seed=int(cfg['seed']) + delay) for delay in delays}
    stage, updates, loss_tokens = 0, 0, 0
    history = []
    start, cpu = time.monotonic(), time.process_time()
    while updates < int(cfg['max_updates']) and time.monotonic() - start < float(cfg['max_seconds']):
        delay = delays[stage]
        # Retain some shorter examples after each promotion to reduce forgetting.
        chosen = delays[rng.randrange(stage + 1)] if rng.random() < .2 else delay
        docs = curriculum[chosen]
        batch = sample_batch(docs, int(cfg['batch_size']), 80, rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, count, _ = _loss(model, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        updates += 1
        loss_tokens += count
        if updates % 25 == 0:
            probe = make_recall_documents(100, delay=delay, seed=int(cfg['seed']) + 10000 + delay)
            measurement = _recall_accuracy(model, probe, int(cfg['batch_size']))
            history.append({'updates': updates, 'delay': delay, 'validation_accuracy': measurement})
            _json(directory / 'status.json', {'status': 'running', 'updates': updates, 'delay': delay,
                                              'latest': history[-1], 'elapsed_seconds': time.monotonic() - start})
            if measurement >= .97 and stage < len(delays) - 1:
                stage += 1
            elif measurement >= .99 and stage == len(delays) - 1:
                break
    results = {}
    for delay in (32, 64):
        docs = make_recall_documents(int(cfg['test_documents']), delay=delay, seed=int(cfg['seed']) + 20000 + delay)
        accuracy = _recall_accuracy(model, docs, int(cfg['batch_size']))
        ablated = []
        for doc in docs:
            tokens = list(doc.tokens)
            tokens[doc.metadata['source_digit_index']] = 32  # source digit -> space; answer untouched
            ablated.append(SequenceDocument(tokens=tuple(tokens), loss_mask=doc.loss_mask,
                                             kind=doc.kind, metadata=doc.metadata))
        ablation_accuracy = _recall_accuracy(model, ablated, int(cfg['batch_size']))
        results[str(delay)] = {'accuracy': accuracy, 'ablated_source_accuracy': ablation_accuracy,
                               'chance': .1, 'examples': len(docs),
                               'meets_95_percent_goal': accuracy >= .95,
                               'evidence_uses_source_cue': accuracy - ablation_accuracy >= .5}
    report = {'schema': 'golem-neural-memory-v1', 'status': 'complete', 'config': cfg,
              'updates': updates, 'supervised_training_tokens': loss_tokens,
              'wall_seconds': time.monotonic() - start, 'cpu_seconds': time.process_time() - cpu,
              'curriculum': history, 'recall': results,
              'goal_achieved': all(row['meets_95_percent_goal'] and row['evidence_uses_source_cue'] for row in results.values()),
              'interpretation': 'Synthetic digit recall only; this does not establish English understanding or conversation.'}
    save_neural_checkpoint(directory / 'memory_champion.pt', model, metadata={'task': 'synthetic_recall', 'results': results})
    _save(directory / 'memory_training.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': rng.getstate(), 'updates': updates,
        'curriculum_stage': stage, 'config': cfg})
    _json(directory / 'report.json', report)
    _json(directory / 'status.json', report)
    return report


def _recall_accuracy(model, docs, batch_size):
    # Evaluate every document exactly once; balanced digit groups make chance 10%.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(docs), batch_size):
            selected = docs[start:start + batch_size]
            length = max(len(doc.tokens) - 1 for doc in selected)
            inputs = torch.zeros((len(selected), length), dtype=torch.long)
            for row, doc in enumerate(selected):
                inputs[row, :len(doc.tokens) - 1] = torch.tensor(doc.tokens[:-1])
            logits, _ = model(inputs)
            for row, doc in enumerate(selected):
                answer = doc.metadata['answer_index']
                correct += int(logits[row, answer - 1].argmax() == doc.tokens[answer])
                total += 1
    return correct / total
