"""Joint final-reply/meaning training and bounded four-candidate recipe evolution.

Past dialogue is conditioning data only, including deliberately wrong assistant
messages. Only the current verified reference reply and its act are supervised.
The first-response weight applies to current reply bytes, never history tokens.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import threading
import time

import torch
from torch import nn

from .semantic_model import (ACTS, SemanticGRU, MEMORY_POLICY, encode_history,
    generate_semantic, save_semantic_checkpoint, warm_start_semantic)
from .neural_model import NeuralConfig, PAD, END, encode_bytes
from .neural_training import _json, _save, _hash_weights, _checkpoint_hash
from .chat_training import _device, _seed, _cpu, _rng_state, _restore_rng, _sync_device


SCHEMA = 'golem-semantic-training-v1'
BOUNDS = {'learning_rate': (1e-5, .01), 'dropout': (0., .4),
          'intent_loss_weight': (.25, 4.), 'first_response_weight': (1., 6.), 'act_weight': (.25, 4.)}


def _config(config):
    c = {'strategies': ['ordinary', 'evolution'], 'population_size': 4, 'rounds': 2,
         'max_total_updates': 40, 'batch_size': 8, 'max_context_tokens': 512,
         'max_reply_tokens': 128, 'first_response_bytes': 8, 'max_new_tokens': 120,
         'threads': 2, 'device': 'cpu', 'gpu_memory_limit_gib': 2., 'generation_device': 'cpu',
         'seed': 79, 'max_seconds': 1800., 'checkpoint_every': 50, 'checkpoint_seconds': 60.,
         'validation_batches': 4, 'validation_episodes': 16, 'test_episodes': 32,
         'evaluate_test_on_complete': True, 'strategy_schedule': 'round_robin',
         'learning_rate': .0005, 'dropout': .1, 'weight_decay': .01, 'gradient_clip': 1.,
         'intent_loss_weight': 1., 'first_response_weight': 3., 'meaning_condition_gain': 1., 'replace_count': 2,
         'evolve_curriculum': True, 'model': NeuralConfig().to_dict()}
    c.update(copy.deepcopy(config))
    c['model'] = {**NeuralConfig().to_dict(), **config.get('model', {})}
    if 'dropout' not in config:
        c['dropout'] = c['model']['dropout']
    c['model']['dropout'] = c['dropout']
    # Explicitly set these to the same values across long-run stages. Defaults
    # preserve the original seed-derived evaluation behavior for legacy runs.
    c.setdefault('validation_seed', c['seed'] + 40000)
    c.setdefault('validation_nll_seed', c['seed'] + 30000)
    c.setdefault('validation_batch_size', c['batch_size'])
    if type(c['evaluate_test_on_complete']) is not bool:
        raise ValueError('evaluate_test_on_complete must be a boolean')
    for name in ('validation_seed', 'validation_nll_seed'):
        if type(c[name]) is not int:
            raise ValueError(f'{name} must be an integer')
    for key, low, high in (('population_size', 4, 4), ('rounds', 1, 100000),
            ('max_total_updates', 4, 10000000), ('batch_size', 1, 512), ('max_context_tokens', 32, 512),
            ('max_reply_tokens', 2, 512), ('first_response_bytes', 1, 32), ('max_new_tokens', 1, 120),
            ('threads', 1, 128), ('checkpoint_every', 1, 1000000), ('validation_batches', 1, 512),
            ('validation_episodes', 1, 256), ('test_episodes', 1, 256), ('replace_count', 1, 2),
            ('validation_batch_size', 1, 512)):
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(f'{key} must be an integer in {low}..{high}')
    for key in ('max_seconds', 'checkpoint_seconds', 'gpu_memory_limit_gib', 'learning_rate',
                'dropout', 'weight_decay', 'gradient_clip', 'intent_loss_weight', 'first_response_weight',
                'meaning_condition_gain'):
        c[key] = float(c[key])
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('learning_rate', 'dropout', 'intent_loss_weight', 'first_response_weight'):
        if not BOUNDS[key][0] <= c[key] <= BOUNDS[key][1]:
            raise ValueError(f'{key} outside bounded recipe range')
    if not 1. <= c['meaning_condition_gain'] <= 64.:
        raise ValueError('meaning_condition_gain must be in 1..64')
    if c['max_seconds'] > 86400 or not 0 < c['checkpoint_seconds'] <= 3600 or not 0 < c['gpu_memory_limit_gib'] <= 256 or c['gradient_clip'] <= 0:
        raise ValueError('invalid runtime, checkpoint, GPU memory or gradient budget')
    if c['max_total_updates'] % (4 * c['rounds']):
        raise ValueError('max_total_updates must be divisible by 4 * rounds')
    if not isinstance(c['strategies'], (list, tuple)) or not c['strategies'] or len(set(c['strategies'])) != len(c['strategies']) or set(c['strategies']) - {'ordinary', 'independent', 'evolution'}:
        raise ValueError('strategies must be a unique nonempty list of ordinary/independent/evolution')
    c['strategies'] = list(c['strategies'])
    if c['strategy_schedule'] not in ('round_robin', 'sequential'):
        raise ValueError('strategy_schedule must be round_robin or sequential')
    if c['generation_device'] not in ('cpu', 'model'):
        raise ValueError('generation_device must be cpu or model')
    device = torch.device(c['device'])
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and device.index is None):
        raise ValueError('device must be cpu or cuda:N')
    if c.get('init_checkpoint'):
        c['init_checkpoint'] = Path(c['init_checkpoint']).expanduser().as_posix()
        c['init_checkpoint_sha256'] = _checkpoint_hash(c['init_checkpoint'])
    else:
        c.pop('init_checkpoint', None)
        c.pop('init_checkpoint_sha256', None)
    for key in ('repair_overlay', 'teacher_overlay'):
        if c.get(key):
            c[key] = Path(c[key]).expanduser().as_posix()
            c[key + '_sha256'] = _checkpoint_hash(c[key])
        else:
            c.pop(key, None)
            c.pop(key + '_sha256', None)
    NeuralConfig(**c['model'])
    return c


def _validation_protocol(data, config):
    validation_data = {'samples': data.get('samples', {}).get('validation', []),
                       'episodes': data.get('episodes', {}).get('validation', []),
                       'meaning_vocab': data.get('meaning_vocab', [])}
    digest = hashlib.sha256(json.dumps(validation_data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    protocol = {'schema': 'golem-semantic-validation-protocol-v1', 'validation_data_sha256': digest,
                'rollout_seed': config['validation_seed'], 'nll_seed': config['validation_nll_seed'],
                'validation_episodes': config['validation_episodes'], 'validation_batches': config['validation_batches'],
                'validation_batch_size': config['validation_batch_size'], 'max_new_tokens': config['max_new_tokens'],
                'max_context_tokens': config['max_context_tokens'], 'max_reply_tokens': config['max_reply_tokens'],
                'generation_device': config['generation_device'], 'temperature': 0., 'acts': list(ACTS),
                'memory_policy': MEMORY_POLICY,
                'selection': 'zero false-memory, generated score, episode pass rate, negative current-reply NLL'}
    return protocol, hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def encode_samples(data, split, config):
    samples = data.get('samples', {}).get(split)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f'nonempty semantic samples required for {split}')
    encoded = []
    for sample in samples:
        act = sample.get('act', sample.get('intent'))
        reply = sample.get('reply', sample.get('reference'))
        if act not in ACTS or not isinstance(reply, str) or not reply:
            raise ValueError('semantic samples require a declared act and nonempty current reply')
        expected_meaning = ACTS.index(act)
        for labels in (sample, sample.get('frame', {})):
            if not isinstance(labels, dict):
                raise ValueError('semantic frame must be an object')
            if 'meaning_id' in labels and (type(labels['meaning_id']) is not int or labels['meaning_id'] != expected_meaning):
                raise ValueError('integer meaning_id must match the declared ordered act vocabulary')
        if sample.get('frame', {}).get('act', act) != act:
            raise ValueError('frame act differs from sample act')
        prefix = encode_history(sample['turns'], config['max_context_tokens'], trim_history=False)
        targets = encode_bytes(reply) + [END]
        if len(targets) > config['max_reply_tokens']:
            raise ValueError(f'current reply exceeds max_reply_tokens={config["max_reply_tokens"]}')
        encoded.append({'prefix': prefix, 'targets': targets, 'act': expected_meaning,
                        'id': sample.get('id'), 'act_name': act})
    return encoded


def make_batch(samples, device='cpu'):
    """Separate context from targets, so wrong historical replies cannot be labels."""
    maximum_prefix = max(len(row['prefix']) for row in samples)
    maximum_reply = max(len(row['targets']) for row in samples)
    prefix = torch.full((len(samples), maximum_prefix), PAD, dtype=torch.long, device=device)
    targets = torch.full((len(samples), maximum_reply), PAD, dtype=torch.long, device=device)
    mask = torch.zeros((len(samples), maximum_reply), dtype=torch.bool, device=device)
    for index, row in enumerate(samples):
        prefix[index, :len(row['prefix'])] = torch.tensor(row['prefix'], device=device)
        targets[index, :len(row['targets'])] = torch.tensor(row['targets'], device=device)
        mask[index, :len(row['targets'])] = True
    return {'prefix': prefix, 'prefix_lengths': torch.tensor([len(row['prefix']) for row in samples]),
            'reply_inputs': targets[:, :-1], 'targets': targets, 'mask': mask,
            'acts': torch.tensor([row['act'] for row in samples], dtype=torch.long, device=device)}


def joint_loss(model, batch, recipe, first_response_bytes=8):
    logits, meaning_logits = model(batch['prefix'], batch['prefix_lengths'], batch['reply_inputs'])
    errors = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch['targets'].reshape(-1),
                                         reduction='none').reshape_as(batch['targets'])
    mask = batch['mask'].float()
    weights = mask.clone()
    weights[:, :first_response_bytes] *= recipe['first_response_weight']
    reply_loss = (errors * mask).sum() / mask.sum()
    weighted_reply = (errors * weights).sum() / weights.sum()
    meaning_loss = nn.functional.cross_entropy(meaning_logits, batch['acts'])
    total = weighted_reply + recipe['intent_loss_weight'] * meaning_loss
    return total, {'reply_loss': reply_loss, 'meaning_loss': meaning_loss,
                   'meaning_correct': (meaning_logits.argmax(-1) == batch['acts']).sum(),
                   'reply_tokens': int(mask.sum()), 'examples': len(batch['acts']),
                   'prefix_tokens': int(batch['prefix_lengths'].sum())}


def _recipe(config, acts):
    return {**{key: config[key] for key in ('learning_rate', 'dropout', 'intent_loss_weight', 'first_response_weight')},
            'act_weights': {act: 1. for act in sorted(acts)}}


def mutate_recipe(recipe, rng, evolve_curriculum=True):
    child = copy.deepcopy(recipe)
    for name in ('learning_rate', 'intent_loss_weight', 'first_response_weight'):
        low, high = BOUNDS[name]
        child[name] = max(low, min(high, child[name] * rng.choice((.75, 1.25))))
    child['dropout'] = max(0., min(.4, child['dropout'] + rng.choice((-.05, .05))))
    if evolve_curriculum:
        act = rng.choice(sorted(child['act_weights']))
        child['act_weights'][act] = max(.25, min(4., child['act_weights'][act] * rng.choice((.5, 2.))))
    return child


def _sample(buckets, batch_size, recipe, seed):
    rng = random.Random(seed)
    acts = sorted(buckets)
    return [rng.choice(buckets[rng.choices(acts, [recipe['act_weights'][act] for act in acts], k=1)[0]])
            for _ in range(batch_size)]


def _clone(weights, config, dropout):
    with torch.random.fork_rng(devices=[]):
        model = SemanticGRU(NeuralConfig(**{**config['model'], 'dropout': dropout}),
                            meaning_condition_gain=config['meaning_condition_gain'])
    model.load_state_dict(weights)
    return model


def _measure(model, samples, config, device, should_stop):
    model.eval()
    rng = random.Random(config['validation_nll_seed'])
    total_loss = total_meaning = correct = tokens = examples = 0.
    neutral_recipe = {'intent_loss_weight': 1., 'first_response_weight': 1.}
    with torch.no_grad():
        for _ in range(config['validation_batches']):
            if should_stop():
                raise InterruptedError('semantic validation stopped')
            batch = make_batch([rng.choice(samples) for _ in range(config['validation_batch_size'])], device)
            loss, row = joint_loss(model, batch, neutral_recipe)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite semantic validation loss')
            total_loss += float(row['reply_loss']) * row['reply_tokens']
            total_meaning += float(row['meaning_loss']) * row['examples']
            correct += int(row['meaning_correct'])
            tokens += row['reply_tokens']
            examples += row['examples']
    return {'reply_loss': total_loss / tokens, 'bits_per_reply_token': total_loss / tokens / math.log(2),
            'meaning_loss': total_meaning / examples, 'meaning_accuracy': correct / examples,
            'reply_tokens': int(tokens), 'examples': int(examples)}


def _rollout(model, data, split, config, should_stop):
    from .conversation_eval import evaluate_conversations
    evaluated = model
    if config['generation_device'] == 'cpu' and next(model.parameters()).device.type != 'cpu':
        evaluated = _clone(_cpu(model.state_dict()), config, model.config.dropout)
    def generate(turns, **kwargs):
        return generate_semantic(evaluated, turns, max_context_tokens=config['max_context_tokens'], **kwargs)
    return evaluate_conversations(generate, data, split=split,
        max_episodes=config['validation_episodes'] if split == 'validation' else config['test_episodes'],
        max_new_tokens=config['max_new_tokens'], seed=config['validation_seed'],
        temperature=0., should_stop=should_stop)


def _quality(validation):
    report = validation['rollout']
    return (int(report.get('false_memory_claims', 0) == 0), float(report['score']),
            float(report.get('whole_episode_pass_rate', 0)), -float(validation['teacher_forced']['reply_loss']))


def _restore(strategy, config, device):
    models, optimizers = [], []
    for candidate in strategy['population']:
        recipe = candidate['recipe']
        model = _clone(candidate['weights'], config, recipe['dropout']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=config['weight_decay'])
        optimizer.load_state_dict(candidate['optimizer'])
        for item in optimizer.state.values():
            for name, value in item.items():
                if torch.is_tensor(value):
                    item[name] = value.to('cpu' if name == 'step' else device)
        models.append(model)
        optimizers.append(optimizer)
    _restore_rng(strategy['rng'], device)
    return models, optimizers


def _capture(strategy, models, optimizers, device):
    _sync_device(device)
    for candidate, model, optimizer in zip(strategy['population'], models, optimizers):
        candidate['weights'] = _cpu(model.state_dict())
        candidate['optimizer'] = _cpu(optimizer.state_dict())
    strategy['rng'] = _rng_state(device)


def _publish(state, directory):
    eligible = []
    for name, strategy in state['strategies'].items():
        best = strategy['best']
        if best is None:
            continue
        model = _clone(best['weights'], state['config'], best['recipe']['dropout'])
        metadata = {'strategy': name, 'global_updates': best['updates'], 'recipe': best['recipe'],
                    'validation': best['validation'], 'selection_split': 'validation',
                    'memory_policy': MEMORY_POLICY, 'initialization': state['initialization'],
                    'validation_protocol': state['validation_protocol'],
                    'validation_protocol_hash': state['validation_protocol_hash'],
                    'strategy_schedule': state['config']['strategy_schedule']}
        save_semantic_checkpoint(directory / (name + '_champion.pt'), model, metadata)
        eligible.append((name, model, metadata, _quality(best['validation'])))
    if eligible:
        name, model, metadata, _ = max(eligible, key=lambda row: row[3])
        save_semantic_checkpoint(directory / 'best.pt', model, metadata)


def _summary(state, directory):
    strategies = {}
    for name, strategy in state['strategies'].items():
        strategies[name] = {key: strategy[key] for key in ('status', 'updates', 'reply_tokens', 'prefix_tokens',
            'validation_calls', 'cpu_seconds', 'wall_seconds', 'births', 'test')}
        strategies[name]['best_validation'] = strategy['best']['validation'] if strategy['best'] else None
        strategies[name]['champion'] = str(directory / (name + '_champion.pt')) if strategy['best'] else None
    return {'schema': SCHEMA, 'status': state['status'], 'active_strategy': state['active_strategy'],
            'schedule_cursor': state.get('schedule_cursor'),
            'config': state['config'], 'data_fingerprint': state['data_fingerprint'],
            'initialization': state['initialization'], 'parameter_count': state['parameter_count'],
            'validation_protocol': state['validation_protocol'],
            'validation_protocol_hash': state['validation_protocol_hash'],
            'test_policy': ('evaluate selected champions after this run completes' if state['config']['evaluate_test_on_complete']
                            else 'deferred until the long-run deadline; test is not prepared, evaluated or used for selection'),
            'memory_policy': MEMORY_POLICY, 'strategies': strategies, 'error': state.get('error'),
            'selection_rule': 'Zero observed validation false-memory claims first; generated conversation score, episode success, then lower current-reply NLL.',
            'supervision': 'Current reference reply plus act only; extra weight on initial current-reply bytes. No previous assistant token loss. Decoder conditioned on predicted soft acts only.',
            'budget_claim': 'Completed strategies receive the same optimizer-update and validation-request budgets. Interrupted strategies may receive unequal work; actual updates, tokens and CPU/wall time are reported. Evolved curricula can change lengths, so total compute is not asserted equal.'}


def _checkpoint(state, directory, models=None, optimizers=None, device=None):
    if models is not None:
        _capture(state['strategies'][state['active_strategy']], models, optimizers, device)
    _save(directory / 'semantic-training.pt', state)
    _publish(state, directory)
    _json(directory / 'status.json', _summary(state, directory))
    _json(directory / 'population.json', {'schema': SCHEMA, 'bounds': BOUNDS,
        'strategies': {name: [{'id': row['id'], 'parent_id': row['parent_id'], 'recipe': row['recipe'],
                              'ancestral_updates': row['ancestral_updates']} for row in strategy['population']]
                       for name, strategy in state['strategies'].items()}})


def _step(name, strategy, models, optimizers, encoded, buckets, data, config, device, should_stop):
    interval = config['max_total_updates'] // (4 * config['rounds'])
    if strategy['evaluation_due']:
        block = (strategy['updates'] - 1) // interval
        slot = 0 if name == 'ordinary' else block % 4
        model, candidate = models[slot], strategy['population'][slot]
        validation = {'rollout': _rollout(model, data, 'validation', config, should_stop),
                      'teacher_forced': _measure(model, encoded['validation'], config, device, should_stop)}
        candidate['validation'] = validation
        strategy['validation_calls'] += 1
        strategy['history'].append({'updates': strategy['updates'], 'candidate': candidate['id'], 'validation': validation})
        if strategy['best'] is None or _quality(validation) > _quality(strategy['best']['validation']):
            strategy['best'] = {'weights': _cpu(model.state_dict()), 'recipe': copy.deepcopy(candidate['recipe']),
                                'validation': validation, 'updates': strategy['updates']}
        if name == 'evolution' and (block + 1) % 4 == 0 and strategy['updates'] < config['max_total_updates']:
            rank = sorted(range(4), key=lambda i: (_quality(strategy['population'][i]['validation']), -i), reverse=True)
            parent = rank[0]
            elite_hash = _hash_weights(_cpu(models[parent].state_dict()))
            rng = random.Random()
            rng.setstate(strategy['genetic_rng'])
            for child in rank[-config['replace_count']:]:
                p, target = strategy['population'][parent], strategy['population'][child]
                models[child].load_state_dict(models[parent].state_dict())
                optimizers[child].load_state_dict(copy.deepcopy(optimizers[parent].state_dict()))
                recipe = mutate_recipe(p['recipe'], rng, config['evolve_curriculum'])
                models[child].set_dropout(recipe['dropout'])
                for group in optimizers[child].param_groups:
                    group['lr'] = recipe['learning_rate']
                target.update(id=strategy['next_id'], parent_id=p['id'], recipe=recipe,
                              ancestral_updates=p['ancestral_updates'], validation=None)
                strategy['next_id'] += 1
                strategy['births'].append({'updates': strategy['updates'], 'parent_id': p['id'], 'child_id': target['id'],
                    'operator': 'full_model_optimizer_inheritance_and_bounded_recipe_mutation',
                    'weight_sha256': elite_hash, 'before_recipe': copy.deepcopy(p['recipe']),
                    'after_recipe': copy.deepcopy(recipe), 'elite_preserved': True})
            if _hash_weights(_cpu(models[parent].state_dict())) != elite_hash:
                raise RuntimeError('semantic genetic operator changed protected elite')
            strategy['genetic_rng'] = rng.getstate()
        strategy['evaluation_due'] = False
        return
    slot = 0 if name == 'ordinary' else (strategy['updates'] // interval) % 4
    model, optimizer, candidate = models[slot], optimizers[slot], strategy['population'][slot]
    recipe = candidate['recipe']
    rows = _sample(buckets, config['batch_size'], recipe, config['seed'] + 100000 + strategy['updates'])
    batch = make_batch(rows, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = joint_loss(model, batch, recipe, config['first_response_bytes'])
    if not torch.isfinite(loss):
        raise ValueError('nonfinite semantic training loss')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    strategy['updates'] += 1
    strategy['reply_tokens'] += metrics['reply_tokens']
    strategy['prefix_tokens'] += metrics['prefix_tokens']
    candidate['ancestral_updates'] += 1
    strategy['evaluation_due'] = strategy['updates'] % interval == 0


def train_semantic(config, data, run_dir, resume=False):
    # Old checkpoints predate the scheduler option and used a sequential loop.
    # Infer their actual schedule before applying the default for fresh runs.
    state = None
    directory = Path(run_dir)
    if resume:
        state = torch.load(directory / 'semantic-training.pt', map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or state.get('schema') != SCHEMA:
            raise ValueError('unsupported semantic checkpoint schema')
        config = {**config}
        config.setdefault('strategy_schedule', state['config'].get('strategy_schedule', 'sequential'))
    config = _config(config)
    if config.get('teacher_overlay'):
        from .world_replay import apply_repair_overlays
        paths = [config[key] for key in ('repair_overlay', 'teacher_overlay') if config.get(key)]
        data = apply_repair_overlays(data, paths)
    elif config.get('repair_overlay'):
        from .world_replay import apply_repair_overlay
        data = apply_repair_overlay(data, config['repair_overlay'])
    device = _device(config)
    encoded = {split: encode_samples(data, split, config) for split in ('train', 'validation')}
    buckets = {}
    for row in encoded['train']:
        buckets.setdefault(row['act_name'], []).append(row)
    identity = {key: data.get(key, {}) for key in ('samples', 'episodes', 'meaning_graph', 'schema')}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    protocol, protocol_hash = _validation_protocol(data, config)
    directory.mkdir(parents=True, exist_ok=True)
    if resume:
        if state.get('schema') != SCHEMA or state['data_fingerprint'] != fingerprint:
            raise ValueError('semantic checkpoint or dataset identity mismatch')
        for key in set(config) | set(state['config']):
            legacy_default = {'evaluate_test_on_complete': True,
                              'strategy_schedule': 'sequential',
                              'meaning_condition_gain': 1.,
                              'validation_seed': state['config']['seed'] + 40000,
                              'validation_nll_seed': state['config']['seed'] + 30000,
                              'validation_batch_size': state['config']['batch_size']}.get(key)
            if key not in ('max_seconds', 'checkpoint_every', 'checkpoint_seconds') and config.get(key) != state['config'].get(key, legacy_default):
                raise ValueError(f'cannot change {key} during semantic resume')
        if state.get('validation_protocol_hash', protocol_hash) != protocol_hash:
            raise ValueError('validation protocol changed during semantic resume')
        state['config'] = config
        state.setdefault('schedule_cursor', config['strategies'].index(state['active_strategy']))
        state['validation_protocol'], state['validation_protocol_hash'] = protocol, protocol_hash
        if state['status'] == 'complete':
            return _summary(state, directory)
    else:
        if any(directory.iterdir()):
            raise ValueError('semantic run directory must be empty')
        _seed(config['seed'], device)
        if config.get('init_checkpoint'):
            prototype, metadata = warm_start_semantic(config['init_checkpoint'], NeuralConfig(**config['model']))
            if _checkpoint_hash(config['init_checkpoint']) != config['init_checkpoint_sha256']:
                raise ValueError('semantic initialization source changed')
            initialization = {'checkpoint_sha256': config['init_checkpoint_sha256'], 'source_metadata': metadata,
                              'optimizer_reset': True, 'shared_semantic_initialization': True}
        else:
            prototype = SemanticGRU(NeuralConfig(**config['model']),
                                    meaning_condition_gain=config['meaning_condition_gain'])
            initialization = {'random_semantic_initialization': True, 'shared_semantic_initialization': True}
        initial_weights = _cpu(prototype.state_dict())
        if any(not torch.isfinite(value).all() for value in initial_weights.values()):
            raise ValueError('initial semantic weights must be finite')
        initialization['weight_sha256'] = _hash_weights(initial_weights)
        state = {'schema': SCHEMA, 'config': config, 'data_fingerprint': fingerprint,
                 'status': 'running', 'active_strategy': config['strategies'][0], 'strategies': {},
                 'schedule_cursor': 0,
                 'initialization': initialization, 'parameter_count': prototype.parameter_count}
        state['validation_protocol'], state['validation_protocol_hash'] = protocol, protocol_hash
        for name in config['strategies']:
            population = []
            for index in range(1 if name == 'ordinary' else 4):
                model = _clone(initial_weights, config, config['dropout'])
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
                population.append({'id': index, 'parent_id': None, 'recipe': _recipe(config, buckets),
                    'weights': _cpu(model.state_dict()), 'optimizer': _cpu(optimizer.state_dict()),
                    'validation': None, 'ancestral_updates': 0})
            _seed(config['seed'] + 20000, device)
            state['strategies'][name] = {'population': population, 'updates': 0, 'evaluation_due': False,
                'next_id': len(population), 'rng': _rng_state(device), 'genetic_rng': random.Random(config['seed'] + 50000).getstate(),
                'best': None, 'history': [], 'births': [], 'test': None, 'status': 'pending',
                'reply_tokens': 0, 'prefix_tokens': 0, 'validation_calls': 0, 'cpu_seconds': 0., 'wall_seconds': 0.}
        _checkpoint(state, directory)
    state['status'] = 'running'
    state.pop('error', None)
    stop, handlers = [False], {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = last_checkpoint = time.monotonic()
    def should_stop():
        return stop[0] or bool(config['max_seconds'] and time.monotonic() - started >= config['max_seconds'])
    models = optimizers = None
    try:
        while True:
            pending = {name for name, row in state['strategies'].items()
                       if row['updates'] < config['max_total_updates'] or row['evaluation_due']}
            if not pending:
                break
            if should_stop():
                raise InterruptedError('semantic run budget reached')
            if config['strategy_schedule'] == 'sequential':
                name = next(name for name in config['strategies'] if name in pending)
            else:
                count = len(config['strategies'])
                cursor = state['schedule_cursor']
                if type(cursor) is not int or not 0 <= cursor < count:
                    raise ValueError('invalid semantic scheduler cursor')
                index = next((cursor + offset) % count for offset in range(count)
                             if config['strategies'][(cursor + offset) % count] in pending)
                state['schedule_cursor'] = index
                name = config['strategies'][index]
            strategy = state['strategies'][name]
            state['active_strategy'] = name
            models, optimizers = _restore(strategy, config, device)
            strategy['status'] = 'running'
            while strategy['updates'] < config['max_total_updates'] or strategy['evaluation_due']:
                if should_stop():
                    raise InterruptedError('semantic run budget reached')
                was_evaluation = strategy['evaluation_due']
                cpu, wall = time.process_time(), time.monotonic()
                try:
                    _step(name, strategy, models, optimizers, encoded, buckets, data, config, device, should_stop)
                finally:
                    _sync_device(device)
                    strategy['cpu_seconds'] += time.process_time() - cpu
                    strategy['wall_seconds'] += time.monotonic() - wall
                switch = was_evaluation and config['strategy_schedule'] == 'round_robin'
                if switch:
                    # Capture every block in memory, but preserve the configured
                    # disk checkpoint cadence. Only one strategy owns live
                    # device models/optimizers; its RNG is restored on return.
                    _capture(strategy, models, optimizers, device)
                    models = optimizers = None
                    state['schedule_cursor'] = (state['schedule_cursor'] + 1) % len(config['strategies'])
                    if strategy['updates'] >= config['max_total_updates']:
                        strategy['status'] = 'trained'
                if was_evaluation:
                    _publish(state, directory)
                    _json(directory / 'status.json', _summary(state, directory))
                if (not was_evaluation and strategy['updates'] % config['checkpoint_every'] == 0) or time.monotonic() - last_checkpoint >= config['checkpoint_seconds']:
                    _checkpoint(state, directory, models, optimizers, device)
                    last_checkpoint = time.monotonic()
                if switch:
                    break
            if strategy['updates'] >= config['max_total_updates'] and not strategy['evaluation_due']:
                strategy['status'] = 'trained'
                _checkpoint(state, directory, models, optimizers, device)
                last_checkpoint = time.monotonic()
                models = optimizers = None
        if config['evaluate_test_on_complete']:
            test_samples = encode_samples(data, 'test', config)
            for name in config['strategies']:
                strategy = state['strategies'][name]
                state['active_strategy'] = name
                if should_stop():
                    raise InterruptedError('semantic final test budget reached')
                if strategy['test'] is None:
                    model = _clone(strategy['best']['weights'], config, strategy['best']['recipe']['dropout']).to(device)
                    strategy['test'] = {'rollout': _rollout(model, data, 'test', config, should_stop),
                                        'teacher_forced': _measure(model, test_samples, config, device, should_stop)}
                    strategy['status'] = 'complete'
                    _checkpoint(state, directory)
        else:
            for strategy in state['strategies'].values():
                strategy['status'] = 'complete_test_deferred'
        state['status'] = 'complete'
    except InterruptedError:
        state['status'] = 'interrupted' if stop[0] else 'time_limit'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        raise
    finally:
        try:
            _checkpoint(state, directory, models, optimizers, device)
            _json(directory / 'report.json', _summary(state, directory))
        finally:
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
    return _summary(state, directory)
