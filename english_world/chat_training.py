"""Bounded generated-chat selection with ordinary, independent and genetic controls.

This trains Golem's byte GRU with gradients. Genetic operators copy bounded
tensor modules and training recipes; they never execute generated source code.
Every strategy receives the same optimizer-update and processed-token budgets.
Evolved curricula can change supervised-token counts, which are reported rather
than asserted equal. Generated rubric scores are diagnostics, not human accuracy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import threading
import time

import torch
from torch import nn

from .neural_data import prepare_documents, dialogue_document, sample_batch
from .neural_model import ByteGRULM, NeuralConfig, load_neural_checkpoint, save_neural_checkpoint
from .neural_training import _json, _save, _hash_weights, _checkpoint_hash


SCHEMA = 'golem-chat-training-v1'
STRATEGIES = ('ordinary', 'independent', 'evolution')
GENE_BOUNDS = {'learning_rate': (1e-5, .01), 'dropout': (0., .4),
               'weight_decay': (0., .1), 'gradient_clip': (.25, 2.),
               'intent_weight': (.25, 4.)}


def _config(config):
    result = {'strategies': list(STRATEGIES), 'population_size': 4, 'rounds': 2,
              'max_total_updates': 40, 'batch_size': 4, 'sequence_length': 192,
              'threads': 2, 'device': 'cpu', 'seed': 17, 'max_seconds': 1800.,
              'generation_device': 'cpu',
              'gpu_memory_limit_gib': 2.,
              'checkpoint_every': 50, 'checkpoint_seconds': 60.,
              'validation_batches': 2, 'validation_cases': 8, 'test_cases': 32,
              'max_new_tokens': 96, 'learning_rate': .0005, 'weight_decay': .01,
              'gradient_clip': 1., 'replace_count': 2, 'weight_mutation_probability': .35,
              'weight_noise_rms_fraction': .002, 'crossover_probability': .25,
              'evolve_curriculum': True, 'model': NeuralConfig().to_dict()}
    result.update(copy.deepcopy(config))
    result['model'] = {**NeuralConfig().to_dict(), **config.get('model', {})}
    for name, low, high in (('population_size', 4, 4), ('rounds', 1, 100000),
            ('max_total_updates', 4, 10000000), ('batch_size', 1, 512), ('sequence_length', 16, 4096),
            ('threads', 1, 128), ('checkpoint_every', 1, 1000000), ('validation_batches', 1, 512),
            ('validation_cases', 1, 512), ('test_cases', 1, 512), ('max_new_tokens', 1, 120),
            ('replace_count', 1, 2)):
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    for name in ('max_seconds', 'checkpoint_seconds', 'learning_rate', 'weight_decay',
                 'gradient_clip', 'weight_mutation_probability', 'weight_noise_rms_fraction', 'crossover_probability'):
        result[name] = float(result[name])
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    result['gpu_memory_limit_gib'] = float(result['gpu_memory_limit_gib'])
    if not math.isfinite(result['gpu_memory_limit_gib']) or not 0 < result['gpu_memory_limit_gib'] <= 256:
        raise ValueError('gpu_memory_limit_gib must be finite and in (0,256]')
    if not 0 < result['checkpoint_seconds'] <= 3600 or result['max_seconds'] > 86400:
        raise ValueError('checkpoint_seconds must be (0,3600], max_seconds at most 86400')
    for name in ('learning_rate', 'weight_decay', 'gradient_clip'):
        if not GENE_BOUNDS[name][0] <= result[name] <= GENE_BOUNDS[name][1]:
            raise ValueError(f'{name} outside gene bounds')
    if not 0 <= result['model']['dropout'] <= .4:
        raise ValueError('initial dropout must be in 0..0.4')
    if any(result[name] > 1 for name in ('weight_mutation_probability', 'crossover_probability')) or result['weight_noise_rms_fraction'] > .02:
        raise ValueError('mutation/crossover probabilities must be <=1; RMS noise fraction <=.02')
    if result['max_total_updates'] % (4 * result['rounds']):
        raise ValueError('max_total_updates must be divisible by 4 * rounds')
    if not isinstance(result['strategies'], (list, tuple)) or not result['strategies'] or len(set(result['strategies'])) != len(result['strategies']) or set(result['strategies']) - set(STRATEGIES):
        raise ValueError('strategies must be a nonempty unique list of ordinary/independent/evolution')
    result['strategies'] = list(result['strategies'])
    if not re.fullmatch(r'cpu|cuda:[0-9]+', str(result['device'])):
        raise ValueError('device must be cpu or cuda:N (including the ROCm PyTorch CUDA API)')
    if result['generation_device'] not in ('cpu', 'model'):
        raise ValueError('generation_device must be cpu or model')
    if result.get('init_checkpoint'):
        path = Path(result['init_checkpoint']).expanduser()
        result['init_checkpoint'] = path.as_posix()
        result['init_checkpoint_sha256'] = _checkpoint_hash(path)
    else:
        result.pop('init_checkpoint', None)
        result.pop('init_checkpoint_sha256', None)
    NeuralConfig(**result['model'])
    return result


def _device(config):
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError(f'requested device is unavailable: {device}')
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(1., config['gpu_memory_limit_gib'] * 1024**3 / total)
        # Configure the allocator before moving models or training batches onto
        # CUDA/ROCm. This limits this process, not another user's GPU allocations.
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.set_device(device)
    torch.set_num_threads(config['threads'])
    # Strict CPU replay is tested. GPU RNN implementations can vary by backend;
    # unsupported deterministic kernels should not silently be called identical.
    torch.use_deterministic_algorithms(device.type == 'cpu')
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = device.type == 'cpu'
    return device


def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _seed(seed, device):
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _rng_state(device):
    result = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        result['accelerator'] = torch.cuda.get_rng_state(device)
    return result


def _restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        if 'accelerator' not in state:
            raise ValueError('accelerator RNG missing; resume on the original device type')
        torch.cuda.set_rng_state(state['accelerator'], device)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _prepare(data, config):
    documents = {split: prepare_documents(data, split) for split in ('train', 'validation')}
    for split, docs in documents.items():
        maximum = max(len(doc.tokens) - 1 for doc in docs)
        if maximum > config['sequence_length']:
            raise ValueError(f'{split} chat requires sequence_length >= {maximum}; request/history cropping is disabled')
    labels = {}
    for conversation in data.get('conversations', {}).get('train', []):
        doc = dialogue_document(conversation['turns'])
        intent = str(conversation.get('intent', 'general'))
        if doc.tokens in labels and labels[doc.tokens] != intent:
            raise ValueError('the same training conversation has conflicting intent labels')
        labels[doc.tokens] = intent
    buckets = {}
    for document in documents['train']:
        buckets.setdefault(labels.get(document.tokens, 'general'), []).append(document)
    # Held-out values identify the experiment but are never used for gradients
    # or candidate selection. Test documents are prepared only after training.
    identity = {key: data.get(key, {}) for key in ('texts', 'conversations', 'chat_cases', 'groups')}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()
    return documents, buckets, digest


def _initial_gene(config, intents):
    return {'learning_rate': config['learning_rate'], 'dropout': config['model']['dropout'],
            'weight_decay': config['weight_decay'], 'gradient_clip': config['gradient_clip'],
            'intent_weights': {intent: 1. for intent in sorted(intents)}}


def _gene_hash(gene):
    return hashlib.sha256(json.dumps(gene, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _source(config):
    if not config.get('init_checkpoint'):
        return None, {'mode': 'random_initialization'}
    model, metadata = load_neural_checkpoint(config['init_checkpoint'])
    expected = NeuralConfig(**config['model']).to_dict()
    if any(value != expected[key] for key, value in model.config.to_dict().items() if key != 'dropout'):
        raise ValueError('warm-start checkpoint architecture differs from configured model')
    if _checkpoint_hash(config['init_checkpoint']) != config['init_checkpoint_sha256']:
        raise ValueError('warm-start checkpoint changed during loading')
    weights = _cpu(model.state_dict())
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('warm-start checkpoint contains nonfinite weights')
    return weights, {'mode': 'shared_golem_checkpoint', 'checkpoint_sha256': config['init_checkpoint_sha256'],
                     'weight_sha256': _hash_weights(weights), 'source_metadata': metadata,
                     'optimizer_reset': True, 'prior_training_not_in_new_update_budget': True}


def _new_strategy(name, config, intents, source_weights, device):
    candidates = []
    for index in range(1 if name == 'ordinary' else 4):
        _seed(config['seed'] + index, device)
        model = ByteGRULM(NeuralConfig(**config['model']))
        if source_weights is not None:
            model.load_state_dict(source_weights)
        gene = _initial_gene(config, intents)
        optimizer = torch.optim.AdamW(model.parameters(), lr=gene['learning_rate'], weight_decay=gene['weight_decay'])
        candidates.append({'id': index, 'genome': gene, 'model_state': _cpu(model.state_dict()),
                           'optimizer_state': _cpu(optimizer.state_dict()), 'ancestral_updates': 0,
                           'parent_id': None, 'birth_round': 0, 'latest_validation': None})
    _seed(config['seed'] + 10000, device)
    return {'candidates': candidates, 'updates': 0, 'batch_index': 0, 'evaluation_due': False,
            'next_id': len(candidates), 'processed_training_tokens': 0, 'supervised_training_tokens': 0,
            'validation_forward_batches': 0, 'generated_validation_calls': 0, 'generated_validation_cases': 0,
            'generated_validation_output_bytes': 0, 'history': [], 'births': [], 'best': None,
            'test': None, 'status': 'pending', 'cpu_seconds': 0., 'wall_seconds': 0.,
            'rng_state': _rng_state(device), 'genetic_rng_state': random.Random(config['seed'] + 20000).getstate()}


def _runtime(strategy, config, device):
    models, optimizers = [], []
    for candidate in strategy['candidates']:
        gene = candidate['genome']
        model = ByteGRULM(NeuralConfig(**{**config['model'], 'dropout': gene['dropout']})).to(device)
        model.load_state_dict(candidate['model_state'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=gene['learning_rate'], weight_decay=gene['weight_decay'])
        optimizer.load_state_dict(candidate['optimizer_state'])
        # Torch normally migrates states while loading; make tensor placement
        # explicit for CPU -> CUDA/ROCm resumes. Noncapturable Adam step stays CPU.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to('cpu' if key == 'step' else device)
        models.append(model)
        optimizers.append(optimizer)
    _restore_rng(strategy['rng_state'], device)
    return models, optimizers


def _capture(strategy, models, optimizers, device):
    _sync_device(device)
    for candidate, model, optimizer in zip(strategy['candidates'], models, optimizers):
        candidate['model_state'] = _cpu(model.state_dict())
        candidate['optimizer_state'] = _cpu(optimizer.state_dict())
    strategy['rng_state'] = _rng_state(device)


def _nll(model, docs, config, device, seed, should_stop=None):
    model.eval()
    rng = random.Random(seed)
    nll = tokens = 0
    with torch.no_grad():
        for _ in range(config['validation_batches']):
            if should_stop is not None and should_stop():
                raise InterruptedError('chat evaluation stopped between validation batches')
            batch = sample_batch(docs, config['batch_size'], config['sequence_length'], rng, device=device)
            logits, _ = model(batch['inputs'])
            mask = batch['loss_mask'].reshape(-1)
            count = int(mask.sum())
            if not count:
                raise ValueError('validation batch has no assistant targets')
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask],
                                               batch['targets'].reshape(-1)[mask], reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('nonfinite assistant validation loss')
            nll += float(loss)
            tokens += count
    return {'loss': nll / tokens, 'bits_per_target': nll / tokens / math.log(2),
            'supervised_tokens': tokens, 'forward_batches': config['validation_batches']}


def _chat_evaluate(model, data, split, config, should_stop=None):
    from .chat_eval import evaluate_chat
    evaluation_model = model
    if config['generation_device'] == 'cpu' and next(model.parameters()).device.type != 'cpu':
        # Single-token GPU launches can dominate small recurrent inference. A
        # separate CPU clone avoids changing the live model or optimizer device.
        with torch.random.fork_rng(devices=[]):
            evaluation_model = ByteGRULM(model.config)
        evaluation_model.load_state_dict(_cpu(model.state_dict()))
        evaluation_model.eval()
    return evaluate_chat(evaluation_model, data, split=split,
                         max_cases=config['validation_cases'] if split == 'validation' else config['test_cases'],
                         max_new_tokens=config['max_new_tokens'], seed=config['seed'] + 30000,
                         temperature=0., include_reference_nll=False, should_stop=should_stop)


def _quality(validation):
    return float(validation['chat']['score']), -float(validation['nll']['loss'])


def _clamp(name, value):
    low, high = GENE_BOUNDS[name]
    return max(low, min(high, value))


def mutate_genome(parent, rng, evolve_curriculum=True):
    child = copy.deepcopy(parent)
    child['learning_rate'] = _clamp('learning_rate', child['learning_rate'] * rng.choice((.7, 1.3)))
    child['dropout'] = _clamp('dropout', child['dropout'] + rng.choice((-.05, .05)))
    child['weight_decay'] = _clamp('weight_decay', child['weight_decay'] * rng.choice((.5, 1.5)) + rng.choice((0., .0001)))
    child['gradient_clip'] = _clamp('gradient_clip', child['gradient_clip'] * rng.choice((.8, 1.2)))
    if evolve_curriculum and child['intent_weights']:
        intent = rng.choice(sorted(child['intent_weights']))
        child['intent_weights'][intent] = _clamp('intent_weight', child['intent_weights'][intent] * rng.choice((.5, 2.)))
    return child


def _evolve_population(strategy, models, optimizers, config, device, round_number):
    population = strategy['candidates']
    rank = sorted(range(4), key=lambda index: (_quality(population[index]['latest_validation']), -index), reverse=True)
    elite, protected = rank[0], rank[:4 - config['replace_count']]
    elite_hash = _hash_weights(_cpu(models[elite].state_dict()))
    rng = random.Random()
    rng.setstate(strategy['genetic_rng_state'])
    for child_slot in rank[-config['replace_count']:]:
        parent_slot = elite
        parent = population[parent_slot]
        old_id = population[child_slot]['id']
        before_hash = _hash_weights(_cpu(models[child_slot].state_dict()))
        models[child_slot].load_state_dict(models[parent_slot].state_dict())
        optimizers[child_slot].load_state_dict(copy.deepcopy(optimizers[parent_slot].state_dict()))
        gene = mutate_genome(parent['genome'], rng, config['evolve_curriculum'])
        operators = [{'operator': 'inherit_model_and_optimizer', 'parent_id': parent['id']},
                     {'operator': 'bounded_recipe_mutation', 'before': parent['genome'], 'after': copy.deepcopy(gene)}]
        reset_names = set()
        donors = [index for index in protected if index != parent_slot]
        if donors and rng.random() < config['crossover_probability']:
            donor = rng.choice(donors)
            module = rng.choice(('embedding.', 'gru.', 'output.'))
            donor_parameters = dict(models[donor].named_parameters())
            with torch.no_grad():
                for name, parameter in models[child_slot].named_parameters():
                    if name.startswith(module):
                        parameter.copy_(donor_parameters[name])
                        reset_names.add(name)
            operators.append({'operator': 'module_crossover', 'module': module,
                              'donor_id': population[donor]['id'],
                              'donor_weight_sha256': _hash_weights(_cpu(models[donor].state_dict()))})
        if rng.random() < config['weight_mutation_probability'] and config['weight_noise_rms_fraction']:
            matrices = [(name, parameter) for name, parameter in models[child_slot].named_parameters()
                        if parameter.ndim == 2 and name != 'embedding.weight']
            name, parameter = rng.choice(matrices)
            fraction = config['weight_noise_rms_fraction']
            rms = max(float(parameter.detach().float().square().mean().sqrt()), 1e-6)
            generator = torch.Generator(device='cpu').manual_seed(rng.randrange(2**63))
            noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32).clamp_(-3., 3.).to(device)
            with torch.no_grad():
                parameter.add_(noise.to(parameter.dtype), alpha=rms * fraction)
            reset_names.add(name)
            operators.append({'operator': 'bounded_weight_noise', 'parameter': name,
                              'rms_fraction': fraction, 'noise_std': rms * fraction,
                              'maximum_absolute_delta': 3. * rms * fraction})
        for name, parameter in models[child_slot].named_parameters():
            if name in reset_names:
                optimizers[child_slot].state.pop(parameter, None)
        models[child_slot].set_dropout(gene['dropout'])
        for group in optimizers[child_slot].param_groups:
            group['lr'], group['weight_decay'] = gene['learning_rate'], gene['weight_decay']
        child = population[child_slot]
        child.update(id=strategy['next_id'], genome=gene, parent_id=parent['id'],
                     birth_round=round_number, ancestral_updates=parent['ancestral_updates'], latest_validation=None)
        strategy['next_id'] += 1
        strategy['births'].append({'round': round_number, 'global_updates': strategy['updates'],
            'replaced_id': old_id, 'child_id': child['id'], 'parent_id': parent['id'], 'slot': child_slot,
            'before_weight_sha256': before_hash, 'parent_weight_sha256': elite_hash,
            'after_weight_sha256': _hash_weights(_cpu(models[child_slot].state_dict())),
            'genome_sha256': _gene_hash(gene), 'operators': operators,
            'optimizer_reset_parameters': sorted(reset_names), 'elite_preserved': True})
    if _hash_weights(_cpu(models[elite].state_dict())) != elite_hash:
        raise RuntimeError('genetic operator changed the protected elite')
    strategy['genetic_rng_state'] = rng.getstate()


def _step(name, strategy, models, optimizers, documents, buckets, data, config, device, should_stop=None):
    interval = config['max_total_updates'] // (4 * config['rounds'])
    if strategy['evaluation_due']:
        block = (strategy['updates'] - 1) // interval
        slot = 0 if name == 'ordinary' else block % 4
        candidate, model = strategy['candidates'][slot], models[slot]
        validation = {'chat': _chat_evaluate(model, data, 'validation', config, should_stop=should_stop),
                      'nll': _nll(model, documents['validation'], config, device, config['seed'] + 40000, should_stop=should_stop)}
        if not math.isfinite(float(validation['chat']['score'])) or not 0 <= validation['chat']['score'] <= 1:
            raise ValueError('generated chat evaluator score must be finite in 0..1')
        candidate['latest_validation'] = validation
        strategy['validation_forward_batches'] += config['validation_batches']
        strategy['generated_validation_calls'] += 1
        strategy['generated_validation_cases'] += int(validation['chat'].get('cases', len(validation['chat'].get('rows', []))))
        strategy['generated_validation_output_bytes'] += sum(len(str(row.get('generated', row.get('output', row.get('response', '')))).encode('utf-8'))
                                                           for row in validation['chat'].get('rows', []))
        strategy['history'].append({'updates': strategy['updates'], 'slot': slot, 'candidate_id': candidate['id'],
                                    'genome': copy.deepcopy(candidate['genome']), 'validation': validation})
        if strategy['best'] is None or _quality(validation) > _quality(strategy['best']['validation']):
            weights = _cpu(model.state_dict())
            strategy['best'] = {'model_state': weights, 'weight_sha256': _hash_weights(weights),
                                'genome': copy.deepcopy(candidate['genome']), 'candidate_id': candidate['id'],
                                'updates': strategy['updates'], 'validation': validation}
        if name == 'evolution' and (block + 1) % 4 == 0 and strategy['updates'] < config['max_total_updates']:
            _evolve_population(strategy, models, optimizers, config, device, (block + 1) // 4)
        strategy['evaluation_due'] = False
        return
    block = strategy['updates'] // interval
    slot = 0 if name == 'ordinary' else block % 4
    model, optimizer, candidate = models[slot], optimizers[slot], strategy['candidates'][slot]
    gene = candidate['genome']
    rng = random.Random(config['seed'] + 100000 + strategy['batch_index'])
    intents = sorted(buckets)
    intent = rng.choices(intents, weights=[gene['intent_weights'][key] for key in intents], k=1)[0]
    batch = sample_batch(buckets[intent], config['batch_size'], config['sequence_length'], rng, device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(batch['inputs'])
    mask = batch['loss_mask'].reshape(-1)
    count = int(mask.sum())
    if count == 0:
        raise ValueError('chat training batch has no assistant targets')
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask], batch['targets'].reshape(-1)[mask])
    if not torch.isfinite(loss):
        raise ValueError('nonfinite chat training loss')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), gene['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    strategy['updates'] += 1
    strategy['batch_index'] += 1
    strategy['processed_training_tokens'] += batch['inputs'].numel()
    strategy['supervised_training_tokens'] += count
    candidate['ancestral_updates'] += 1
    strategy['evaluation_due'] = strategy['updates'] % interval == 0


def _publish_best(name, strategy, config, directory, initialization, filename=None):
    best = strategy['best']
    if best is None:
        return
    if any(not torch.isfinite(value).all() for value in best['model_state'].values()):
        raise ValueError('refusing to publish nonfinite champion weights')
    with torch.random.fork_rng(devices=[]):
        model = ByteGRULM(NeuralConfig(**{**config['model'], 'dropout': best['genome']['dropout']}))
    model.load_state_dict(best['model_state'])
    save_neural_checkpoint(directory / (filename or name + '_champion.pt'), model,
        {'training_schema': SCHEMA, 'strategy': name, 'selection_split': 'validation',
         'selection_rule': 'generated rubric score first, lower assistant NLL breaks ties',
         'validation': best['validation'], 'genome': best['genome'], 'global_updates': best['updates'],
         'weight_sha256': best['weight_sha256'], 'initialization': initialization,
         'capability': 'Experimental byte-GRU chat; rubric scores do not establish general conversational ability.'})


def _public_state(state):
    strategies = {}
    for name, strategy in state['strategies'].items():
        strategies[name] = {key: strategy[key] for key in ('status', 'updates', 'evaluation_due',
            'processed_training_tokens', 'supervised_training_tokens', 'validation_forward_batches',
            'generated_validation_calls', 'generated_validation_cases', 'generated_validation_output_bytes',
            'cpu_seconds', 'wall_seconds', 'test')}
        strategies[name]['best_validation'] = strategy['best']['validation'] if strategy['best'] else None
        strategies[name]['genetic_births'] = len(strategy['births'])
    return {'schema': SCHEMA, 'status': state['status'], 'active_strategy': state['active_strategy'],
            'data_fingerprint': state['data_fingerprint'], 'device': state['config']['device'],
            'strategies': strategies, 'error': state.get('error')}


def _checkpoint(state, directory, models=None, optimizers=None, device=None):
    if models is not None:
        _capture(state['strategies'][state['active_strategy']], models, optimizers, device)
    _save(directory / 'chat-training.pt', state)
    _json(directory / 'status.json', _public_state(state))
    population = {name: [{'slot': slot, 'id': candidate['id'], 'parent_id': candidate['parent_id'],
        'genome': candidate['genome'], 'genome_sha256': _gene_hash(candidate['genome']),
        'ancestral_updates': candidate['ancestral_updates'], 'birth_round': candidate['birth_round']}
        for slot, candidate in enumerate(strategy['candidates'])] for name, strategy in state['strategies'].items()}
    _json(directory / 'population.json', {'schema': SCHEMA, 'populations': population,
                                         'gene_bounds': GENE_BOUNDS, 'initialization': state['initialization']})
    if state['active_strategy'] in state['strategies']:
        name = state['active_strategy']
        _publish_best(name, state['strategies'][name], state['config'], directory, state['initialization'])
    _publish_leader(state, directory)


def _publish_leader(state, directory):
    eligible = [name for name in state['config']['strategies'] if state['strategies'][name]['best'] is not None]
    if eligible:
        winner = max(eligible, key=lambda name: _quality(state['strategies'][name]['best']['validation']))
        _publish_best(winner, state['strategies'][winner], state['config'], directory,
                      state['initialization'], filename='best.pt')


def _report(state, directory):
    result = _public_state(state)
    result.update(config=state['config'], initialization=state['initialization'],
                  environment=state['environment'], intent_document_counts=state['intent_document_counts'],
                  gene_bounds=GENE_BOUNDS, selection_rule='lexicographic generated rubric score, then negative assistant NLL',
                  budget_claim='Equal configured optimizer updates, processed training tokens and validation requests per strategy. Curricula can change supervised targets; generation lengths and total device/CPU/wall work are measured, not claimed equal.',
                  timing_scope='Per-strategy initialization, restore, training, evaluation, genetic operators and periodic/live-publication writes. Shared input preparation and final serialization are outside these counters; GPU wall timings synchronize the selected device.',
                  test_policy='Held-out generated-chat test and NLL evaluated only after every requested strategy finishes.',
                  determinism='CPU replay is tested; GPU/ROCm kernel determinism depends on hardware and Torch backend.')
    for name, strategy in state['strategies'].items():
        result['strategies'][name].update(champion=str(directory / (name + '_champion.pt')) if strategy['best'] else None,
            births=strategy['births'], best_genome=strategy['best']['genome'] if strategy['best'] else None)
    values = list(state['strategies'].values())
    result['matched_counts'] = {key: len({strategy[key] for strategy in values}) == 1
                               for key in ('updates', 'processed_training_tokens', 'supervised_training_tokens',
                                   'validation_forward_batches', 'generated_validation_calls', 'generated_validation_cases')}
    result['comparison_complete'] = state['status'] == 'complete' and len(values) > 1
    eligible = [name for name in state['config']['strategies'] if state['strategies'][name]['best'] is not None]
    result['best_checkpoint'] = str(directory / 'best.pt') if eligible else None
    result['best_strategy_by_validation'] = max(eligible, key=lambda name: _quality(state['strategies'][name]['best']['validation'])) if eligible else None
    return result


def train_chat(config, data, run_dir, resume=False):
    config = _config(config)
    device = _device(config)
    documents, buckets, fingerprint = _prepare(data, config)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / 'chat-training.pt'
    if resume:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if state.get('schema') != SCHEMA or state['data_fingerprint'] != fingerprint:
            raise ValueError('chat checkpoint schema or dataset fingerprint mismatch')
        for key in set(config) | set(state['config']):
            if key not in ('max_seconds', 'checkpoint_every', 'checkpoint_seconds') and config.get(key) != state['config'].get(key):
                raise ValueError(f'cannot change {key} while resuming chat training')
        state['config'] = config
        if state['status'] == 'complete':
            report = _report(state, directory)
            _json(directory / 'report.json', report)
            return report
    else:
        if any(directory.iterdir()):
            raise ValueError('chat run directory must be empty; use resume or a new run')
        weights, initialization = _source(config)
        state = {'schema': SCHEMA, 'config': config, 'data_fingerprint': fingerprint,
            'status': 'running', 'active_strategy': config['strategies'][0], 'strategies': {},
            'initialization': initialization, 'intent_document_counts': {key: len(value) for key, value in buckets.items()},
            'environment': {'torch': str(torch.__version__), 'device': str(device),
                'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'rocm': str(torch.version.hip) if torch.version.hip else None}}
        for name in config['strategies']:
            cpu, wall = time.process_time(), time.monotonic()
            state['strategies'][name] = _new_strategy(name, config, buckets, weights, device)
            state['strategies'][name]['cpu_seconds'] += time.process_time() - cpu
            state['strategies'][name]['wall_seconds'] += time.monotonic() - wall
        _checkpoint(state, directory)
    state['status'] = 'running'
    state.pop('error', None)
    stop = [False]
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = last_checkpoint = time.monotonic()
    def should_stop():
        return stop[0] or bool(config['max_seconds'] and time.monotonic() - started >= config['max_seconds'])
    models = optimizers = None
    try:
        for name in config['strategies']:
            strategy = state['strategies'][name]
            state['active_strategy'] = name
            if strategy['updates'] >= config['max_total_updates'] and not strategy['evaluation_due']:
                continue
            strategy['status'] = 'running'
            cpu, wall = time.process_time(), time.monotonic()
            models, optimizers = _runtime(strategy, config, device)
            _sync_device(device)
            strategy['cpu_seconds'] += time.process_time() - cpu
            strategy['wall_seconds'] += time.monotonic() - wall
            while strategy['updates'] < config['max_total_updates'] or strategy['evaluation_due']:
                if should_stop():
                    state['status'] = 'interrupted' if stop[0] else 'time_limit'
                    _checkpoint(state, directory, models, optimizers, device)
                    report = _report(state, directory)
                    _json(directory / 'report.json', report)
                    return report
                cpu, wall = time.process_time(), time.monotonic()
                was_evaluation = strategy['evaluation_due']
                try:
                    _step(name, strategy, models, optimizers, documents, buckets, data, config, device,
                          should_stop=should_stop)
                finally:
                    _sync_device(device)
                    strategy['cpu_seconds'] += time.process_time() - cpu
                    strategy['wall_seconds'] += time.monotonic() - wall
                if was_evaluation:
                    # Publish newly evaluated text behavior immediately, without
                    # copying every optimizer tensor at every small block.
                    cpu, wall = time.process_time(), time.monotonic()
                    _publish_best(name, strategy, config, directory, state['initialization'])
                    _publish_leader(state, directory)
                    _json(directory / 'status.json', _public_state(state))
                    strategy['cpu_seconds'] += time.process_time() - cpu
                    strategy['wall_seconds'] += time.monotonic() - wall
                due = (not was_evaluation and strategy['updates'] % config['checkpoint_every'] == 0)
                if due or time.monotonic() - last_checkpoint >= config['checkpoint_seconds']:
                    cpu, wall = time.process_time(), time.monotonic()
                    _checkpoint(state, directory, models, optimizers, device)
                    strategy['cpu_seconds'] += time.process_time() - cpu
                    strategy['wall_seconds'] += time.monotonic() - wall
                    last_checkpoint = time.monotonic()
            strategy['status'] = 'trained'
            _checkpoint(state, directory, models, optimizers, device)
            models = optimizers = None
        # Only completed, selected champions see test labels or generated cases.
        test_docs = prepare_documents(data, 'test')
        if max(len(doc.tokens) - 1 for doc in test_docs) > config['sequence_length']:
            raise ValueError('test conversations do not fit configured full-context length')
        for name in config['strategies']:
            strategy = state['strategies'][name]
            state['active_strategy'] = name
            if should_stop():
                raise InterruptedError('chat run stopped before final test strategy')
            if strategy['test'] is None:
                cpu, wall = time.process_time(), time.monotonic()
                best = strategy['best']
                with torch.random.fork_rng(devices=[]):
                    model = ByteGRULM(NeuralConfig(**{**config['model'], 'dropout': best['genome']['dropout']})).to(device)
                model.load_state_dict(best['model_state'])
                try:
                    strategy['test'] = {'chat': _chat_evaluate(model, data, 'test', config, should_stop=should_stop),
                                        'nll': _nll(model, test_docs, config, device, config['seed'] + 50000,
                                                    should_stop=should_stop)}
                finally:
                    _sync_device(device)
                    strategy['cpu_seconds'] += time.process_time() - cpu
                    strategy['wall_seconds'] += time.monotonic() - wall
                strategy['status'] = 'complete'
                _checkpoint(state, directory)
        state['status'] = 'complete'
        _checkpoint(state, directory)
        report = _report(state, directory)
        _json(directory / 'report.json', report)
        return report
    except InterruptedError:
        state['status'] = 'interrupted' if stop[0] else 'time_limit'
        _checkpoint(state, directory, models, optimizers, device)
        report = _report(state, directory)
        _json(directory / 'report.json', report)
        return report
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        _checkpoint(state, directory, models, optimizers, device)
        raise
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
