"""Straight-through ternary fine-tuning of Golem's own FP32 byte-GRU.

Float master parameters receive gradients; GRU/output matrix forward values are
ternary with one detached absmean scale per matrix. Embeddings and biases remain
FP32. All computation still uses ordinary floating-point PyTorch kernels. This is
neither native BitNet training nor a packed low-bit inference implementation.
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

from .neural_model import ByteGRULM, load_neural_checkpoint, save_neural_checkpoint
from .neural_data import prepare_documents, sample_batch
from .neural_training import _evaluate, _json, _save


SCHEMA = 'golem-gru-ternary-qat-v1'


def is_ternary_matrix(name, parameter):
    return parameter.ndim == 2 and (name.startswith('gru.weight_') or name == 'output.weight')


def ternary_value(parameter, straight_through=True):
    """Return exact ternary forward values with optional identity backward slope."""
    detached = parameter.detach()
    if not torch.isfinite(detached).all():
        raise ValueError('ternary quantization requires finite weights')
    scale = detached.abs().mean()
    codes = (detached / scale).round().clamp(-1, 1) if scale.item() else torch.zeros_like(detached)
    value = codes * scale
    # Unlike p + (q-p).detach(), this has no rounding cancellation in the
    # forward value: p-p.detach() is exactly zero, but its backward slope is one.
    return value + (parameter - detached) if straight_through else value


def quantized_parameter_dict(model, straight_through=True):
    return {name: ternary_value(parameter, straight_through)
            if is_ternary_matrix(name, parameter) else parameter
            for name, parameter in model.named_parameters()}


def forward_ternary(model, inputs, hidden=None, straight_through=True):
    """Temporarily substitute ternary matrices, restoring registered masters."""
    replacement = quantized_parameter_dict(model, straight_through)
    return torch.func.functional_call(model, replacement, (inputs, hidden), strict=False)


class TernaryForward(nn.Module):
    def __init__(self, master):
        super().__init__()
        self.master = master

    def forward(self, inputs, hidden=None):
        return forward_ternary(self.master, inputs, hidden, straight_through=self.training)


def _fresh_clone(master):
    # GRU internals may cache nonleaf substituted tensors after a training
    # forward. Reconstruct from registered masters, not cached module graphs.
    with torch.random.fork_rng(devices=[]):
        clone = ByteGRULM(master.config)
    clone.load_state_dict(master.state_dict())
    clone.train(master.training)
    return clone


def materialize_ternary(master):
    """Freeze current quantized values and scales without a second quantization."""
    reference = _fresh_clone(master)
    matrices = []
    with torch.no_grad():
        source = dict(master.named_parameters())
        for name, target in reference.named_parameters():
            if not is_ternary_matrix(name, target):
                continue
            parameter = source[name]
            scale = float(parameter.detach().abs().mean())
            target.copy_(ternary_value(parameter, straight_through=False))
            codes = (target / scale).round().to(torch.int8) if scale else torch.zeros_like(target, dtype=torch.int8)
            matrices.append({'name': name, 'shape': list(target.shape), 'scale': scale,
                             'elements': target.numel(), 'negative': int((codes == -1).sum()),
                             'zeros': int((codes == 0).sum()), 'positive': int((codes == 1).sum())})
    reference.eval()
    metadata = {'method': 'straight_through_absmean_ternary_finetuning', 'native_bitnet': False,
                'compute': 'FP32 reference kernels', 'packed_storage': False,
                'already_quantized': True, 'matrices': matrices,
                'preserved_precision': ['embedding weights FP32', 'all biases FP32'],
                'packing_instruction': 'Pack existing matrix codes using these scales; do not apply absmean quantization again.'}
    return reference, metadata


def _configuration(config):
    result = {'max_updates': 200, 'batch_size': 4, 'sequence_length': 128,
              'learning_rate': .0003, 'weight_decay': .01, 'gradient_clip': 1.,
              'threads': 2, 'max_seconds': 180., 'validation_batches': 8,
              'evaluate_every': 50, 'validation_seed': 4101, 'seed': 7}
    result.update(config)
    if 'validation_every' in config:
        if 'evaluate_every' in config and config['evaluate_every'] != config['validation_every']:
            raise ValueError('validation_every and evaluate_every must agree when both are provided')
        result['evaluate_every'] = config['validation_every']
    for name, low, high in (('max_updates', 1, 10000), ('batch_size', 1, 128),
                           ('sequence_length', 2, 4096), ('threads', 1, 2),
                           ('validation_batches', 1, 128), ('evaluate_every', 1, 10000)):
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    for name in ('learning_rate', 'weight_decay', 'gradient_clip', 'max_seconds'):
        result[name] = float(result[name])
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 < result['max_seconds'] <= 180 or not 0 < result['learning_rate'] <= .1 or result['gradient_clip'] == 0:
        raise ValueError('max_seconds must be (0,180], learning_rate (0,.1], gradient_clip positive')
    return result


def _fingerprint(documents):
    material = {split: [{'tokens': list(doc.tokens), 'loss_mask': list(doc.loss_mask), 'kind': doc.kind}
                        for doc in docs] for split, docs in documents.items()}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def train_ternary(checkpoint, data, run_dir, config):
    """Fine-tune float masters through ternary forward passes; select on validation.

    Test data is neither prepared nor fingerprinted here. The runtime limit is
    checked between updates; an active update/evaluation and atomic final writes
    can add a small completion grace period. The full final master/optimizer/RNG
    state is saved, alongside the best already-quantized float reference.
    """
    config = _configuration(config)
    wall_start, cpu_start = time.monotonic(), time.process_time()
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('QAT run directory must be empty')
    master, source_metadata = load_neural_checkpoint(checkpoint)
    torch.manual_seed(config['seed'])
    rng = random.Random(config['seed'])
    documents = {split: prepare_documents(data, split) for split in ('train', 'validation')}
    fingerprint = _fingerprint(documents)
    optimizer = torch.optim.AdamW(master.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    quantized = TernaryForward(master)
    original = _evaluate(master, documents['validation'], config,
                         config['validation_batches'], config['validation_seed'])
    ptq = _evaluate(quantized, documents['validation'], config,
                    config['validation_batches'], config['validation_seed'])
    best = {'update': 0, 'metrics': ptq, 'master': copy.deepcopy(master.state_dict())}
    history = [{'update': 0, 'validation': ptq}]
    updates = loss_tokens = processed_tokens = 0
    status = 'running'
    error = None
    stop = [False]
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    try:
        while updates < config['max_updates']:
            if stop[0] or time.monotonic() - wall_start >= config['max_seconds']:
                status = 'interrupted' if stop[0] else 'time_limit'
                break
            batch = sample_batch(documents['train'], config['batch_size'], config['sequence_length'], rng)
            quantized.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = quantized(batch['inputs'])
            mask = batch['loss_mask'].reshape(-1)
            count = int(mask.sum())
            if count == 0:
                raise ValueError('QAT batch has no supervised tokens')
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask], batch['targets'].reshape(-1)[mask])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite QAT training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(master.parameters(), config['gradient_clip'])
            optimizer.step()
            updates += 1
            processed_tokens += batch['inputs'].numel()
            loss_tokens += count
            if updates % config['evaluate_every'] == 0 or updates == config['max_updates']:
                metrics = _evaluate(quantized, documents['validation'], config,
                                    config['validation_batches'], config['validation_seed'])
                history.append({'update': updates, 'validation': metrics})
                if metrics['loss'] < best['metrics']['loss']:
                    best = {'update': updates, 'metrics': metrics, 'master': copy.deepcopy(master.state_dict())}
                _json(directory / 'status.json', {'status': status, 'updates': updates,
                      'latest_validation': metrics, 'best_validation': best['metrics'],
                      'best_update': best['update'], 'wall_seconds': time.monotonic() - wall_start})
        else:
            status = 'complete'
    except Exception as exc:
        status = 'failed'
        error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
        raise
    finally:
        try:
            _save(directory / 'qat-training.pt', {'schema': SCHEMA, 'status': status, 'config': config,
                'model_config': master.config.to_dict(), 'model_state': master.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'torch_rng_state': torch.get_rng_state(),
                'sampler_rng_state': rng.getstate(), 'updates': updates,
                'processed_training_tokens': processed_tokens, 'supervised_training_tokens': loss_tokens,
                'data_fingerprint': fingerprint, 'best': best, 'history': history, 'error': error})
            # Loading the selected float masters into a temporary copy preserves
            # the actual final-training-state checkpoint above.
            selected = _fresh_clone(master)
            selected.load_state_dict(best['master'])
            reference, metadata = materialize_ternary(selected)
            metadata.update(selection_split='validation', best_update=best['update'],
                            validation=best['metrics'], source_checkpoint_name=Path(checkpoint).name)
            save_neural_checkpoint(directory / 'qat-reference.pt', reference, metadata)
            report = {'schema': SCHEMA, 'status': status, 'config': config,
                'source_checkpoint_name': Path(checkpoint).name,
                'source_checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'source_metadata': source_metadata, 'data_fingerprint': fingerprint,
                'updates': updates, 'best_update': best['update'],
                'processed_training_tokens': processed_tokens, 'supervised_training_tokens': loss_tokens,
                'evaluation_split': 'validation', 'validation_seed': config['validation_seed'],
                'original_fp32': original, 'post_training_ternary': ptq, 'best_qat': best['metrics'],
                'validation_history': history,
                'recovered_nll': ptq['loss'] - best['metrics']['loss'],
                'remaining_nll_gap_to_original_fp32': best['metrics']['loss'] - original['loss'],
                'quantization': metadata, 'test_evaluated': False,
                'compute': 'FP32 PyTorch kernels with fake-ternary forward weights and float master gradients',
                'native_bitnet': False, 'packed_kernel': False,
                'wall_seconds': time.monotonic() - wall_start, 'cpu_seconds': time.process_time() - cpu_start,
                'error': error}
            _json(directory / 'report.json', report)
            _json(directory / 'status.json', report)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return report
