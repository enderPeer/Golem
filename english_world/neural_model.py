"""FP32 byte-GRU reference backend, separate from the legacy evolutionary VM.

This is a conventional recurrent network, not BitNet or a packed ternary model.
Artifacts contain tensors and plain metadata only. Generated text is returned as
data; nothing here executes text or edits the program implementing this model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

PAD = PAD_TOKEN = 256
USER = USER_TOKEN = 257
ASSISTANT = ASSISTANT_TOKEN = 258
END = END_TOKEN = 259
VOCAB_SIZE = 260
NEURAL_SCHEMA = 'golem-byte-gru-v1'


@dataclass(frozen=True)
class NeuralConfig:
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = .1

    def __post_init__(self):
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError('byte-GRU vocabulary must contain 256 bytes and four role tokens')
        for key, upper in (('embedding_dim', 4096), ('hidden_size', 4096), ('num_layers', 16)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{key} must be an integer in 1..{upper}')
        if not isinstance(self.dropout, (float, int)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be finite and in [0, 1)')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> 'NeuralConfig':
        if not isinstance(value, dict):
            raise ValueError('neural config must be an object')
        allowed = set(cls.__dataclass_fields__)
        if set(value) - allowed:
            raise ValueError('unknown neural architecture fields: ' + ', '.join(sorted(set(value) - allowed)))
        return cls(**value)


def encode_bytes(text: str) -> list[int]:
    """Lossless UTF-8 byte encoding; no lowercasing or whitespace normalization."""
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    return list(text.encode('utf-8'))


def decode_bytes(tokens: Iterable[int], errors: str = 'replace') -> str:
    """Decode ordinary bytes; role/padding markers are never printed as text."""
    values = []
    for token in tokens:
        token = int(token)
        if not 0 <= token < VOCAB_SIZE:
            raise ValueError('token outside byte-GRU vocabulary')
        if token < 256:
            values.append(token)
    return bytes(values).decode('utf-8', errors=errors)


class ByteGRULM(nn.Module):
    def __init__(self, config: NeuralConfig | dict | None = None):
        super().__init__()
        self.config = NeuralConfig.from_dict(config) if isinstance(config, dict) else config or NeuralConfig()
        if not isinstance(self.config, NeuralConfig):
            raise ValueError('config must be NeuralConfig or a config object')
        c = self.config
        self.embedding = nn.Embedding(c.vocab_size, c.embedding_dim, padding_idx=PAD)
        self.input_dropout = nn.Dropout(c.dropout)
        self.gru = nn.GRU(c.embedding_dim, c.hidden_size, num_layers=c.num_layers,
                          dropout=c.dropout if c.num_layers > 1 else 0., batch_first=True)
        self.output = nn.Linear(c.hidden_size, c.vocab_size)

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError('input_ids must have shape (batch, nonempty sequence)')
        values, hidden = self.gru(self.input_dropout(self.embedding(input_ids)), hidden)
        return self.output(values), hidden

    def set_dropout(self, value: float):
        from dataclasses import replace
        value = float(value)
        self.config = replace(self.config, dropout=value)
        self.input_dropout.p = value
        self.gru.dropout = value if self.config.num_layers > 1 else 0.

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def create_model(config: NeuralConfig | dict | None = None) -> ByteGRULM:
    return ByteGRULM(config)


def generate_neural(model: ByteGRULM, prompt: str | list[int], max_new_tokens: int = 160,
                    temperature: float = .8, seed: int = 1, dialogue: bool = True) -> str:
    """Prefill once, then carry recurrent state; stop at a turn boundary.

    A token-list prompt is already formatted and used unchanged. String prompts
    default to USER + UTF8(prompt) + END + ASSISTANT. Generation neither updates weights
    nor consumes the caller's training RNG stream.
    """
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 10000:
        raise ValueError('max_new_tokens must be an integer in 0..10000')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if isinstance(prompt, str):
        tokens = encode_bytes(prompt)
        if dialogue:
            tokens = [USER] + tokens + [END, ASSISTANT]
    elif isinstance(prompt, list) and all(type(x) is int and 0 <= x < VOCAB_SIZE for x in prompt):
        tokens = list(prompt)
    else:
        raise ValueError('prompt must be text or a list of valid token IDs')
    if not tokens:
        tokens = [ASSISTANT]
    if len(tokens) > 1000000:
        raise ValueError('prompt exceeds the one-million-byte prefill limit')
    if max_new_tokens == 0:
        return ''
    device = next(model.parameters()).device
    rng = torch.Generator(device='cpu').manual_seed(seed)
    was_training = model.training
    model.eval()
    output = []
    try:
        with torch.inference_mode():
            hidden = None
            # Avoid allocating every prompt logit at once for long documents.
            for start in range(0, len(tokens), 1024):
                inputs = torch.tensor([tokens[start:start + 1024]], dtype=torch.long, device=device)
                logits, hidden = model(inputs, hidden)
            scores = logits[0, -1]
            for _ in range(max_new_tokens):
                scores = scores.float().clone()
                # PAD is an implementation detail, not a valid generated symbol.
                scores[PAD] = -torch.inf
                if temperature == 0:
                    token = int(scores.argmax().item())
                else:
                    probabilities = torch.softmax(scores / max(temperature, .01), dim=-1).cpu()
                    if not torch.isfinite(probabilities).all():
                        raise ValueError('model produced nonfinite generation probabilities')
                    token = int(torch.multinomial(probabilities, 1, generator=rng).item())
                if token in (USER, ASSISTANT, END):
                    break
                output.append(token)
                logits, hidden = model(torch.tensor([[token]], dtype=torch.long, device=device), hidden)
                scores = logits[0, -1]
    finally:
        model.train(was_training)
    return decode_bytes(output)


def _plain_metadata(metadata: dict | None) -> dict:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be a plain object')
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('metadata must contain finite JSON-compatible values only') from exc


def save_neural_checkpoint(path: str | Path, model: ByteGRULM, metadata: dict | None = None) -> Path:
    """Atomically publish CPU tensors plus plain config/metadata, never a module pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {'schema': NEURAL_SCHEMA, 'config': model.config.to_dict(),
                'model_state': {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
                'metadata': _plain_metadata(metadata)}
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(artifact, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def load_neural_checkpoint(path: str | Path, map_location: str | torch.device = 'cpu') -> tuple[ByteGRULM, dict]:
    artifact = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(artifact, dict) or artifact.get('schema') != NEURAL_SCHEMA:
        raise ValueError('unsupported neural checkpoint schema')
    config = NeuralConfig.from_dict(artifact.get('config'))
    metadata = _plain_metadata(artifact.get('metadata'))
    # Construction should not perturb a caller's training RNG when merely loading
    # a model for comparison or interactive inference.
    with torch.random.fork_rng(devices=[]):
        model = ByteGRULM(config)
    model.load_state_dict(artifact['model_state'], strict=True)
    model.to(map_location)
    model.eval()
    return model, metadata


def clone_ternary(model: ByteGRULM) -> tuple[ByteGRULM, dict]:
    """Post-training fake-ternary reference, NOT native BitNet or faster inference.

    Each GRU projection and output matrix is rounded to {-1,0,1} times its own
    mean absolute weight. Embeddings and biases remain unchanged. PyTorch still
    stores and computes floating-point matrices; actual packed kernels are a
    separate experiment and no speed/memory benefit is claimed here.
    """
    clone = copy.deepcopy(model)
    matrices = []
    with torch.no_grad():
        for name, parameter in clone.named_parameters():
            if parameter.ndim != 2 or not (name.startswith('gru.weight_') or name == 'output.weight'):
                continue
            if not torch.isfinite(parameter).all():
                raise ValueError(f'cannot quantize nonfinite weights: {name}')
            scale = parameter.abs().mean()
            quantized = (parameter / scale).round().clamp(-1, 1) if scale.item() else torch.zeros_like(parameter)
            parameter.copy_(quantized * scale)
            matrices.append({'name': name, 'shape': list(parameter.shape), 'elements': parameter.numel(),
                             'scale': float(scale.item()), 'zeros': int((quantized == 0).sum().item()),
                             'negative': int((quantized == -1).sum().item()),
                             'positive': int((quantized == 1).sum().item())})
    return clone, {'method': 'post_training_absmean_ternary', 'native_bitnet': False,
                   'packed_storage': False, 'compute': 'floating_point_reference',
                   'dtype': str(next(clone.parameters()).dtype), 'matrices': matrices,
                   'quantized_parameters': sum(matrix['elements'] for matrix in matrices),
                   'preserved': ['embedding weights', 'all biases'],
                   'note': 'Accuracy experiment only; not a packed kernel or a training method.'}
