"""Meaning-conditioned byte GRU with explicit USER-only fact preprocessing.

The fact prefix is neuro-symbolic memory, not a learned-memory claim. Both the
meaning distribution and reply are learned. Neither gold frames nor reference
answers enter inference, and no response lookup or answer substitution exists.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from .neural_model import (ByteGRULM, NeuralConfig, USER, ASSISTANT, END, PAD,
                           encode_bytes, decode_bytes, load_neural_checkpoint)


ACTS = ('greeting', 'presence', 'identity', 'request_help', 'clarify', 'rephrase',
        'correction', 'loneliness', 'empathy', 'thanks', 'goodbye', 'preferences',
        'name_set', 'name_recall_unknown', 'name_recall_known', 'name_correction',
        'topic_chat', 'uncertainty')
SCHEMA = 'golem-semantic-gru-v1'
MEMORY_POLICY = 'Explicit USER-only verified fact prefix; learned meaning and decoder; no gold frames at inference.'


def _validate_model_size(config):
    e, h, layers, vocab = config.embedding_dim, config.hidden_size, config.num_layers, config.vocab_size
    total = vocab * e + 3 * h * e + 3 * h * h + 6 * h
    total += (layers - 1) * (6 * h * h + 6 * h) + vocab * h + vocab
    total += 2 * len(ACTS) * h + len(ACTS) + 1
    if total > 10_000_000:
        raise ValueError('semantic prototype is limited to ten million parameters')


def _finite_weights(state):
    if not isinstance(state, dict) or any(not torch.is_tensor(value) or not torch.isfinite(value).all()
                                           for value in state.values()):
        raise ValueError('semantic checkpoint contains invalid or nonfinite weights')


def encode_history(turns, max_context_tokens=512, trim_history=False):
    """Use the same verified-fact prefix and complete history in train and chat."""
    from .dialogue_state import render_verified_facts
    if not isinstance(turns, list) or not turns or not isinstance(turns[-1], dict) or turns[-1].get('role') != 'user':
        raise ValueError('semantic history must end with a user message')
    for turn in turns:
        if not isinstance(turn, dict) or turn.get('role') not in ('user', 'assistant') or not isinstance(turn.get('content'), str):
            raise ValueError('history turns require user/assistant roles and string content')
    facts = encode_bytes(render_verified_facts(turns))
    blocks = []
    for turn in turns:
        blocks.append([USER if turn['role'] == 'user' else ASSISTANT] + encode_bytes(turn['content']) + [END])
    total, first = len(facts) + sum(len(block) for block in blocks) + 1, 0
    if trim_history:
        while total > max_context_tokens and first < len(blocks) - 1:
            total -= len(blocks[first])
            first += 1
        # Keep complete recent turns and avoid a dangling initial assistant turn.
        while first < len(blocks) - 1 and turns[first]['role'] == 'assistant':
            total -= len(blocks[first])
            first += 1
    # Keep verified state close to the current request so a long recurrent
    # history need not remember an auxiliary prefix from hundreds of bytes ago.
    tokens = [token for block in blocks[first:-1] for token in block] + facts + blocks[-1] + [ASSISTANT]
    if len(tokens) > max_context_tokens:
        raise ValueError(f'full history plus verified facts requires {len(tokens)} tokens; bound is {max_context_tokens}')
    return tokens


class SemanticGRU(nn.Module):
    def __init__(self, config=None, acts=ACTS):
        super().__init__()
        self.config = NeuralConfig.from_dict(config) if isinstance(config, dict) else config or NeuralConfig()
        _validate_model_size(self.config)
        self.acts = tuple(acts)
        if self.acts != ACTS:
            raise ValueError('semantic-v1 requires the declared ordered 18-act vocabulary')
        self.core = ByteGRULM(self.config)
        self.meaning_head = nn.Linear(self.config.hidden_size, len(self.acts))
        self.meaning_embedding = nn.Embedding(len(self.acts), self.config.hidden_size)
        nn.init.normal_(self.meaning_embedding.weight, std=.02)
        self.condition_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))

    def set_dropout(self, value):
        self.core.set_dropout(value)
        self.config = self.core.config

    def encode(self, prefix, lengths):
        if prefix.ndim != 2 or lengths.ndim != 1 or len(lengths) != prefix.shape[0]:
            raise ValueError('prefix must be B,T and lengths B')
        values = self.core.input_dropout(self.core.embedding(prefix))
        packed = nn.utils.rnn.pack_padded_sequence(values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.core.gru(packed)
        act_logits = self.meaning_head(hidden[-1])
        # Always use predicted probabilities, including training. There is no
        # gold-act conditioning API and therefore no teacher/oracle routing gap.
        probabilities = torch.softmax(act_logits, dim=-1)
        meaning = torch.tanh(probabilities @ self.meaning_embedding.weight)
        top = hidden[-1] + torch.sigmoid(self.condition_logit) * meaning
        conditioned = torch.cat((hidden[:-1], top.unsqueeze(0)), dim=0)
        return conditioned, act_logits

    def forward(self, prefix, lengths, reply_inputs):
        hidden, act_logits = self.encode(prefix, lengths)
        first = self.core.output(hidden[-1]).unsqueeze(1)
        if reply_inputs.shape[1]:
            subsequent, _ = self.core(reply_inputs, hidden)
            logits = torch.cat((first, subsequent), dim=1)
        else:
            logits = first
        return logits, act_logits

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def generate_semantic(model, turns, max_new_tokens=120, temperature=0., seed=7, max_context_tokens=512):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be in 1..512')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    tokens = encode_history(turns, max_context_tokens, trim_history=True)
    device = next(model.parameters()).device
    rng = torch.Generator(device='cpu').manual_seed(seed)
    was_training = model.training
    model.eval()
    generated = []
    try:
        with torch.inference_mode():
            prefix = torch.tensor([tokens], dtype=torch.long, device=device)
            hidden, act_logits = model.encode(prefix, torch.tensor([len(tokens)], dtype=torch.long))
            if not torch.isfinite(act_logits).all():
                raise ValueError('nonfinite predicted meaning logits')
            probabilities = torch.softmax(act_logits[0], dim=-1).cpu()
            if not torch.isfinite(probabilities).all():
                raise ValueError('nonfinite predicted meaning probabilities')
            scores = model.core.output(hidden[-1])[0]
            for _ in range(max_new_tokens):
                if not torch.isfinite(scores).all():
                    raise ValueError('nonfinite semantic generation logits')
                scores = scores.float().clone()
                scores[PAD] = -torch.inf
                if temperature == 0:
                    token = int(scores.argmax())
                else:
                    distribution = torch.softmax(scores / max(.01, temperature), dim=-1).cpu()
                    if not torch.isfinite(distribution).all():
                        raise ValueError('nonfinite semantic generation probabilities')
                    token = int(torch.multinomial(distribution, 1, generator=rng))
                if token in (USER, ASSISTANT, END):
                    break
                generated.append(token)
                logits, hidden = model.core(torch.tensor([[token]], dtype=torch.long, device=device), hidden)
                scores = logits[0, -1]
    finally:
        model.train(was_training)
    return {'text': decode_bytes(generated), 'meaning_id': model.acts[int(probabilities.argmax())],
            'meaning_probabilities': {name: float(probabilities[index]) for index, name in enumerate(model.acts)},
            'memory_policy': MEMORY_POLICY}


def save_semantic_checkpoint(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    _finite_weights(model.state_dict())
    artifact = {'schema': SCHEMA, 'config': model.config.to_dict(), 'acts': list(model.acts),
                'model_state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                'metadata': metadata, 'memory_policy': MEMORY_POLICY}
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def load_semantic_checkpoint(path, map_location='cpu'):
    artifact = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or artifact.get('schema') != SCHEMA:
        raise ValueError('unsupported semantic checkpoint schema')
    _finite_weights(artifact.get('model_state'))
    with torch.random.fork_rng(devices=[]):
        model = SemanticGRU(NeuralConfig.from_dict(artifact['config']), artifact['acts'])
    model.load_state_dict(artifact['model_state'], strict=True)
    model.to(map_location)
    model.eval()
    return model, artifact.get('metadata', {})


def warm_start_semantic(path, config=None):
    """Load our core/semantic artifact without modifying it or inventing labels."""
    artifact = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError('semantic warm-start artifact must be an object')
    _validate_model_size(NeuralConfig.from_dict(artifact.get('config')))
    _finite_weights(artifact.get('model_state'))
    if artifact.get('schema') == SCHEMA:
        model, metadata = load_semantic_checkpoint(path)
    else:
        core, metadata = load_neural_checkpoint(path)
        model = SemanticGRU(core.config)
        model.core.load_state_dict(core.state_dict(), strict=True)
    if config is not None:
        expected = NeuralConfig.from_dict(config) if isinstance(config, dict) else config
        if any(value != expected.to_dict()[key] for key, value in model.config.to_dict().items() if key != 'dropout'):
            raise ValueError('semantic warm-start architecture mismatch')
        model.set_dropout(expected.dropout)
    return model, metadata
