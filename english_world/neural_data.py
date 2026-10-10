"""Contiguous document and dialogue batches for the neural reference backend.

Loss masks belong to TOKEN POSITIONS, then shift with targets in sample_batch.
Inputs never contain the answer token at the position used to predict it. Each
sample is from one document and each independent batch starts with fresh state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Iterable

import torch

from .neural_model import ASSISTANT, END, PAD, USER, VOCAB_SIZE, encode_bytes


@dataclass(frozen=True)
class SequenceDocument:
    tokens: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    kind: str = 'text'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.tokens) != len(self.loss_mask) or len(self.tokens) < 2:
            raise ValueError('document requires equally sized tokens/mask and at least two tokens')
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in self.tokens):
            raise ValueError('document contains an invalid byte/role token')
        if any(type(flag) is not bool for flag in self.loss_mask):
            raise ValueError('document loss_mask entries must be booleans')
        if not any(self.loss_mask[1:]):
            raise ValueError('document has no next-token prediction targets')


_ROLE_LINE = re.compile(r'(?im)^(user|assistant):[ \t]*')


def dialogue_document(turns: Iterable[dict | tuple[str, str]]) -> SequenceDocument:
    tokens, mask = [], []
    roles = []
    for turn in turns:
        if isinstance(turn, dict):
            role = turn.get('role')
            content = turn.get('content', turn.get('text', ''))
        else:
            role, content = turn
        role = str(role).lower()
        if role not in ('user', 'assistant') or not isinstance(content, str):
            raise ValueError('dialogue turns require a user/assistant role and string content')
        tokens.append(USER if role == 'user' else ASSISTANT)
        mask.append(False)
        encoded = encode_bytes(content)
        tokens.extend(encoded)
        mask.extend([role == 'assistant'] * len(encoded))
        tokens.append(END)
        mask.append(role == 'assistant')
        roles.append(role)
    if 'user' not in roles or 'assistant' not in roles:
        raise ValueError('dialogue requires both user and assistant turns')
    return SequenceDocument(tuple(tokens), tuple(mask), 'dialogue')


def document_from_text(text: str) -> SequenceDocument:
    if not isinstance(text, str):
        raise ValueError('documents must be strings')
    markers = list(_ROLE_LINE.finditer(text))
    if (markers and not text[:markers[0].start()].strip() and
            {m.group(1).lower() for m in markers} == {'user', 'assistant'}):
        turns = []
        for index, marker in enumerate(markers):
            stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            turns.append((marker.group(1).lower(), text[marker.end():stop].rstrip('\r\n')))
        return dialogue_document(turns)
    tokens = encode_bytes(text) + [END]
    if len(tokens) < 2:
        raise ValueError('empty text cannot form a training document')
    return SequenceDocument(tuple(tokens), tuple([True] * len(tokens)), 'text')


def prepare_documents(dataset: dict, split: str = 'train') -> list[SequenceDocument]:
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    texts = dataset.get('texts', {}).get(split)
    if not isinstance(texts, list) or not texts:
        raise ValueError(f'neural training requires contiguous texts for {split}, not isolated target examples')
    documents = [document_from_text(text) for text in texts if text]
    if not documents:
        raise ValueError(f'no nonempty documents in {split}')
    return documents


def sample_batch(documents: list[SequenceDocument], batch_size: int, sequence_length: int,
                 rng: random.Random, device: str | torch.device = 'cpu') -> dict[str, torch.Tensor]:
    """Draw document-local windows, return target-aligned masks, and pad safely.

    Windows are sampled by a scored target so long user-only segments cannot
    produce an all-masked training batch. A recall document must fit in full: its
    cue may not be silently cropped while its answer is still scored.
    """
    if not documents or type(batch_size) is not int or batch_size < 1:
        raise ValueError('a nonempty document list and positive batch size are required')
    if type(sequence_length) is not int or not 1 <= sequence_length <= 100000:
        raise ValueError('sequence_length must be in 1..100000')
    inputs, targets, masks = [], [], []
    for _ in range(batch_size):
        document = rng.choice(documents)
        if document.kind == 'recall' and len(document.tokens) > sequence_length + 1:
            raise ValueError('sequence_length is too short to retain the entire recall cue and answer')
        if len(document.tokens) <= sequence_length + 1:
            start = 0
        else:
            if document.kind == 'text':
                focus = rng.randrange(1, len(document.tokens))
            else:
                positions = [i for i in range(1, len(document.tokens)) if document.loss_mask[i]]
                focus = rng.choice(positions)
            earliest = max(0, focus - sequence_length)
            latest = min(focus - 1, len(document.tokens) - 2)
            start = rng.randint(earliest, latest)
        end = min(len(document.tokens), start + sequence_length + 1)
        x = list(document.tokens[start:end - 1])
        y = list(document.tokens[start + 1:end])
        loss_mask = list(document.loss_mask[start + 1:end])
        padding = sequence_length - len(x)
        inputs.append(x + [PAD] * padding)
        targets.append(y + [PAD] * padding)
        masks.append(loss_mask + [False] * padding)
    return {'inputs': torch.tensor(inputs, dtype=torch.long, device=device),
            'targets': torch.tensor(targets, dtype=torch.long, device=device),
            'loss_mask': torch.tensor(masks, dtype=torch.bool, device=device)}


def make_recall_documents(count: int = 128, delay: int = 32, seed: int = 1) -> list[SequenceDocument]:
    """Balanced digit recall with shared suffixes and exactly one scored answer.

    Format: USER key:<digit>|<delay distractors>|? ASSISTANT <digit> END.
    Source digit is at position 5, answer at 10+delay. Within each ten-example
    group all distractors/query bytes match and only the remembered digit varies.
    Different seeds give different distractors; no other distractor contains a digit.
    """
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError('count must be in 1..100000')
    if type(delay) is not int or not 0 <= delay <= 10000:
        raise ValueError('delay must be in 0..10000')
    rng = random.Random(seed)
    documents = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    while len(documents) < count:
        filler = ''.join(rng.choice(alphabet) for _ in range(delay))
        digits = list('0123456789')
        rng.shuffle(digits)
        for digit in digits:
            tokens = [USER] + encode_bytes('key:' + digit + '|' + filler + '|?') + [ASSISTANT]
            answer_index = len(tokens)
            tokens += encode_bytes(digit) + [END]
            mask = [False] * len(tokens)
            mask[answer_index] = True
            documents.append(SequenceDocument(tuple(tokens), tuple(mask), 'recall',
                             {'source_digit_index': 5, 'answer_index': answer_index, 'delay': delay}))
            if len(documents) == count:
                break
    return documents
