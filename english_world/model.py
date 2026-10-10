"""A small, explicitly evolutionary character predictor and composite-copy VM.

This is English-world v1, not the unchanged DNA-2 machine. Its COPY moves up to
64 serialized bytes; its TERM observes the end of the ENTIRE composite record.
Every byte costs one copying unit. Prediction tables are inherited, never fitted
by this module. Corpus-guided proposals are mutations, not gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import re
from typing import Iterable

ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?:;"\'()-\n'
ALPHABET_SET = frozenset(ALPHABET)
CODE_ALPHABET = ''.join(chr(i) for i in range(32, 127)) + '\n\t'
PROGRAM = ('START', 'ALLOC', 'COPY', 'TERM', 'DIVIDE')
FORMAT = 'english-world-genome-v1'
CODE_FORMAT = 'english-world-genome-v2'
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def alphabet_for(domain: str = 'english') -> str:
    if domain == 'english':
        return ALPHABET
    if domain == 'code':
        return CODE_ALPHABET
    raise ValueError('domain must be english or code')


def order_limit_for(domain: str = 'english') -> int:
    alphabet_for(domain)  # Validate even callers that only need the size bound.
    return 16 if domain == 'code' else 5


def normalize_text(text: str, domain: str = 'english') -> str:
    alphabet_for(domain)
    if domain == 'code':
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.encode('ascii', 'backslashreplace').decode('ascii')
        return ''.join(c if c in CODE_ALPHABET else f'\\x{ord(c):02x}' for c in text)
    text = text.translate(str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"',
                                        '\u201d': '"', '\u2013': '-', '\u2014': '-',
                                        '\r': '\n', '\t': ' '})).lower()
    text = ''.join(c if c in ALPHABET_SET else ' ' for c in text)
    return re.sub(r' +', ' ', text)


@dataclass
class Genome:
    max_order: int = 3
    rules: dict[str, dict[str, int]] = field(default_factory=dict)
    reproduction: tuple[str, ...] = PROGRAM
    smoothing: int = 16
    max_contexts: int = 512
    domain: str = 'english'

    def __post_init__(self):
        alphabet = frozenset(alphabet_for(self.domain))
        order_limit = order_limit_for(self.domain)
        if not 1 <= self.max_order <= order_limit:
            raise ValueError(f'max_order must be in 1..{order_limit} for {self.domain}')
        if not 1 <= self.smoothing <= 256 or not 1 <= self.max_contexts <= 4096:
            raise ValueError('invalid smoothing or max_contexts')
        if not 1 <= len(self.reproduction) <= 32 or any(x not in PROGRAM for x in self.reproduction):
            raise ValueError('invalid bounded reproduction program')
        self.reproduction = tuple(self.reproduction)
        if not isinstance(self.rules, dict):
            raise ValueError('rules must be an object')
        if len(self.rules) > self.max_contexts:
            raise ValueError('too many contexts')
        clean = {}
        for context, weights in self.rules.items():
            if not isinstance(context, str) or len(context) > order_limit or any(c not in alphabet for c in context):
                raise ValueError('invalid rule context')
            if not isinstance(weights, dict):
                raise ValueError('rule weights must be an object')
            row = {}
            for char, weight in weights.items():
                if char not in alphabet or len(char) != 1 or type(weight) is not int or not 0 <= weight <= 255:
                    raise ValueError('invalid character weight')
                if weight:
                    row[char] = weight
            if row:
                clean[context] = row
        self.rules = clean

    def to_dict(self) -> dict:
        value = {'format': FORMAT, 'max_order': self.max_order, 'smoothing': self.smoothing,
                'max_contexts': self.max_contexts, 'reproduction': list(self.reproduction),
                'rules': {k: dict(v) for k, v in self.rules.items()}}
        # Deliberately omit domain for English: legacy serialized bytes and hashes
        # are scientific identities and must remain unchanged.
        if self.domain == 'code':
            value['format'] = CODE_FORMAT
            value['domain'] = 'code'
        return value

    @classmethod
    def from_dict(cls, value: dict) -> 'Genome':
        if not isinstance(value, dict):
            raise ValueError('genome must be an object')
        version = value.get('format', FORMAT)
        domain = value.get('domain', 'english')
        if version not in (FORMAT, CODE_FORMAT):
            raise ValueError('unsupported genome format')
        if (version == FORMAT and domain != 'english') or (version == CODE_FORMAT and domain != 'code'):
            raise ValueError('genome format and domain do not match')
        return cls(max_order=int(value.get('max_order', 3)), rules=value.get('rules', {}),
                   reproduction=tuple(value.get('reproduction', PROGRAM)),
                   smoothing=int(value.get('smoothing', 16)), max_contexts=int(value.get('max_contexts', 512)),
                   domain=domain)

    def serialize(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')

    def digest(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()


class Predictor:
    """Longest matching suffix with positive unigram smoothing; no learning."""
    def __init__(self, genome: Genome):
        self.genome = genome
        self.alphabet = alphabet_for(genome.domain)
        self.rows = {k: (v, sum(v.values())) for k, v in genome.rules.items()}
        row, total = self.rows.get('', ({}, 0))
        self.prior = {c: (1 + row.get(c, 0)) / (len(self.alphabet) + total) for c in self.alphabet}

    def row(self, context: str):
        for n in range(min(self.genome.max_order, len(context)), 0, -1):
            r = self.rows.get(context[-n:])
            if r is not None:
                return r
        return None

    def probability(self, context: str, target: str) -> float:
        r = self.row(context)
        if r is None:
            return self.prior[target]
        weights, total = r
        alpha = self.genome.smoothing
        return (weights.get(target, 0) + alpha * self.prior[target]) / (total + alpha)

    def distribution(self, context: str) -> dict[str, float]:
        r = self.row(context)
        if r is None:
            return dict(self.prior)
        weights, total = r
        alpha = self.genome.smoothing
        return {c: (weights.get(c, 0) + alpha * self.prior[c]) / (total + alpha) for c in self.alphabet}


def generate(genome: Genome, prompt: str = '', max_chars: int = 160, seed: int = 1,
             temperature: float = 1.0) -> str:
    if not 0 <= max_chars <= 10000:
        raise ValueError('max_chars must be in 0..10000')
    predictor, rng = Predictor(genome), random.Random(seed)
    alphabet, bound = predictor.alphabet, order_limit_for(genome.domain)
    context = normalize_text(prompt, genome.domain)[-bound:]
    out = []
    for _ in range(max_chars):
        dist = predictor.distribution(context)
        if temperature <= 0:
            char = max(alphabet, key=dist.get)
        else:
            exponent = 1.0 / max(.05, min(temperature, 5.0))
            weights = [dist[c] ** exponent for c in alphabet]
            threshold = rng.random() * sum(weights)
            char = alphabet[-1]
            for c, w in zip(alphabet, weights):
                threshold -= w
                if threshold <= 0:
                    char = c
                    break
        out.append(char)
        context = (context + char)[-bound:]
    return ''.join(out)


def spelling_metrics(sample: str, dictionary: Iterable[str], details: bool = False):
    """Score complete whitespace-delimited tokens, never fragments of garbage."""
    lexicon = dictionary if isinstance(dictionary, (set, frozenset)) else set(dictionary)
    tokens = sample.split()
    # A final plain word may have been cut by max_chars; terminal punctuation or
    # whitespace establishes completion. Mixed garbage remains in the denominator.
    if tokens and sample and not sample[-1].isspace() and sample[-1] not in '.,!?:;"\')':
        tokens.pop()
    cleaned = [token.strip('.,!?:;"\'()-') for token in tokens]
    valid_tokens = [word for word in cleaned if WORD.fullmatch(word) and
                    (len(word) > 1 or word in ('a', 'i')) and word in lexicon]
    valid = set(valid_tokens)
    spelling = len(valid) / max(1, len(tokens))
    repeated = (len(tokens) - len(set(tokens))) / max(1, len(tokens))
    runs = max((len(m.group(0)) for m in re.finditer(r'(.)\1+', sample)), default=0)
    repeated = max(repeated, max(0, runs - 3) / max(1, len(sample)))
    if details:
        return {'spelling_score': spelling, 'repetition_penalty': repeated,
                'completed_tokens': len(tokens), 'valid_word_tokens': len(valid_tokens),
                'unique_valid_words': len(valid)}
    return spelling, repeated


def evaluate(genome: Genome, examples: Iterable[dict], dictionary: Iterable[str] = (),
             generated_chars: int = 64, generation_seed: int = 17) -> dict:
    predictor = Predictor(genome)
    bound = order_limit_for(genome.domain)
    nll, characters = 0.0, 0
    for ex in examples:
        context = normalize_text(ex['context'], genome.domain)[-bound:]
        for target in normalize_text(ex['target'], genome.domain):
            nll -= math.log(predictor.probability(context, target))
            characters += 1
            context = (context + target)[-bound:]
    if not characters:
        raise ValueError('evaluation requires at least one target character')
    loss = nll / characters
    sample = generate(genome, 'def ' if genome.domain == 'code' else 'hello ', generated_chars, generation_seed)
    lexical = ({'spelling_score': 0., 'repetition_penalty': 0., 'completed_tokens': 0,
                'valid_word_tokens': 0, 'unique_valid_words': 0} if genome.domain == 'code' else
               spelling_metrics(sample, dictionary, details=True))
    spelling, repeated = lexical['spelling_score'], lexical['repetition_penalty']
    size = len(genome.serialize())
    prediction_gain = math.log(len(predictor.alphabet)) - loss
    # Spelling contributes at most .10; the main term is held-context prediction.
    fitness = prediction_gain + .10 * spelling - .15 * repeated - .000001 * size
    result = {'loss': loss, 'nll': nll, 'bits_per_character': loss / math.log(2),
            'perplexity': math.exp(loss), 'characters': characters, 'prediction_gain': prediction_gain,
            **lexical, 'fitness': fitness,
            'genome_bytes': size, 'contexts': len(genome.rules), 'sample': sample}
    if genome.domain == 'code':
        result.update({'domain': 'code', 'alphabet_size': len(predictor.alphabet), 'spelling_applicable': False})
    return result


def mutate(genome: Genome, rng: random.Random, train_examples: list[dict] | None = None,
           guided_probability: float = .7, events: int = 4) -> tuple[Genome, dict]:
    child = Genome.from_dict(genome.to_dict())
    alphabet, order_limit = alphabet_for(child.domain), order_limit_for(child.domain)
    report = {'guided_events': 0, 'unguided_events': 0, 'deleted_contexts': 0}
    for _ in range(events):
        guided = bool(train_examples) and rng.random() < guided_probability
        if guided:
            ex = rng.choice(train_examples)
            context = normalize_text(ex['context'], child.domain)
            target = normalize_text(ex['target'], child.domain)
            if not target:
                continue
            n = rng.randrange(min(len(context), child.max_order) + 1)
            key = context[-n:] if n else ''
            char = target[0]
            delta = rng.randint(1, 8)
            report['guided_events'] += 1
        else:
            report['unguided_events'] += 1
            action = rng.random()
            if action < .12:
                child.max_order = max(1, min(order_limit, child.max_order + rng.choice((-1, 1))))
                continue
            if action < .20:
                child.smoothing = max(1, min(256, child.smoothing + rng.choice((-2, 2))))
                continue
            if action < .30 and child.rules:
                del child.rules[rng.choice(sorted(child.rules))]
                report['deleted_contexts'] += 1
                continue
            if child.rules and rng.random() < .8:
                key = rng.choice(sorted(child.rules))
            else:
                key = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(child.max_order + 1)))
            char = rng.choice(alphabet)
            delta = rng.choice((-4, -2, -1, 1, 2, 4))
        if key not in child.rules and len(child.rules) >= child.max_contexts:
            del child.rules[rng.choice(sorted(child.rules))]
            report['deleted_contexts'] += 1
        row = child.rules.setdefault(key, {})
        weight = max(0, min(255, row.get(char, 0) + delta))
        if weight:
            row[char] = weight
        else:
            row.pop(char, None)
        if not row:
            child.rules.pop(key, None)
    return Genome.from_dict(child.to_dict()), report


def reproduce(genome: Genome, mutation_rate: float = 0.0, rng: random.Random | None = None,
              train_examples: list[dict] | None = None, guided_probability: float = .7,
              mutation_events: int = 4, divisions: int = 2, max_steps: int | None = None) -> tuple[Genome, dict]:
    """Execute a bounded reproduction VM, then optionally mutate its copied child.

    ALLOC grants at the END of a 16-instruction tick. COPY cannot move P/Q before
    that grant. TERM skips the following instruction until the full record's NUL
    terminator has just been copied. DIVIDE resets P/Q for the next offspring.
    Two default divisions check repeated fecundity. This is a NEW full-record,
    64-byte COPY rule, not a byte-identical implementation of the DNA-2 baseline.
    """
    if not 0 <= mutation_rate <= 1 or not 1 <= divisions <= 8:
        raise ValueError('invalid mutation probability or division count')
    template = genome.serialize() + b'\x00'
    limit = max_steps if max_steps is not None else 256 + math.ceil(len(template) / 64) * len(genome.reproduction) * divisions * 8
    limit = min(1000000, limit)
    pc = p = q = steps = births = copied = stalled = ticks = 0
    child = None
    pending = False
    first = None
    while steps < limit and births < divisions:
        for _ in range(16):
            if steps >= limit or births >= divisions:
                break
            op = genome.reproduction[pc]
            pc = (pc + 1) % len(genome.reproduction)
            steps += 1
            if op == 'ALLOC' and child is None:
                pending = True
            elif op == 'COPY':
                if child is None:
                    stalled += 1  # P and Q deliberately stay unchanged.
                elif p < len(template):
                    fragment = template[p:p + 64]
                    child.extend(fragment)
                    p += len(fragment)
                    q += len(fragment)
                    copied += len(fragment)
            elif op == 'TERM':
                passed = p == len(template) and p > 0 and template[p - 1] == 0
                if not passed:
                    pc = (pc + 1) % len(genome.reproduction)
            elif op == 'DIVIDE' and child is not None:
                if p == len(template) and bytes(child) == template:
                    first = first or bytes(child[:-1])
                    births += 1
                    p = q = 0  # DNA-2's successful-division rearming rule.
                child = None
                pending = False
        ticks += 1
        if pending and child is None:
            child = bytearray()
            q = 0
            pending = False
    if births < divisions or first is None:
        raise ValueError(f'infertile reproduction program: {births}/{divisions} divisions in {steps} instructions')
    offspring = Genome.from_dict(json.loads(first))
    report = {'model': 'english-world-copy-v1', 'births': births, 'ticks': ticks,
              'instructions': steps, 'bytes_copied': copied, 'offspring_copy_cost': len(template),
              'copy_chunk_bytes': 64, 'stalled_copies': stalled, 'template_pointer_after_divide': p,
              'faithful_before_mutation': offspring.digest() == genome.digest(),
              'guided_events': 0, 'unguided_events': 0, 'deleted_contexts': 0}
    rng = rng or random.Random(0)
    if mutation_rate and rng.random() < mutation_rate:
        offspring, changes = mutate(offspring, rng, train_examples, guided_probability, mutation_events)
        report.update(changes)
    report['faithful_after_mutation'] = offspring.digest() == genome.digest()
    return offspring, report


def self_prediction_metrics(genome: Genome, max_examples: int = 512) -> dict:
    """Measure prediction of one's serialized genome, without rewarding this score.

    The genome is the target, not executable host source. Samples are deterministic
    evenly spaced character positions in canonical ASCII JSON. A changing genome
    changes the target, so this diagnostic MUST NOT be used to select parents or
    champions; fixed held-out source-code examples measure improvement instead.
    """
    if type(max_examples) is not int or not 1 <= max_examples <= 4096:
        raise ValueError('max_examples must be an integer in 1..4096')
    if genome.domain != 'code':
        return {'applicable': False, 'domain': genome.domain,
                'reason': 'English v1 alphabet cannot faithfully represent canonical genome JSON',
                'used_for_fitness': False}
    source = genome.serialize()
    text = source.decode('ascii')
    count = min(max_examples, len(text))
    positions = [i * len(text) // count for i in range(count)]
    predictor = Predictor(genome)
    nll = 0.
    for position in positions:
        context = text[max(0, position - genome.max_order):position]
        nll -= math.log(predictor.probability(context, text[position]))
    loss = nll / count
    return {'applicable': True, 'domain': 'code', 'target': 'canonical_serialized_genome',
            'used_for_fitness': False, 'genome_hash': hashlib.sha256(source).hexdigest(),
            'source_bytes': len(source), 'characters': count, 'max_examples': max_examples,
            'sampling': 'evenly_spaced_character_positions', 'alphabet_size': len(CODE_ALPHABET),
            'loss': loss, 'nll': nll, 'bits_per_character': loss / math.log(2),
            'perplexity': math.exp(loss), 'uniform_loss': math.log(len(CODE_ALPHABET)),
            'prediction_gain': math.log(len(CODE_ALPHABET)) - loss}


def baseline_metrics(train_examples: list[dict], examples: list[dict], order: int = 2,
                     domain: str = 'english') -> dict:
    """Explicit NON-evolutionary empirical-count controls; never used as founders."""
    bound = order_limit_for(domain)
    if not 0 <= order <= bound:
        raise ValueError(f'baseline order must be in 0..{bound}')
    rules: dict[str, dict[str, int]] = {}
    for ex in train_examples:
        context = normalize_text(ex['context'], domain)
        for target in normalize_text(ex['target'], domain):
            for n in range(min(order, len(context)) + 1):
                key = context[-n:] if n else ''
                row = rules.setdefault(key, {})
                row[target] = min(255, row.get(target, 0) + 1)
            context = (context + target)[-bound:]
    keys = sorted(rules, key=lambda k: (-sum(rules[k].values()), k))[:4096]
    empirical = Genome(max_order=max(1, order), rules={k: rules[k] for k in keys}, max_contexts=4096, domain=domain)
    unigram = Genome(rules={'': rules.get('', {})}, domain=domain)
    return {'uniform': evaluate(Genome(domain=domain), examples, generated_chars=0),
            'unigram_count_control': evaluate(unigram, examples, generated_chars=0),
            'ngram_count_control': evaluate(empirical, examples, generated_chars=0)}
