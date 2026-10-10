"""Bounded held-out chat diagnostics using actual, unmodified model generations.

Rubrics are explicit word/phrase checks, NOT semantic accuracy or a human judge.
References and rubric terms are evaluator labels only, never model prompt text.
Counterfactual name pairs test whether different histories produce the expected
different answers. This module does not insert canned responses at inference.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .data import _atomic_json
from .neural_model import USER, ASSISTANT, END, encode_bytes, generate_neural

SCHEMA = 'golem-generated-chat-eval-v1'


def _fold(text: str) -> str:
    return ' '.join(text.replace('\u2019', "'").casefold().split())


def _contains(text: str, phrase: str) -> bool:
    phrase = _fold(phrase)
    return bool(phrase) and re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', _fold(text)) is not None


def prompt_tokens(turns: list[dict], max_prompt_bytes: int = 2048) -> list[int]:
    """Encode complete previous turns and end with a fresh assistant marker."""
    if not isinstance(turns, list) or not turns or not isinstance(turns[-1], dict) or turns[-1].get('role') != 'user':
        raise ValueError('chat evaluation turns must end with a user message')
    result = []
    byte_count = 0
    for turn in turns:
        if not isinstance(turn, dict) or turn.get('role') not in ('user', 'assistant') or not isinstance(turn.get('content'), str):
            raise ValueError('chat turns require user/assistant roles and text content')
        encoded = encode_bytes(turn['content'])
        byte_count += len(encoded)
        result += [USER if turn['role'] == 'user' else ASSISTANT] + encoded + [END]
    if byte_count > max_prompt_bytes:
        raise ValueError('chat prompt exceeds bound; do not silently crop context-use tests')
    return result + [ASSISTANT]


def score_response(response: str, rubric: dict) -> dict:
    """Transparent bounded lexical/format heuristic; no learned or semantic judge."""
    if not isinstance(response, str) or not isinstance(rubric, dict):
        raise ValueError('response must be text and rubric must be an object')
    groups = {}
    for key in ('required_any', 'required_all', 'forbidden'):
        values = rubric.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError('rubric phrases must be nonempty strings in lists')
        groups[key] = values
    if not groups['required_any'] and not groups['required_all']:
        raise ValueError('rubric needs at least one positive phrase criterion')
    bound = rubric.get('max_response_bytes', 120)
    if type(bound) is not int or not 1 <= bound <= 120:
        raise ValueError('rubric max_response_bytes must be in 1..120')
    matched_any = [phrase for phrase in groups['required_any'] if _contains(response, phrase)]
    matched_all = [phrase for phrase in groups['required_all'] if _contains(response, phrase)]
    forbidden = [phrase for phrase in groups['forbidden'] if _contains(response, phrase)]
    words = re.findall(r'\b\w+\b', response.casefold())
    triples = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    repeated_triples = (len(triples) - len(set(triples))) / max(1, len(triples))
    all_same_word = len(words) >= 3 and len(set(words)) == 1
    long_run = re.search(r'(\S)\1{5,}', response) is not None
    controls = sum(not (character.isprintable() or character in '\n\r\t') for character in response)
    checks = {'nonempty_text': bool(response.strip()) and any(character.isalnum() for character in response),
              'valid_decoded_text': '\ufffd' not in response and controls == 0,
              'within_length': len(response.encode('utf-8')) <= bound,
              'not_repetition_spam': not all_same_word and not long_run and repeated_triples < .5,
              'required_any': bool(matched_any) if groups['required_any'] else True,
              'required_all': len(matched_all) == len(groups['required_all']),
              'no_forbidden_phrases': not forbidden}
    components = []
    if groups['required_any']:
        components.append(float(bool(matched_any)))
    if groups['required_all']:
        components.append(len(matched_all) / len(groups['required_all']))
    gate = all(checks[key] for key in ('nonempty_text', 'valid_decoded_text', 'within_length',
                                      'not_repetition_spam', 'no_forbidden_phrases'))
    return {'score': sum(components) / len(components) if gate else 0.,
            'heuristic_pass': all(checks.values()), 'checks': checks,
            'matched_any': matched_any, 'matched_all': matched_all, 'matched_forbidden': forbidden,
            'response_bytes': len(response.encode('utf-8')), 'word_count': len(words),
            'repeated_trigram_fraction': repeated_triples,
            'failures': [key for key, passed in checks.items() if not passed]}


def _cases(dataset: dict, split: str, max_cases: int) -> list[dict]:
    cases = dataset.get('chat_cases', {}).get(split)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f'no generated-chat cases in {split}')
    grouped, identities = {}, set()
    for case in cases:
        if not isinstance(case, dict) or any(not isinstance(case.get(key), str) or not case[key]
                                             for key in ('id', 'intent', 'reference')):
            raise ValueError('chat cases require nonempty id/intent/reference strings')
        if case['id'] in identities:
            raise ValueError('chat case IDs must be unique within a split')
        identities.add(case['id'])
        # Validate rubric without passing its labels to the model.
        score_response('', case.get('rubric'))
        key = ('pair', case['context_pair_id']) if case.get('context_pair_id') else ('case', case['id'])
        grouped.setdefault(key, []).append(case)
    pools = defaultdict(list)
    for key, group in sorted(grouped.items()):
        intents = {case['intent'] for case in group}
        if len(intents) != 1:
            raise ValueError('paired context cases must share one intent')
        if key[0] == 'pair' and len(group) < 2:
            raise ValueError('context pairs must have at least two members in the same split')
        if key[0] == 'pair':
            questions = {_fold(case['turns'][-1]['content']) for case in group}
            if len(questions) != 1:
                raise ValueError('context-use pairs must have an identical final user question')
        pools[next(iter(intents))].append(sorted(group, key=lambda case: case['id']))
    selected = []
    # Round-robin intents; never truncate half of a counterfactual pair.
    while any(pools.values()):
        for intent in sorted(pools):
            if not pools[intent]:
                continue
            group = pools[intent].pop(0)
            if len(selected) + len(group) <= max_cases:
                selected.extend(group)
        if len(selected) == max_cases:
            break
    if not selected:
        raise ValueError('max_cases is too small to include a complete case group')
    return selected


def _reference_score(model, prefix: list[int], reference: str, max_reference_bytes: int = 256) -> dict:
    byte_tokens = encode_bytes(reference)
    answer = byte_tokens[:max_reference_bytes] + [END]
    tokens = prefix + answer
    device = next(model.parameters()).device
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    logits, _ = model(inputs)
    selected = logits[0, len(prefix) - 1:len(prefix) - 1 + len(answer)]
    target = torch.tensor(answer, dtype=torch.long, device=device)
    nll = float(torch.nn.functional.cross_entropy(selected, target, reduction='sum').item())
    return {'reference_nll': nll, 'reference_tokens': len(answer),
            'reference_loss': nll / len(answer), 'reference_truncated': len(byte_tokens) > max_reference_bytes,
            'reference_scoring': 'causal answer UTF-8 bytes plus END only; never used during free generation'}


def evaluate_chat(model, dataset: dict, split: str = 'validation', max_cases: int = 32,
                  max_new_tokens: int = 120, seed: int = 7, temperature: float = 0.,
                  include_reference_nll: bool = True, output: str | Path | None = None,
                  max_prompt_bytes: int = 2048, should_stop=None) -> dict:
    """Evaluate complete cases, or raise InterruptedError without a partial score.

    Optional should_stop() is polled between bounded cases and before publication.
    A cancelled subset must never become a candidate-selection score. The caller
    can checkpoint its unevaluated candidate and retry the full evaluation later.
    """
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be a callable or None')

    def check_stop():
        if should_stop is not None and should_stop():
            raise InterruptedError('chat evaluation interrupted before complete report')

    check_stop()
    if split not in ('validation', 'test'):
        raise ValueError('chat evaluation uses validation or test, never training cases')
    for key, value, upper in (('max_cases', max_cases, 512), ('max_new_tokens', max_new_tokens, 120),
                               ('max_prompt_bytes', max_prompt_bytes, 8192)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f'{key} must be an integer in 1..{upper}')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    cases = _cases(dataset, split, max_cases)
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for case in cases:
                check_stop()
                prefix = prompt_tokens(case['turns'], max_prompt_bytes)
                random_identity = case.get('context_pair_id') or case['id']
                generation_seed = int.from_bytes(hashlib.sha256(f"{seed}:{random_identity}".encode()).digest()[:8], 'big') % (2**63)
                budget = min(max_new_tokens, case['rubric'].get('max_response_bytes', 120))
                generated = generate_neural(model, prefix, max_new_tokens=budget, temperature=temperature,
                                            seed=generation_seed, dialogue=False)
                scored = score_response(generated, case['rubric'])
                row = {'id': case['id'], 'intent': case['intent'], 'family_id': case.get('family_id'),
                       'turns': case['turns'], 'generated': generated, 'reference': case['reference'],
                       'rubric': case['rubric'], 'context_pair_id': case.get('context_pair_id'),
                       'memory_value': case.get('memory_value'), 'generation_seed': generation_seed,
                       'generation_budget_bytes': budget, **scored}
                if include_reference_nll:
                    row.update(_reference_score(model, prefix, case['reference']))
                rows.append(row)
    finally:
        model.train(was_training)
    check_stop()
    intents, paired = defaultdict(list), defaultdict(list)
    for row in rows:
        intents[row['intent']].append(row)
        if row['context_pair_id']:
            paired[row['context_pair_id']].append(row)
    per_intent = {intent: {'cases': len(group), 'score': sum(row['score'] for row in group) / len(group),
                            'heuristic_pass_rate': sum(row['heuristic_pass'] for row in group) / len(group)}
                  for intent, group in sorted(intents.items())}
    context_pairs = []
    for identity, group in sorted(paired.items()):
        expected = [_fold(str(row.get('memory_value') or '')) for row in group]
        eligible = len(group) >= 2 and all(expected) and len(set(expected)) == len(expected)
        distinct_outputs = len({_fold(row['generated']) for row in group}) == len(group)
        passed = eligible and distinct_outputs and all(row['heuristic_pass'] for row in group)
        context_pairs.append({'id': identity, 'case_ids': [row['id'] for row in group],
                              'eligible': eligible, 'distinct_outputs': distinct_outputs,
                              'all_expected_answers_pass': bool(passed)})
    eligible_pairs = [pair for pair in context_pairs if pair['eligible']]
    token_count = sum(row.get('reference_tokens', 0) for row in rows)
    report = {'schema': SCHEMA, 'split': split, 'cases': len(rows), 'max_new_tokens': max_new_tokens,
              'temperature': temperature, 'seed': seed,
              'score': sum(item['score'] for item in per_intent.values()) / len(per_intent),
              'micro_score': sum(row['score'] for row in rows) / len(rows),
              'heuristic_pass_rate': sum(row['heuristic_pass'] for row in rows) / len(rows),
              'reference_loss': sum(row.get('reference_nll', 0.) for row in rows) / token_count if token_count else None,
              'reference_tokens': token_count, 'per_intent': per_intent, 'context_pairs': context_pairs,
              'context_pair_pass_rate': sum(pair['all_expected_answers_pass'] for pair in eligible_pairs) / len(eligible_pairs) if eligible_pairs else None,
              'metric_definition': 'score is macro-average lexical rubric fulfillment across represented intents, with text/repetition/forbidden-phrase gates',
              'limitations': 'Heuristic diagnostics, not semantic accuracy. Keyword inclusion cannot establish correctness, naturalness, understanding, or general conversation quality.',
              'inference_policy': 'Only conversation turns enter generation; no references, rubrics, graph lookup, or canned response substitutions.',
              'selection_policy': 'Predeclare validation ordering by generated score, then lower reference loss; test must not select checkpoints.',
              'rows': rows}
    if output is not None:
        check_stop()
        _atomic_json(Path(output), report)
    return report


def validation_selection_key(report: dict) -> tuple[float, float]:
    """Explicit pre-test ordering for checkpoint selection, never accept test data."""
    if report.get('split') != 'validation':
        raise ValueError('only validation reports may select a checkpoint')
    score = float(report['score'])
    loss = report.get('reference_loss')
    if not math.isfinite(score) or not 0 <= score <= 1 or (loss is not None and not math.isfinite(loss)):
        raise ValueError('invalid validation selection metrics')
    return score, -float(loss) if loss is not None else -math.inf
