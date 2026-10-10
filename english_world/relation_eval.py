"""Held-out relation-answer diagnostics for a neural model, without retrieval.

Facts supply evaluation labels only. The scorer neither feeds a graph lookup
result into the prompt nor replaces generated text with a stored answer.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from .neural_model import USER, END, ASSISTANT, encode_bytes


def _lemma(value):
    return str(value).replace('_', ' ').strip().casefold()


def _graph_positives(graph, facts):
    """Gold-label filtering only; these sets never become model inputs."""
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        raise ValueError('relation graph requires nodes mapping and edges list')
    result, by_edge = {}, defaultdict(list)
    for fact in facts:
        key = (fact['relation'], fact['source'], _lemma(fact.get('source_lemma', '')))
        if key in result:
            continue
        result[key] = {'targets': set(), 'lemmas': set()}
        if fact['relation'] == 'synonym':
            result[key]['targets'].add(fact['source'])
            result[key]['lemmas'].update(_lemma(word) for word in nodes.get(fact['source'], {}).get('lemmas', []))
        else:
            by_edge[(fact['relation'], fact['source'])].append(key)
    for edge in edges:
        for key in by_edge.get((edge.get('relation'), edge.get('source')), []):
            # Lexical relations apply to their specified source word, not every
            # word sharing its sense. Missing fact lemma is handled conservatively.
            if edge.get('source_lemma') and key[2] and _lemma(edge['source_lemma']) != key[2]:
                continue
            target = edge.get('target')
            result[key]['targets'].add(target)
            if edge.get('target_lemma'):
                result[key]['lemmas'].add(_lemma(edge['target_lemma']))
            else:
                result[key]['lemmas'].update(_lemma(word) for word in nodes.get(target, {}).get('lemmas', []))
    return result


def evaluate_relations(model, dataset, split='test', max_cases=128,
                       max_prompt_bytes=384, max_answer_bytes=256, graph_path=None):
    """Score true answers and same-relation distractors from the same held-out split.

    Distractors have different source senses and exclude known positive targets
    or answers for the queried source. Similar answer lengths are preferred.
    This is a descriptive likelihood contrast, not generated-answer accuracy or
    a guarantee that an incomplete graph rules out every possible alternative.
    """
    if split not in ('validation', 'test'):
        raise ValueError('relation evaluation requires validation or test')
    for name, value, upper in (('max_cases', max_cases, 1024),
                               ('max_prompt_bytes', max_prompt_bytes, 4096),
                               ('max_answer_bytes', max_answer_bytes, 4096)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f'{name} must be an integer in 1..{upper}')
    facts = dataset.get('facts', {}).get(split)
    if not isinstance(facts, list) or not facts:
        raise ValueError(f'no relation facts in held-out split {split}')
    groups = defaultdict(list)
    positive_answers, positive_targets = defaultdict(set), defaultdict(set)
    seen_ids = set()
    for fact in facts:
        if not isinstance(fact, dict) or any(not isinstance(fact.get(key), str) or not fact[key]
                                             for key in ('id', 'relation', 'source', 'target', 'question', 'answer')):
            raise ValueError('facts require nonempty id/relation/source/target/question/answer strings')
        if fact['id'] in seen_ids:
            raise ValueError('relation fact IDs must be unique within a split')
        seen_ids.add(fact['id'])
        groups[fact['relation']].append(fact)
        key = (fact['relation'], fact['source'])
        positive_answers[key].add(fact['answer'].strip())
        positive_targets[key].add(fact['target'])
    full_positives = {}
    if graph_path is not None:
        with Path(graph_path).open(encoding='utf-8') as stream:
            full_positives = _graph_positives(json.load(stream), facts)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    def score(question, answer):
        # The exact same question prefix is used for both alternatives. Only
        # answer positions are scored, with causal teacher forcing as in LM loss.
        prefix = [USER] + encode_bytes(question)[:max_prompt_bytes] + [END, ASSISTANT]
        answer_tokens = encode_bytes(answer)[:max_answer_bytes]
        tokens = prefix + answer_tokens
        logits, _ = model(torch.tensor([tokens[:-1]], dtype=torch.long, device=device))
        predictions = logits[0, len(prefix) - 1:len(prefix) - 1 + len(answer_tokens)]
        target = torch.tensor(answer_tokens, dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(predictions, target, reduction='sum')
        return float(loss), len(answer_tokens)

    rows = []
    try:
        with torch.inference_mode():
            for fact in sorted(facts, key=lambda item: item['id'])[:max_cases]:
                key = (fact['relation'], fact['source'])
                correct_nll, count = score(fact['question'], fact['answer'])
                row = {'id': fact['id'], 'relation': fact['relation'], 'source': fact['source'],
                       'target': fact['target'], 'answer_nll': correct_nll, 'answer_bytes': count,
                       'answer_loss': correct_nll / count, 'distractor_id': None,
                       'answer_visible_in_prompt': bool(fact.get('answer_visible_in_prompt', False)),
                       'answer_truncated': len(encode_bytes(fact['answer'])) > max_answer_bytes,
                       'prompt_truncated': len(encode_bytes(fact['question'])) > max_prompt_bytes}
                choices = [candidate for candidate in groups[fact['relation']]
                           if candidate['source'] != fact['source']
                           and candidate['target'] not in positive_targets[key]
                           and candidate['answer'].strip() not in positive_answers[key]]
                if full_positives:
                    known = full_positives[(fact['relation'], fact['source'], _lemma(fact.get('source_lemma', '')))]
                    choices = [candidate for candidate in choices
                               if candidate['target'] not in known['targets'] and not
                               (known['lemmas'] & {_lemma(word) for word in candidate.get('answer_lemmas', [])})]
                if choices:
                    distractor = min(choices, key=lambda candidate:
                        (abs(len(encode_bytes(candidate['answer'])) - len(encode_bytes(fact['answer']))), candidate['id']))
                    wrong_nll, wrong_count = score(fact['question'], distractor['answer'])
                    wrong_loss = wrong_nll / wrong_count
                    row.update(distractor_id=distractor['id'], distractor_source=distractor['source'],
                               distractor_target=distractor['target'], distractor_loss=wrong_loss,
                               distractor_bytes=wrong_count,
                               distractor_truncated=len(encode_bytes(distractor['answer'])) > max_answer_bytes,
                               wrong_minus_correct_loss=wrong_loss - row['answer_loss'],
                               prefers_correct=row['answer_loss'] < wrong_loss)
                rows.append(row)
    finally:
        model.train(was_training)
    answer_bytes = sum(row['answer_bytes'] for row in rows)
    nll = sum(row['answer_nll'] for row in rows)
    contrasts = [row for row in rows if row['distractor_id'] is not None]
    per_relation = {}
    for relation in sorted({row['relation'] for row in rows}):
        selected = [row for row in rows if row['relation'] == relation]
        paired = [row for row in selected if row['distractor_id'] is not None]
        per_relation[relation] = {
            'cases': len(selected), 'contrast_cases': len(paired),
            'answer_loss': sum(row['answer_nll'] for row in selected) / sum(row['answer_bytes'] for row in selected),
            'fraction_preferring_correct': sum(row['prefers_correct'] for row in paired) / len(paired) if paired else None,
        }
    visibility = {}
    for label, visible in (('visible_in_prompt', True), ('not_visible_in_prompt', False)):
        selected = [row for row in rows if row['answer_visible_in_prompt'] == visible]
        paired = [row for row in selected if row['distractor_id'] is not None]
        visibility[label] = {'cases': len(selected), 'contrast_cases': len(paired),
            'answer_loss': sum(row['answer_nll'] for row in selected) / sum(row['answer_bytes'] for row in selected) if selected else None,
            'fraction_preferring_correct': sum(row['prefers_correct'] for row in paired) / len(paired) if paired else None}
    return {'schema': 'golem-relation-evaluation-v1', 'split': split, 'cases': len(rows),
            'answer_bytes': answer_bytes, 'held_out_answer_loss': nll / answer_bytes,
            'truncated_answers': sum(row['answer_truncated'] for row in rows),
            'truncated_prompts': sum(row['prompt_truncated'] for row in rows),
            'held_out_bits_per_byte': nll / answer_bytes / math.log(2),
            'contrast_cases': len(contrasts),
            'fraction_preferring_correct': sum(row['prefers_correct'] for row in contrasts) / len(contrasts) if contrasts else None,
            'mean_wrong_minus_correct_loss': sum(row['wrong_minus_correct_loss'] for row in contrasts) / len(contrasts) if contrasts else None,
            'per_relation': per_relation, 'prompt_visibility': visibility, 'rows': rows,
            'retrieval_used_by_model': False, 'generation_overridden': False,
            'full_graph_used_only_to_exclude_valid_distractors': graph_path is not None,
            'distractor_policy': 'Same held-out partition and relation type; different source sense, known positives excluded, nearest answer length.',
            'interpretation': 'Descriptive teacher-forced answer likelihood and contrast. Frequency, answer wording and incomplete relation coverage can affect it; this is not generated-answer accuracy.'}
