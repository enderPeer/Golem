"""Deployment smoke gate on known chat tasks, NOT a held-out quality benchmark.

Candidates retain the collector's validation ordering. Only a candidate passing
all six single-turn checks AND the last turn of a generated three-turn exchange
is released. Nothing is published here, and inference text is never substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch
from english_world.chat_eval import prompt_tokens, score_response
from english_world.data import _atomic_json
from english_world.neural_model import generate_neural, load_neural_checkpoint

FORBIDDEN_DRIFT = ['favorite', 'favorites', 'favourite', 'favourites', 'name', 'bye', 'goodbye', 'see you']
STATUS_WORDS = ['ready', 'chat', 'here', 'day', 'feelings']
CASES = (
    ('identity', 'who are you', ['golem'], ['favorites', 'bye', 'see you']),
    ('whats_up', 'whats up', STATUS_WORDS, FORBIDDEN_DRIFT),
    ('wellbeing', 'how are you', STATUS_WORDS, FORBIDDEN_DRIFT),
    ('greeting', 'hello', ['hello', 'hi', 'hey'], ['bye', 'goodbye', 'see you']),
    ('capabilities', 'what can you do', ['help', 'answer', 'questions', 'explain', 'chat'], ['favorites', 'bye', 'see you']),
    ('thanks', 'thanks', ['welcome', 'glad', 'happy to help', 'anytime'], ['favorites', 'bye', 'see you']),
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _check(model, case, history=None):
    identity, question, required, forbidden = case
    turns = list(history or []) + [{'role': 'user', 'content': question}]
    generated = generate_neural(model, prompt_tokens(turns), max_new_tokens=120,
                                temperature=0., seed=7, dialogue=False)
    rubric = {'required_any': required, 'required_all': [], 'forbidden': forbidden, 'max_response_bytes': 120}
    scored = score_response(generated, rubric)
    return {'id': identity, 'prompt': question, 'turns': turns, 'generated': generated,
            'rubric': rubric, 'passed': scored['heuristic_pass'], 'checks': scored}, turns


def evaluate_release(model):
    """Return actual outputs, with no changes to generation or model weights."""
    singles = [_check(model, case)[0] for case in CASES]
    history, multi = [], []
    by_id = {case[0]: case for case in CASES}
    for identity in ('greeting', 'identity', 'whats_up'):
        result, turns = _check(model, by_id[identity], history)
        multi.append(result)
        history = turns + [{'role': 'assistant', 'content': result['generated']}]
    pass_count = sum(row['passed'] for row in singles) + int(multi[-1]['passed'])
    return {'passed': pass_count == 7, 'pass_count': pass_count, 'required_checks': 7,
            'single_turns': singles, 'multi_turns': multi,
            'multi_turn_gate': 'last generated whats-up response; earlier generated turns remain in the actual history'}


def run_release_gate(report_path, output, max_candidates=33):
    if type(max_candidates) is not int or not 1 <= max_candidates <= 33:
        raise ValueError('max_candidates must be in 1..33')
    report_path = Path(report_path).resolve()
    collector = json.loads(report_path.read_text(encoding='utf-8'))
    if collector.get('schema') != 'golem-chat-collector-v1':
        raise ValueError('expected a chat collector report')
    torch.set_num_threads(1)
    candidates = sorted(collector.get('candidates', []), key=lambda row: (-row['score'], row['reference_loss'], row['id']))
    checked, chosen = [], None
    for candidate in candidates[:max_candidates]:
        row = {'id': candidate['id'], 'host': candidate.get('host'), 'run_dir': candidate.get('run_dir'),
               'strategy': candidate.get('strategy'), 'updates': candidate.get('updates'),
               'source_sha256': candidate['sha256'], 'validation_score': candidate['score'],
               'validation_reference_loss': candidate['reference_loss']}
        try:
            path = Path(candidate['local_path']).resolve(strict=True)
            if not path.is_relative_to(report_path.parent) or _sha(path) != candidate['sha256']:
                raise ValueError('candidate is outside the collection or its SHA changed')
            model, _ = load_neural_checkpoint(path, map_location='cpu')
            result = evaluate_release(model)
            if _sha(path) != candidate['sha256']:
                raise ValueError('checkpoint changed during release evaluation')
            row.update(result)
            if result['passed']:
                chosen = dict(candidate)
                chosen['release_gate'] = {'schema': 'golem-chat-release-gate-v1', 'passed': True,
                                          'pass_count': 7, 'required_checks': 7,
                                          'checkpoint_sha256': candidate['sha256']}
        except Exception as exc:
            row.update(passed=False, pass_count=0, error=str(exc))
        checked.append(row)
        if chosen is not None:
            break
    destination = Path(output)
    if destination.suffix.lower() != '.json':
        destination = destination / 'release-report.json'
    result = {'schema': 'golem-chat-release-gate-v1', 'collector_report': str(report_path),
              'collector_report_sha256': _sha(report_path), 'chosen': chosen,
              'checked_candidates': len(checked), 'available_candidates': len(candidates),
              'required_checks': 7, 'best_pass_count': max((row['pass_count'] for row in checked), default=0),
              'rows': checked, 'published': False,
              'selection': 'First all-pass candidate in original validation score/reference-loss ordering. None if no candidate passes.',
              'limitation': 'Known-task deployment smoke checks, not held-out semantic accuracy or a guarantee of general chat quality.',
              'inference_policy': 'Greedy CPU model outputs only; no canned answers, substitutions, checkpoint edits, or automatic publication.'}
    _atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='runs/chat-collector/report.json')
    parser.add_argument('--output', default='runs/chat-collector/release-report.json')
    args = parser.parse_args()
    report = run_release_gate(args.report, args.output)
    print(json.dumps({'chosen': report['chosen'], 'checked_candidates': report['checked_candidates'],
                      'best_pass_count': report['best_pass_count'], 'report': str(Path(args.output).resolve())}, indent=2))
    return 0 if report['chosen'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
