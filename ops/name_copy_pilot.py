"""Paired, bounded pilot of verified-name loss weighting; no live deployment.

Both arms use the same frozen model, existing TRAIN examples, teacher overlay,
sample stream, optimizer budget and conversational-validation protocol. Only the
opt-in verified_name_weight differs (1 versus 4). No program curriculum is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from english_world import semantic_training as training
from english_world.semantic_model import load_semantic_checkpoint, generate_semantic
from english_world.conversation_eval import evaluate_conversations, score_conversation_turn, _name_claims
from english_world.dialogue_state import extract_user_facts

SCHEMA = 'golem-name-copy-pilot-v1'
WEIGHTS = {'baseline': 1., 'targeted': 4.}
GATES = {'conversation_score_gain': .02, 'unweighted_nll_tolerance': .02,
         'name_recall_gain_vs_baseline_and_parent': .10}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _name(value):
    return unicodedata.normalize('NFC', value).casefold()


def strict_name_recall(report):
    """Measure existing validation recall turns, not names invented for a probe.

    Full name equality is Unicode-NFC/case-insensitive. A substring, uncertain
    guess, denial, question, or assertion of an additional name receives no credit.
    This is an explicit linguistic diagnostic, not unrestricted semantic judging.
    """
    if report.get('split') != 'validation':
        raise ValueError('name recall gating accepts conversation VALIDATION only')
    cases, seen = [], set()
    for episode in report.get('rows', []):
        previous = []
        for row in episode.get('rows', []):
            if row.get('act') == 'name_recall_known':
                frame = row['frame']
                facts = extract_user_facts(row['input_history'])
                expected = frame.get('user_name')
                if (not isinstance(expected, str) or not expected or
                        facts != {'user_name': expected, 'source_user_turn': frame.get('source_user_turn')} or
                        row.get('expected_user_name') != expected or frame.get('act') != 'name_recall_known'):
                    raise ValueError('validation recall frame contradicts explicit USER evidence')
                case_id = (episode['id'], row['turn_index'])
                if case_id in seen:
                    raise ValueError('duplicate validation recall case')
                seen.add(case_id)
                text = row['generated']
                scored = score_conversation_turn(text, frame, previous)
                claims = _name_claims(text, 'name_recall_known')
                bare_exact = _name(text.strip(' \t\r\n.!\"\'')) == _name(expected)
                if bare_exact and not claims:
                    claims = [{'value': expected, 'kind': 'bare_exact_name', 'evidence': text}]
                all_exact = bool(claims) and all(_name(claim['value']) == _name(expected) for claim in claims)
                uncertain = bool(re.search(r"\b(?:maybe|perhaps|probably|guess|might|could|may be|i think|not|isn't|aren't|don't|cannot)\b", text.casefold()))
                passed = bool(all_exact and not uncertain and '?' not in text and scored['passed'] and
                              scored['relevant_to_request'] and not scored['false_memory_claims'])
                cases.append({'episode_id': episode['id'], 'turn_index': row['turn_index'],
                    'case_id': row.get('case_id'), 'expected_name': expected,
                    'source_user_turn': facts['source_user_turn'], 'generated': text,
                    'claims': claims, 'false_memory_claims': scored['false_memory_claims'],
                    'passed': passed})
            previous.append(row)
    if not cases:
        raise ValueError('fixed validation selection contains no known-name recall targets')
    signature = _digest(sorted((row['episode_id'], row['turn_index'], row['expected_name'], row['source_user_turn'])
                              for row in cases))
    return {'schema': 'golem-exact-name-recall-diagnostic-v1', 'split': 'validation',
        'denominator': len(cases), 'correct': sum(row['passed'] for row in cases),
        'accuracy': sum(row['passed'] for row in cases) / len(cases),
        'false_memory_claims': sum(len(row['false_memory_claims']) for row in cases),
        'case_signature_sha256': signature, 'expected_names': sorted({row['expected_name'] for row in cases}),
        'rows': cases, 'selection': 'All name_recall_known turns in the existing fixed conversation-validation selection.',
        'limitations': 'Small, correlated scenario set. Exact recognized name assertions and heuristic relevance; no general memory or independent-statistical claim.'}


@contextmanager
def observe_sampler():
    """Record the unchanged production sample stream and actual token work."""
    original = training._sample
    digest = hashlib.sha256()
    stats = {'calls': 0, 'examples': 0, 'reply_tokens': 0, 'prefix_tokens': 0,
             'verified_name_tokens': 0, 'examples_with_verified_name': 0}
    def sample(buckets, batch_size, recipe, seed):
        rows = original(buckets, batch_size, recipe, seed)
        digest.update(json.dumps([seed, [row['id'] for row in rows]], separators=(',', ':')).encode())
        stats['calls'] += 1
        stats['examples'] += len(rows)
        for row in rows:
            mask = row.get('verified_name_mask')
            if mask is None or len(mask) != len(row['targets']):
                raise ValueError('name pilot requires explicit aligned verified_name_mask from trainer')
            count = sum(bool(value) for value in mask)
            stats['verified_name_tokens'] += count
            stats['examples_with_verified_name'] += int(count > 0)
            stats['reply_tokens'] += len(row['targets'])
            stats['prefix_tokens'] += len(row['prefix'])
        stats['sample_stream_sha256'] = digest.hexdigest()
        return rows
    training._sample = sample
    try:
        yield stats
    finally:
        training._sample = original


def _quality(report):
    return {key: report[key] for key in ('score', 'false_memory_claims', 'whole_episode_pass_rate')}


def _validation(report):
    return {**_quality(report['rollout']), 'reply_loss': report['teacher_forced']['reply_loss']}


def _nondegraded(value, reference):
    return (value['score'] >= reference['score'] and
            value['false_memory_claims'] <= reference['false_memory_claims'] and
            value['whole_episode_pass_rate'] >= reference['whole_episode_pass_rate'])


def adoption_gate(pairs, frozen):
    decisions = []
    for pair in pairs:
        a, b = pair['baseline'], pair['targeted']
        av, bv, fv = a['validation'], b['validation'], frozen['validation']
        names = [a['name_recall'], b['name_recall'], frozen['name_recall']]
        checks = {
            'full_predeclared_budget': pair['planned_updates'] == 200,
            'complete_equal_updates': a['complete'] and b['complete'] and a['updates'] == b['updates'] == pair['planned_updates'],
            'equal_validation_calls': a['validation_calls'] == b['validation_calls'],
            'same_parent_weights': a['initial_weight_sha256'] == b['initial_weight_sha256'] == frozen['weight_sha256'],
            'same_protocol': a['validation_protocol_hash'] == b['validation_protocol_hash'] == frozen['validation_protocol_hash'],
            'same_training_data': a['data_fingerprint'] == b['data_fingerprint'],
            'intended_weight_comparison': a['verified_name_weight'] == 1. and b['verified_name_weight'] == 4.,
            'same_training_stream': a['sampling']['sample_stream_sha256'] == b['sampling']['sample_stream_sha256'],
            'same_token_work': all(a['sampling'][key] == b['sampling'][key] for key in
                                   ('examples', 'reply_tokens', 'prefix_tokens', 'verified_name_tokens')),
            'conversation_gain': bv['score'] - av['score'] >= GATES['conversation_score_gain'] - 1e-12,
            'zero_validation_false_memory': bv['false_memory_claims'] == 0,
            'baseline_episode_no_drop': bv['whole_episode_pass_rate'] >= av['whole_episode_pass_rate'],
            'baseline_unweighted_nll': bv['reply_loss'] <= av['reply_loss'] + GATES['unweighted_nll_tolerance'] + 1e-12,
            'parent_conversation_no_drop': _nondegraded(bv, fv),
            'parent_unweighted_nll': bv['reply_loss'] <= fv['reply_loss'] + GATES['unweighted_nll_tolerance'] + 1e-12,
            'baseline_regression_no_drop': _nondegraded(b['regression'], a['regression']),
            'parent_regression_no_drop': _nondegraded(b['regression'], frozen['regression']),
            'same_name_recall_cases': len({row['case_signature_sha256'] for row in names}) == 1 and
                len({row['denominator'] for row in names}) == 1 and names[0]['denominator'] > 0,
            'name_gain_vs_baseline': names[1]['accuracy'] - names[0]['accuracy'] >= GATES['name_recall_gain_vs_baseline_and_parent'] - 1e-12,
            'name_gain_vs_parent': names[1]['accuracy'] - names[2]['accuracy'] >= GATES['name_recall_gain_vs_baseline_and_parent'] - 1e-12,
            'zero_name_recall_false_claims': names[1]['false_memory_claims'] == 0,
        }
        decisions.append({'seed': pair['seed'], 'checks': checks, 'passed': all(checks.values())})
    eligible = len(pairs) == 2 and {row['seed'] for row in pairs} == {3401, 3402} and all(row['passed'] for row in decisions)
    return {'eligible_for_controlled_followup': eligible, 'live_release_authorized': False,
            'pairs': decisions, 'thresholds': GATES,
            'policy': 'Both predeclared paired comparisons must pass. No automatic training, configuration or live-chat change.',
            'independence_note': 'Seeds 3401/3402 have nearby, overlapping deterministic sample streams across updates. These are paired robustness checks, not independent statistical replications.'}


def _train_diagnostic(model, data, stopping):
    episodes = [episode for episode in data['episodes']['train'] if not episode.get('known_regression')]
    result = evaluate_conversations(lambda turns, **kwargs: generate_semantic(model, turns, **kwargs),
        {'episodes': {'regression': episodes}}, split='regression', max_episodes=32, seed=7, should_stop=stopping)
    result.update(split='train', source_partition='train', evaluation_kind='explanatory_training_diagnostic',
                  release_eligible=False, selection_policy='Explanation only; never used in checkpoint selection or adoption gate.')
    return result


def run_name_copy_pilot(data_path, init_checkpoint, run_dir, *, teacher_overlay,
                        seeds=(3401, 3402), updates=200, batch_size=8, threads=1,
                        max_seconds=1800., trial_seconds=1000., deadline=None):
    if tuple(seeds) != (3401, 3402):
        raise ValueError('predeclared name-copy seeds are 3401 and 3402')
    if type(updates) is not int or updates not in (4, 200):
        raise ValueError('use four updates for a functional canary or 200 for the frozen pilot')
    if batch_size != 8 or type(batch_size) is not int or threads != 1 or type(threads) is not int:
        raise ValueError('name-copy pilot uses batch eight and one CPU thread')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in (max_seconds, trial_seconds)) or not 0 < max_seconds <= 1800 or not 0 < trial_seconds <= 1000:
        raise ValueError('pilot maximum is 30 minutes, at most 1000 seconds per arm')
    cutoff = time.time() + max_seconds
    if deadline:
        parsed = datetime.fromisoformat(str(deadline).replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError('deadline must explicitly be UTC')
        cutoff = min(cutoff, parsed.timestamp())
    started = time.monotonic()
    def stopping():
        return time.time() >= cutoff - 10
    def check_stop():
        if stopping():
            raise InterruptedError('name-copy pilot deadline reached')
    check_stop()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=False)
    from english_world.conversation_sim import load_conversation_sim
    from english_world.world_replay import apply_repair_overlays
    base = load_conversation_sim(data_path)
    corpus_sha, teacher_sha = _sha(data_path), _sha(teacher_overlay)
    data = apply_repair_overlays(base, [teacher_overlay])
    assert data['samples']['validation'] is base['samples']['validation']
    assert data['samples']['test'] is base['samples']['test']
    assert data['episodes'] is base['episodes']
    checkpoint_sha = _sha(init_checkpoint)
    frozen_path = root / 'frozen-init.pt'
    frozen_path.write_bytes(Path(init_checkpoint).read_bytes())
    if _sha(frozen_path) != checkpoint_sha or _sha(init_checkpoint) != checkpoint_sha:
        raise ValueError('parent changed while freezing')
    torch.set_num_threads(1)
    model, metadata = load_semantic_checkpoint(frozen_path)
    protocol, recipe = metadata.get('validation_protocol', {}), metadata.get('recipe', {})
    if metadata.get('selection_split') != 'validation' or not isinstance(metadata.get('validation'), dict):
        raise ValueError('parent must have recorded conversation-validation selection')
    config = {'strategies': ['ordinary'], 'population_size': 4, 'rounds': 1,
        'max_total_updates': updates, 'batch_size': 8, 'threads': 1, 'device': 'cpu',
        'generation_device': 'cpu', 'evaluate_test_on_complete': False,
        'strategy_schedule': 'round_robin', 'init_checkpoint': str(frozen_path.resolve()),
        'model': model.config.to_dict(), 'dropout': model.config.dropout,
        'meaning_condition_gain': model.meaning_condition_gain,
        'learning_rate': recipe.get('learning_rate', .0005),
        'intent_loss_weight': recipe.get('intent_loss_weight', 1.),
        'first_response_weight': recipe.get('first_response_weight', 3.),
        'verified_name_weight': 1., 'validation_seed': protocol.get('rollout_seed', 40097),
        'validation_nll_seed': protocol.get('nll_seed', 30097),
        'validation_batch_size': protocol.get('validation_batch_size', 8),
        'validation_episodes': protocol.get('validation_episodes', 24),
        'validation_batches': protocol.get('validation_batches', 4),
        'max_context_tokens': protocol.get('max_context_tokens', 512),
        'max_reply_tokens': protocol.get('max_reply_tokens', 128),
        'max_new_tokens': protocol.get('max_new_tokens', 120),
        'checkpoint_every': updates, 'checkpoint_seconds': 1000.,
        'name_pilot_teacher_sha256': teacher_sha}
    cfg = training._config(config)
    _, protocol_hash = training._validation_protocol(data, cfg)
    if metadata.get('validation_protocol_hash') != protocol_hash:
        raise ValueError('parent and pilot must have the exact same validation protocol')
    frozen = {'checkpoint_sha256': checkpoint_sha, 'weight_sha256': training._hash_weights(model.state_dict()),
              'validation_protocol_hash': protocol_hash, 'validation': _validation(metadata['validation']),
              'name_recall': strict_name_recall(metadata['validation']['rollout'])}
    plan = {'schema': SCHEMA, 'seeds': list(seeds), 'updates_per_arm': updates, 'batch_size': 8,
        'weights': WEIGHTS, 'config': config, 'corpus_sha256': corpus_sha, 'teacher_overlay_sha256': teacher_sha,
        'teacher_overlay_policy': 'Same existing teacher overlay applied once to pristine TRAIN for both arms; no new examples or changed labels.',
        'validation_protocol_hash': protocol_hash, 'parent_checkpoint_sha256': checkpoint_sha,
        'parent_weight_sha256': frozen['weight_sha256'], 'gate_thresholds': GATES,
        'arm_order': [{'seed': seed, 'order': ['baseline', 'targeted'] if i == 0 else ['targeted', 'baseline']}
                      for i, seed in enumerate(seeds)],
        'recipe_policy': 'Identical parent scalar recipe and condition gain; reset AdamW and uniform base-act sampler in both arms. Only verified-name target loss factor differs.',
        'scope': 'Existing verified USER-name spans in name_set/name_correction/name_recall_known replies; no invented names, classes, lookup at inference or program examples.',
        'deadline_utc': datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
        'test_evaluated': False, 'live_release_authorized': False,
        'purpose': 'functional_canary' if updates == 4 else 'predeclared_paired_pilot',
        'independence_note': 'Nearby seeds and overlapping update-derived sample seeds do not supply independent statistical replication.'}
    _write(root / 'plan.json', plan)
    result = {'schema': SCHEMA, 'status': 'running', 'plan': plan, 'frozen': frozen, 'pairs': []}
    try:
        frozen['regression'] = evaluate_conversations(lambda turns, **kwargs: generate_semantic(model, turns, **kwargs),
            data, split='regression', max_episodes=4, seed=7, should_stop=stopping)
        _write(root / 'frozen-diagnostics.json', frozen)
        del model
        for i, seed in enumerate(seeds):
            pair = {'seed': seed, 'planned_updates': updates}
            result['pairs'].append(pair)
            for arm in plan['arm_order'][i]['order']:
                check_stop()
                remaining = cutoff - time.time() - 20
                if remaining < 10:
                    raise InterruptedError('insufficient remaining budget for another arm')
                arm_config = {**config, 'seed': seed, 'verified_name_weight': WEIGHTS[arm],
                              'max_seconds': min(trial_seconds, remaining), 'name_pilot_arm': arm}
                arm_dir = root / f'seed-{seed}-{arm}'
                with observe_sampler() as sampling:
                    report = training.train_semantic(arm_config, data, arm_dir)
                strategy = report['strategies']['ordinary']
                if report['status'] != 'complete' or strategy['updates'] != updates:
                    pair[arm] = {'complete': False, 'status': report['status'], 'updates': strategy['updates'], 'sampling': sampling}
                    raise InterruptedError('incomplete arm cannot support matched adoption')
                if sampling['calls'] != updates or sampling['examples'] != updates * 8:
                    raise ValueError('sampled examples differ from the optimizer budget')
                if any(sampling[key] != strategy[key] for key in ('reply_tokens', 'prefix_tokens')):
                    raise ValueError('sampled token accounting differs from trainer')
                if report['config']['verified_name_weight'] != WEIGHTS[arm]:
                    raise ValueError('trainer did not retain the intended name-loss weight')
                check_stop()
                champion, champion_metadata = load_semantic_checkpoint(arm_dir / 'best.pt')
                if champion_metadata['recipe']['verified_name_weight'] != WEIGHTS[arm]:
                    raise ValueError('published champion lacks the intended name-loss recipe')
                validation = strategy['best_validation']
                names = strict_name_recall(validation['rollout'])
                regression = evaluate_conversations(lambda turns, **kwargs: generate_semantic(champion, turns, **kwargs),
                    data, split='regression', max_episodes=4, seed=7, should_stop=stopping)
                _write(arm_dir / 'validation-name-recall.json', names)
                _write(arm_dir / 'known-regression.json', regression)
                pair[arm] = {'complete': True, 'updates': strategy['updates'], 'validation_calls': strategy['validation_calls'],
                    'validation_protocol_hash': report['validation_protocol_hash'],
                    'initial_weight_sha256': report['initialization']['weight_sha256'],
                    'data_fingerprint': report['data_fingerprint'],
                    'checkpoint_sha256': _sha(arm_dir / 'best.pt'), 'verified_name_weight': WEIGHTS[arm],
                    'validation': _validation(validation), 'name_recall': names,
                    'regression': _quality(regression), 'sampling': sampling,
                    'actual_work': {key: strategy[key] for key in ('cpu_seconds', 'wall_seconds', 'reply_tokens', 'prefix_tokens')}}
                # This diagnostic is saved for explanation only; no field enters the gate.
                train_report = _train_diagnostic(champion, data, stopping)
                _write(arm_dir / 'train-explanation.json', train_report)
                pair[arm]['train_explanation'] = {**_quality(train_report),
                    'episodes': train_report['episodes'], 'turns': train_report['turns']}
                del champion
                _write(root / 'pilot-report.json', result)
        result['gate'] = adoption_gate(result['pairs'], frozen)
        result['status'] = 'complete'
    except InterruptedError as exc:
        result.update(status='stopped', reason=str(exc), gate={'eligible_for_controlled_followup': False, 'live_release_authorized': False})
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                      gate={'eligible_for_controlled_followup': False, 'live_release_authorized': False})
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['source_files_unchanged'] = (_sha(data_path) == corpus_sha and _sha(init_checkpoint) == checkpoint_sha and _sha(teacher_overlay) == teacher_sha)
        if not result['source_files_unchanged']:
            result.update(status='invalid_sources_changed', gate={'eligible_for_controlled_followup': False, 'live_release_authorized': False})
        _write(root / 'pilot-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--teacher-overlay', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--updates', type=int, choices=(4, 200), default=200)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--trial-seconds', type=float, default=1000.)
    parser.add_argument('--deadline', help='Explicit UTC cutoff; earlier than the whole-pilot cap if applicable')
    args = parser.parse_args()
    report = run_name_copy_pilot(args.data, args.init_checkpoint, args.run_dir,
        teacher_overlay=args.teacher_overlay, updates=args.updates, max_seconds=args.max_seconds,
        trial_seconds=args.trial_seconds, deadline=args.deadline)
    print(json.dumps({'status': report['status'], 'gate': report.get('gate'), 'wall_seconds': report['wall_seconds']}, indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
