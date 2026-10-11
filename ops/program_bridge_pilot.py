"""Bounded paired experiment: certified DNA-3 examples versus conversation only.

This never edits the frozen corpus, production trainer, active configurations or
live checkpoint. Program probes are phrasing holdouts over the SAME four inputs,
not evidence of general arithmetic. Only validation selects each arm's champion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from english_world import semantic_training as training
from english_world.semantic_model import ACTS, load_semantic_checkpoint, generate_semantic
from english_world.conversation_eval import evaluate_conversations

SCHEMA = 'golem-program-bridge-pilot-v1'
TRAIN_FAMILIES = {
    'train_add_one': 'Add one to {x} modulo 4. Give the result.',
    'train_increment': 'Increment the two-bit value {x}, wrapping at 4. Give the result.',
    'train_compute': 'Compute ({x} + 1) mod 4 for a two-bit value.',
    'train_next': 'What comes after {x} in the cycle 0, 1, 2, 3? Give the result.',
    'train_program': 'Run the two-bit increment modulo 4 program on input {x}.',
    'train_help': 'Help me add 1 to {x} with the answer reduced modulo 4.',
    'train_wrap': 'For input {x}, increase by one and wrap the result modulo 4.',
    'train_request': 'Please return the two-bit result of adding one to {x} modulo 4.',
}
PROBE_FAMILIES = {
    'probe_clock': 'A four-position counter shows {x}. It advances once and wraps after 3. Which value appears?',
    'probe_successor': 'Find the successor of {x} in arithmetic modulo four. State its numeric value.',
    'probe_remainder': 'Take the remainder after dividing {x} plus one by four. What is it?',
    'probe_register': 'A two-bit register contains {x}. After a single increment with overflow discarded, what does it contain?',
}
GATE = {'conversation_score_gain': .02, 'reply_nll_tolerance': .02,
        'program_probe_gain': .10, 'program_accuracy_floor': .75,
        'mixture_probability': .10}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    # All destinations are within a newly created, dedicated pilot directory.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _load_certificate(path):
    from english_world.program_bridge import load_verified_bridge
    return load_verified_bridge(path)


def prepare_program_curriculum(certificate_path):
    certificate = _load_certificate(certificate_path)
    if certificate.get('schema') != 'golem-dna3-compute-slice-v1' or certificate.get('task') != 'increment_mod4':
        raise ValueError('pilot requires a verified DNA-3 increment_mod4 certificate')
    table = certificate.get('truth_table')
    if not isinstance(table, list) or any(type(value) is not int for value in table) or table != [1, 2, 3, 0]:
        raise ValueError('verified increment table must cover exactly the four two-bit inputs')
    if certificate.get('domain') != [0, 1, 2, 3]:
        raise ValueError('certificate must declare the exact two-bit input domain')
    certificate_sha = certificate.get('certificate_sha256')
    if not isinstance(certificate_sha, str) or not re.fullmatch('[0-9a-f]{64}', certificate_sha):
        raise ValueError('verified certificate must expose its immutable content hash')
    partitions = {}
    for split, families in (('train', TRAIN_FAMILIES), ('program_probe', PROBE_FAMILIES)):
        rows = []
        for family, template in families.items():
            for x, y in enumerate(table):
                prompt = template.format(x=x)
                rows.append({'id': 'bridge-' + _digest([split, family, x, certificate_sha])[:24],
                    'family_id': family, 'split': split, 'input': x, 'expected_output': y,
                    'turns': [{'role': 'user', 'content': prompt}], 'reply': f'The result is {y}.',
                    'act': 'request_help', 'meaning_id': ACTS.index('request_help'),
                    'frame': {'act': 'request_help', 'meaning_id': ACTS.index('request_help'),
                              'user_name': None, 'source_user_turn': None},
                    'program_provenance': {'certificate_sha256': certificate_sha,
                        'program_id': certificate.get('program_id'), 'task': 'increment_mod4',
                        'domain': 'integer inputs 0..3; addition modulo 4'}})
        partitions[split] = rows
    if set(TRAIN_FAMILIES) & set(PROBE_FAMILIES):
        raise ValueError('program phrasing families overlap')
    train_prompts = {row['turns'][0]['content'].casefold() for row in partitions['train']}
    if train_prompts & {row['turns'][0]['content'].casefold() for row in partitions['program_probe']}:
        raise ValueError('program prompts overlap across partitions')
    curriculum = {'schema': 'golem-program-curriculum-v1', 'certificate': certificate,
        'certificate_sha256': certificate_sha, **partitions,
        'split_policy': 'Entire original paraphrase families assigned before expanding the four inputs; no probe family is in TRAIN.',
        'probe_role': 'Held-out program phrasing VALIDATION used for adoption gating; not an untouched final test. Do not tune and rebrand it as fresh evaluation.',
        'scope': 'Human-selected certified computation slice; language distillation over four inputs. No claim that evolution is necessary to discover increment or that phrasing generalization is novel arithmetic.'}
    curriculum['content_sha256'] = _digest(curriculum)
    return curriculum


def augment_training(base, curriculum):
    if base.get('program_bridge_modifier'):
        raise ValueError('program augmentation must start from the unmodified conversational base')
    rows = curriculum['train']
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('only TRAIN program families may augment training')
    existing = {row.get('id') for row in base['samples']['train']}
    if any(row['id'] in existing for row in rows):
        raise ValueError('program IDs collide with base training data')
    result = dict(base)
    result['samples'] = dict(base['samples'])
    result['samples']['train'] = list(base['samples']['train']) + list(rows)
    result['program_bridge_modifier'] = {'curriculum_sha256': curriculum['content_sha256'],
        'certificate_sha256': curriculum['certificate_sha256'], 'train_examples': len(rows),
        'sampling_probability': GATE['mixture_probability']}
    # All episode partitions and both held-out sample partitions remain identical.
    assert result['episodes'] is base['episodes']
    assert result['samples']['validation'] is base['samples']['validation']
    assert result['samples']['test'] is base['samples']['test']
    return result


@contextmanager
def controlled_sampler(program_rows, *, mixed):
    """Patch only this serial pilot process; paired arms keep one base stream.

    The mask RNG is separate from the production sampler and model/dropout RNG.
    Baseline records the same intended mask without replacing its examples.
    """
    original = training._sample
    program_ids = {row['id'] for row in program_rows}
    if not program_rows or len(program_ids) != len(program_rows):
        raise ValueError('program rows must be nonempty with unique identities')
    stats = {'calls': 0, 'examples': 0, 'program_examples': 0, 'base_examples': 0,
             'planned_program_positions': 0, 'program_reply_tokens': 0, 'base_reply_tokens': 0,
             'program_prefix_tokens': 0, 'base_prefix_tokens': 0}
    stream_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    def sample(buckets, batch_size, recipe, seed):
        base_buckets = {act: [row for row in rows if row['id'] not in program_ids] for act, rows in buckets.items()}
        base_buckets = {act: rows for act, rows in base_buckets.items() if rows}
        rows = original(base_buckets, batch_size, recipe, seed)
        rng = random.Random(int(_digest(['program-mixture-v1', seed]), 16))
        planned = [rng.random() < GATE['mixture_probability'] for _ in rows]
        choices = [rng.randrange(len(program_rows)) for _ in rows]
        stream_hash.update(json.dumps([seed, [row['id'] for row in rows]], separators=(',', ':')).encode())
        mask_hash.update(json.dumps([seed, planned, choices], separators=(',', ':')).encode())
        selected = [program_rows[choices[i]] if mixed and planned[i] else row for i, row in enumerate(rows)]
        stats['calls'] += 1
        stats['planned_program_positions'] += sum(planned)
        stats['examples'] += len(selected)
        for row in selected:
            kind = 'program' if row['id'] in program_ids else 'base'
            stats[kind + '_examples'] += 1
            stats[kind + '_reply_tokens'] += len(row['targets'])
            stats[kind + '_prefix_tokens'] += len(row['prefix'])
        stats['base_stream_sha256'] = stream_hash.hexdigest()
        stats['mixture_mask_sha256'] = mask_hash.hexdigest()
        return selected
    training._sample = sample
    try:
        yield stats
    finally:
        training._sample = original
        stats['realized_program_fraction'] = stats['program_examples'] / stats['examples'] if stats['examples'] else 0.


def parse_program_answer(text):
    """Accept one explicit result; reject extra numbers, calculations and prose."""
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r'(?:(?:the )?(?:result|answer)\s*(?:is|:)\s*|it is\s+)?([0-3])[.!]?', text.strip().casefold())
    return int(match.group(1)) if match else None


def evaluate_program_probes(model, rows, should_stop=lambda: False):
    measured = []
    for row in rows:
        if should_stop():
            raise InterruptedError('program diagnostic budget reached')
        response = generate_semantic(model, row['turns'], temperature=0., seed=7, max_new_tokens=64)
        parsed = parse_program_answer(response['text'])
        measured.append({'id': row['id'], 'family_id': row['family_id'], 'input': row['input'],
                         'prompt': row['turns'][0]['content'], 'expected': row['expected_output'],
                         'generated': response['text'], 'parsed_output': parsed,
                         'passed': parsed == row['expected_output']})
    return {'schema': 'golem-program-phrasing-diagnostic-v1', 'cases': len(measured),
            'accuracy': sum(row['passed'] for row in measured) / len(measured), 'rows': measured,
            'correct_by_input': {str(x): sum(row['passed'] for row in measured if row['input'] == x) for x in range(4)},
            'cases_by_input': {str(x): sum(row['input'] == x for row in measured) for x in range(4)},
            'scope': 'Unseen phrasing families for the same four certified inputs; not unseen arithmetic functions.',
            'evaluation_role': 'Program validation diagnostic for the predeclared adoption gate; not the final conversation test.',
            'selection_policy': 'Measured after conversational-validation checkpoint selection; never used to select a checkpoint.'}


def _regression_summary(report):
    return {key: report[key] for key in ('score', 'false_memory_claims', 'whole_episode_pass_rate')}


def _require_parent_protocol(metadata, protocol_hash):
    if metadata.get('validation_protocol_hash') != protocol_hash:
        raise ValueError('parent must have the exact frozen conversation validation protocol hash')


def adoption_gate(pairs, frozen):
    """Predeclared conservative gate: every complete paired seed must pass."""
    results = []
    for pair in pairs:
        a, b = pair['baseline'], pair['mixed']
        av, bv = a['validation'], b['validation']
        checks = {
            'full_pilot_budget': pair['planned_updates'] == 200,
            'complete_equal_updates': a['complete'] and b['complete'] and a['updates'] == b['updates'] == pair['planned_updates'],
            'equal_validation_calls': a['validation_calls'] == b['validation_calls'],
            'same_validation_protocol': a['validation_protocol_hash'] == b['validation_protocol_hash'] == frozen['validation_protocol_hash'],
            'same_initial_weights': a['initial_weight_sha256'] == b['initial_weight_sha256'] == frozen['weight_sha256'],
            'same_base_stream': a['sampling']['base_stream_sha256'] == b['sampling']['base_stream_sha256'],
            'same_mixture_mask': a['sampling']['mixture_mask_sha256'] == b['sampling']['mixture_mask_sha256'],
            'conversation_gain': bv['score'] - av['score'] >= GATE['conversation_score_gain'] - 1e-12,
            'no_false_memory_increase': bv['false_memory_claims'] <= av['false_memory_claims'] and bv['false_memory_claims'] == 0,
            'no_episode_pass_decrease': bv['whole_episode_pass_rate'] >= av['whole_episode_pass_rate'],
            'reply_nll_within_tolerance': bv['reply_loss'] <= av['reply_loss'] + GATE['reply_nll_tolerance'] + 1e-12,
            'program_probe_gain': b['program']['accuracy'] - a['program']['accuracy'] >= GATE['program_probe_gain'] - 1e-12,
            'program_not_constant': b['program']['accuracy'] >= GATE['program_accuracy_floor'] and
                all(b['program'].get('correct_by_input', {}).get(str(x), 0) >= 1 for x in range(4)),
            'parent_program_not_worse': b['program']['accuracy'] >= frozen['program']['accuracy'],
            'parent_conversation_not_worse': bv['score'] >= frozen['validation']['score'] and
                bv['false_memory_claims'] <= frozen['validation']['false_memory_claims'] and
                bv['whole_episode_pass_rate'] >= frozen['validation']['whole_episode_pass_rate'] and
                bv['reply_loss'] <= frozen['validation']['reply_loss'] + 1e-12,
            'frozen_regression_not_worse': b['regression']['score'] >= frozen['regression']['score'] and
                b['regression']['false_memory_claims'] <= frozen['regression']['false_memory_claims'] and
                b['regression']['whole_episode_pass_rate'] >= frozen['regression']['whole_episode_pass_rate'],
            'paired_regression_not_worse': b['regression']['score'] >= a['regression']['score'] and
                b['regression']['false_memory_claims'] <= a['regression']['false_memory_claims'] and
                b['regression']['whole_episode_pass_rate'] >= a['regression']['whole_episode_pass_rate'],
        }
        results.append({'seed': pair['seed'], 'checks': checks, 'passed': all(checks.values())})
    eligible = len(results) >= 2 and len({pair['seed'] for pair in pairs}) == len(pairs) and all(row['passed'] for row in results)
    return {'eligible_for_controlled_adoption': eligible, 'pairs': results, 'thresholds': GATE,
            'policy': 'All checks must pass on both predeclared seeds. This is a small pilot gate, not statistical proof. No active run is changed automatically.'}


def run_program_bridge_pilot(certificate_path, data_path, init_checkpoint, run_dir, *,
                             seeds=(1701, 1702), updates=200, batch_size=8, threads=1,
                             teacher_overlay=None, max_seconds=1800., trial_seconds=1000., deadline=None):
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(type(seed) is not int for seed in seeds):
        raise ValueError('pilot requires exactly two distinct predeclared integer seeds')
    if type(updates) is not int or updates < 4 or updates % 4 or updates > 200:
        raise ValueError('first pilot uses at most 200 updates, divisible by four; no adaptive larger trial')
    if type(threads) is not int or not 1 <= threads <= 2:
        raise ValueError('pilot is limited to one or two CPU threads')
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('pilot batch size must be 1..32')
    if not 0 < max_seconds <= 1800 or not 0 < trial_seconds <= 1000:
        raise ValueError('pilot budget is at most 30 minutes and each trial at most 1000 seconds')
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
            raise InterruptedError('whole pilot deadline reached')
    check_stop()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=False)
    curriculum = prepare_program_curriculum(certificate_path)
    from english_world.conversation_sim import load_conversation_sim
    base = load_conversation_sim(data_path)
    base_file_sha = _sha(data_path)
    teacher_sha = None
    if teacher_overlay:
        from english_world.world_replay import apply_repair_overlays
        teacher_sha = _sha(teacher_overlay)
        base = apply_repair_overlays(base, [teacher_overlay])
    mixed = augment_training(base, curriculum)
    checkpoint_sha = _sha(init_checkpoint)
    frozen_path = root / 'frozen-init.pt'
    frozen_path.write_bytes(Path(init_checkpoint).read_bytes())
    if _sha(frozen_path) != checkpoint_sha or _sha(init_checkpoint) != checkpoint_sha:
        raise ValueError('initial checkpoint changed while freezing')
    torch.set_num_threads(threads)
    model, metadata = load_semantic_checkpoint(frozen_path)
    source_protocol = metadata.get('validation_protocol', {})
    recipe = metadata.get('recipe', {})
    config = {'strategies': ['ordinary'], 'population_size': 4, 'rounds': 1,
        'max_total_updates': updates, 'batch_size': batch_size, 'threads': threads,
        'device': 'cpu', 'generation_device': 'cpu', 'evaluate_test_on_complete': False,
        'strategy_schedule': 'round_robin', 'init_checkpoint': str(frozen_path.resolve()),
        'model': model.config.to_dict(), 'dropout': model.config.dropout,
        'meaning_condition_gain': model.meaning_condition_gain,
        'learning_rate': recipe.get('learning_rate', .0005),
        'intent_loss_weight': recipe.get('intent_loss_weight', 1.),
        'first_response_weight': recipe.get('first_response_weight', 3.),
        'validation_seed': source_protocol.get('rollout_seed', 40097),
        'validation_nll_seed': source_protocol.get('nll_seed', 30097),
        'validation_batch_size': source_protocol.get('validation_batch_size', 8),
        'validation_episodes': source_protocol.get('validation_episodes', 24),
        'validation_batches': source_protocol.get('validation_batches', 4),
        'max_context_tokens': source_protocol.get('max_context_tokens', 512),
        'max_reply_tokens': source_protocol.get('max_reply_tokens', 128),
        'max_new_tokens': source_protocol.get('max_new_tokens', 120),
        'checkpoint_every': updates, 'checkpoint_seconds': 1000.,
        'program_pilot_curriculum_sha256': curriculum['content_sha256'],
        'program_pilot_teacher_overlay_sha256': teacher_sha}
    normalized = training._config(config)
    _, protocol_hash = training._validation_protocol(base, normalized)
    if training._validation_protocol(mixed, normalized)[1] != protocol_hash:
        raise ValueError('program augmentation changed conversation validation')
    _require_parent_protocol(metadata, protocol_hash)
    program_encoded = training.encode_samples({'samples': {'train': curriculum['train']}}, 'train', normalized)
    frozen = {'checkpoint_sha256': checkpoint_sha, 'weight_sha256': training._hash_weights(model.state_dict()),
              'validation_protocol_hash': protocol_hash, 'meaning_condition_gain': model.meaning_condition_gain}
    source_validation = metadata.get('validation')
    if not isinstance(source_validation, dict) or metadata.get('selection_split') != 'validation':
        raise ValueError('pilot parent must have recorded conversational-validation selection')
    frozen['validation'] = {**_regression_summary(source_validation['rollout']),
                            'reply_loss': source_validation['teacher_forced']['reply_loss']}
    plan = {'schema': SCHEMA, 'seeds': list(seeds), 'updates_per_arm': updates, 'batch_size': batch_size,
        'config': config, 'base_file_sha256': base_file_sha, 'teacher_overlay_sha256': teacher_sha,
        'curriculum_sha256': curriculum['content_sha256'], 'frozen': dict(frozen), 'gate_thresholds': GATE,
        'deadline_utc': datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
        'arm_order': [{'seed': seed, 'order': ['baseline', 'mixed'] if i % 2 == 0 else ['mixed', 'baseline']}
                      for i, seed in enumerate(seeds)], 'final_test_evaluation': False,
        'recipe_policy': 'Both arms inherit the frozen parent learning rate, dropout, meaning-loss and first-reply scalar weights and condition gain. Both reset AdamW and use the same uniform base-act sampler; parent evolved act weights are not inherited.',
        'purpose': 'paired_200_update_pilot' if updates == 200 else 'functional_canary_not_adoption_evidence',
        'budget_claim': 'Matched optimizer updates and batch sizes; actual sampled tokens and measured CPU/wall times can differ.',
        'adoption': 'Report only. No active configuration or live checkpoint is modified.'}
    _write(root / 'plan.json', plan)
    _write(root / 'curriculum.json', curriculum)
    result = {'schema': SCHEMA, 'status': 'running', 'plan': plan, 'pairs': [], 'frozen': frozen}
    try:
        frozen['program'] = evaluate_program_probes(model, curriculum['program_probe'], stopping)
        frozen['regression'] = evaluate_conversations(lambda turns, **kwargs: generate_semantic(model, turns, **kwargs),
            base, split='regression', max_episodes=4, seed=7, should_stop=stopping)
        _write(root / 'frozen-diagnostics.json', frozen)
        del model
        for i, seed in enumerate(seeds):
            pair = {'seed': seed, 'planned_updates': updates}
            result['pairs'].append(pair)
            for arm in plan['arm_order'][i]['order']:
                check_stop()
                remaining = cutoff - time.time() - 20
                if remaining < 10:
                    raise InterruptedError('insufficient time for another bounded trial')
                arm_config = {**config, 'seed': seed, 'max_seconds': min(trial_seconds, remaining),
                              'program_pilot_arm': arm}
                arm_data = base if arm == 'baseline' else mixed
                arm_dir = root / f'seed-{seed}-{arm}'
                with controlled_sampler(program_encoded, mixed=arm == 'mixed') as sampling:
                    report = training.train_semantic(arm_config, arm_data, arm_dir)
                strategy = report['strategies']['ordinary']
                if report['status'] != 'complete' or strategy['updates'] != updates:
                    pair[arm] = {'complete': False, 'status': report['status'], 'sampling': sampling,
                                 'updates': strategy['updates']}
                    raise InterruptedError('incomplete trial cannot support a matched comparison')
                if sampling['calls'] != updates or sampling['examples'] != updates * batch_size:
                    raise ValueError('sampled update count differs from planned optimizer budget')
                if sampling['base_reply_tokens'] + sampling['program_reply_tokens'] != strategy['reply_tokens']:
                    raise ValueError('sampler token accounting differs from trainer')
                check_stop()
                champion, _ = load_semantic_checkpoint(arm_dir / 'best.pt')
                probes = evaluate_program_probes(champion, curriculum['program_probe'], stopping)
                regression = evaluate_conversations(lambda turns, **kwargs: generate_semantic(champion, turns, **kwargs),
                    base, split='regression', max_episodes=4, seed=7, should_stop=stopping)
                _write(arm_dir / 'program-probes.json', probes)
                _write(arm_dir / 'known-regression.json', regression)
                validation = strategy['best_validation']
                pair[arm] = {'complete': True, 'updates': strategy['updates'], 'validation_calls': strategy['validation_calls'],
                    'validation_protocol_hash': report['validation_protocol_hash'],
                    'initial_weight_sha256': report['initialization']['weight_sha256'],
                    'checkpoint_sha256': _sha(arm_dir / 'best.pt'),
                    'validation': {**_regression_summary(validation['rollout']),
                                   'reply_loss': validation['teacher_forced']['reply_loss']},
                    'program': {key: probes[key] for key in ('accuracy', 'cases', 'correct_by_input', 'cases_by_input')},
                    'regression': _regression_summary(regression), 'sampling': sampling,
                    'actual_work': {key: strategy[key] for key in ('cpu_seconds', 'wall_seconds', 'reply_tokens', 'prefix_tokens')}}
                del champion
                _write(root / 'pilot-report.json', result)
        result['gate'] = adoption_gate(result['pairs'], frozen)
        result['status'] = 'complete'
    except InterruptedError as exc:
        result.update(status='stopped', reason=str(exc), gate={'eligible_for_controlled_adoption': False})
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                      gate={'eligible_for_controlled_adoption': False})
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['source_files_unchanged'] = _sha(data_path) == base_file_sha and _sha(init_checkpoint) == checkpoint_sha
        if teacher_overlay:
            result['source_files_unchanged'] &= _sha(teacher_overlay) == teacher_sha
        if not result['source_files_unchanged']:
            result.update(status='invalid_sources_changed', gate={'eligible_for_controlled_adoption': False})
        _write(root / 'pilot-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--teacher-overlay')
    parser.add_argument('--seeds', type=int, nargs=2, default=(1701, 1702))
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--trial-seconds', type=float, default=1000.)
    parser.add_argument('--deadline', help='Explicit UTC cutoff; the earlier of this and max-seconds applies')
    args = parser.parse_args()
    report = run_program_bridge_pilot(args.certificate, args.data, args.init_checkpoint, args.run_dir,
        seeds=tuple(args.seeds), updates=args.updates, batch_size=args.batch_size, threads=args.threads,
        teacher_overlay=args.teacher_overlay, max_seconds=args.max_seconds, trial_seconds=args.trial_seconds,
        deadline=args.deadline)
    print(json.dumps({'status': report['status'], 'gate': report.get('gate'), 'wall_seconds': report['wall_seconds']}, indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
