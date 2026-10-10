"""Select verified TRAIN targets from a saved learner rollout, without new labels.

This creates an immutable world_replay overlay; it does not apply it, train a
model, or alter the frozen corpus. A correct predicted act and failed reply are
selection diagnostics only. Learner text is always untrusted context.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from english_world import world_replay as replay
from english_world.conversation_eval import score_conversation_turn
from english_world.semantic_model import ACTS, encode_history

MAX_TEACHER_SAMPLES = 127
FOCUS_ACTS = ('loneliness', 'empathy', 'name_recall_known', 'name_recall_unknown',
              'name_correction', 'name_set', 'correction', 'presence', 'greeting',
              'identity', 'request_help', 'clarify', 'rephrase')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def build_teacher_repairs(report_path, data, checkpoint, output, *,
                          max_samples=MAX_TEACHER_SAMPLES, focus_acts=FOCUS_ACTS,
                          should_stop=None):
    """Return a report with ``path`` after validation by apply_repair_overlay.

Selection is deterministic and balanced round-robin across the requested acts.
First-turn failures, unchanged prior histories, and duplicate sample identities
cannot be represented in the existing replay schema and are never fabricated.
"""
    if type(max_samples) is not int or not 1 <= max_samples <= MAX_TEACHER_SAMPLES:
        raise ValueError('teacher overlay must contain fewer than 128 samples')
    focus_acts = tuple(focus_acts)
    if not focus_acts or len(set(focus_acts)) != len(focus_acts) or any(act not in ACTS for act in focus_acts):
        raise ValueError('focus acts must be distinct known meanings')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    def check_stop():
        if should_stop is not None and should_stop():
            raise InterruptedError('teacher overlay cancelled; no artifact published')
    started = time.monotonic()
    check_stop()
    report_path, checkpoint, output = map(Path, (report_path, checkpoint, output))
    if output.exists() or output.is_symlink():
        raise ValueError('teacher overlay output must be a new immutable artifact')
    if not report_path.is_file() or report_path.stat().st_size > replay.MAX_OVERLAY_BYTES:
        raise ValueError('TRAIN diagnostic report must be a file of at most 16 MiB')
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report)
    if (not isinstance(report, dict) or report.get('split') != 'train' or
            report.get('source_partition', 'train') != 'train'):
        raise ValueError('teacher repairs accept TRAIN diagnostic reports only')
    checkpoint_sha = _sha(checkpoint)
    if report.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('diagnostic and source checkpoint hashes differ')
    base = replay._data(data)
    identity = replay._base_identity(base)
    if report.get('corpus_content_sha256') != identity['declared_content_sha256']:
        raise ValueError('diagnostic belongs to another frozen corpus')
    episodes = replay._episode_index(base)
    traces = report.get('rows')
    if not isinstance(traces, list) or not 1 <= len(traces) <= 256:
        raise ValueError('diagnostic must contain 1..256 complete TRAIN episode traces')
    buckets, rollouts, evidence = defaultdict(list), {}, {}
    counts = Counter()
    for trace in traces:
        check_stop()
        episode_id = trace.get('id') if isinstance(trace, dict) else None
        if episode_id not in episodes or episode_id in rollouts:
            raise ValueError('diagnostic episode is duplicated or outside TRAIN')
        episode = episodes[episode_id]
        if episode.get('known_regression'):
            raise ValueError('known regressions cannot supply teacher repairs')
        source, frames = episode['turns'], replay._frames(episode)
        initial = episode.get('initial_history', [])
        first = episode.get('initial_history_count', len(initial))
        if first != len(initial) or source[:first] != initial or trace.get('initial_history', []) != initial:
            raise ValueError('diagnostic changed the initial source history')
        rows = trace.get('rows')
        indices = [i for i in range(first, len(source)) if source[i]['role'] == 'assistant']
        if not isinstance(rows, list) or [row.get('turn_index') for row in rows] != indices:
            raise ValueError('diagnostic must preserve every source assistant turn in order')
        history, previous, generated = copy.deepcopy(initial), [], []
        cursor = first
        for row, index in zip(rows, indices):
            check_stop()
            while cursor < index:
                if source[cursor]['role'] != 'user':
                    raise ValueError('source episode has unexpected turn order')
                history.append(copy.deepcopy(source[cursor]))
                cursor += 1
            frame = frames[index]
            if (row.get('input_history') != history or row.get('reference') != source[index]['content'] or
                    row.get('frame') != frame or row.get('act') != frame['act'] or
                    row.get('meaning_id') != frame['meaning_id']):
                raise ValueError('diagnostic changed USER facts, source labels, or actual rollout continuity')
            text = row.get('generated')
            if not isinstance(text, str) or len(text.encode('utf-8')) > 512:
                raise ValueError('diagnostic generated reply exceeds its bound')
            scored = score_conversation_turn(text, frame, previous)
            predicted = row.get('predicted_meaning_id')
            correct_meaning = ((type(predicted) is int and predicted == frame['meaning_id']) or
                               (isinstance(predicted, str) and predicted == frame['act']))
            counts['observed_turns'] += 1
            if frame['act'] not in focus_acts:
                counts['skipped_nonfocus'] += 1
            elif not correct_meaning:
                counts['skipped_incorrect_meaning'] += 1
            elif scored['passed']:
                counts['skipped_reply_passed'] += 1
            elif not any(history[j]['role'] == 'assistant' and history[j]['content'] != source[j]['content']
                         for j in range(first, index)):
                counts['skipped_unchanged_history'] += 1
            else:
                try:
                    encode_history(history, max_context_tokens=512, trim_history=False)
                except ValueError:
                    counts['skipped_context_too_long'] += 1
                else:
                    # This rechecks causal USER facts and reconstructs all labels.
                    entry = replay._entry(episode, index, history, checkpoint_sha)
                    buckets[frame['act']].append(entry)
                    evidence[entry['id']] = {'sample_id': entry['id'], 'episode_id': episode_id,
                        'turn_index': index, 'act': frame['act'], 'predicted_meaning_id': predicted,
                        'current_generated_reply': text, 'recomputed_passed': scored['passed'],
                        'recomputed_score': scored['score'], 'false_memory_claims': scored['false_memory_claims']}
            generated.append({'turn_index': index, 'text': text, 'predicted_meaning': predicted})
            previous.append({'generated': text, 'act': frame['act'], **scored})
            history.append({'role': 'assistant', 'content': text})
            cursor = index + 1
        if cursor != len(source) or trace.get('actual_history') != history:
            raise ValueError('diagnostic actual history differs from its complete rollout')
        rollouts[episode_id] = {'episode_id': episode_id, 'generated': generated}
    queues = {act: deque(sorted(buckets[act], key=lambda item: (item['episode_id'], item['target_turn_index'])))
              for act in focus_acts}
    selected, seen = [], set()
    while any(queues.values()) and len(selected) < max_samples:
        for act in focus_acts:
            if queues[act] and len(selected) < max_samples:
                entry = queues[act].popleft()
                if entry['id'] in seen:
                    raise ValueError('duplicate teacher sample identity')
                seen.add(entry['id'])
                selected.append(entry)
    selected_episodes = {entry['episode_id'] for entry in selected}
    counts.update(samples=len(selected), episodes=len(selected_episodes),
                  eligible_samples=sum(len(rows) for rows in buckets.values()))
    artifact = {'schema': replay.SCHEMA, 'base_identity': identity,
        'source_checkpoint_sha256': checkpoint_sha, 'source_checkpoint_name': checkpoint.name,
        'seed': report.get('seed', 7), 'max_episodes': len(traces), 'max_samples': max_samples,
        'samples': selected, 'rollouts': [rollouts[key] for key in sorted(selected_episodes)],
        'policy': 'Teacher-selected non-regression TRAIN replies only. Original USER turns, facts, simulator targets and acts remain unchanged. Correct predicted meaning plus failed reply is a diagnostic filter; generated text is untrusted context, never a target.',
        'counts': dict(counts), 'wall_seconds': time.monotonic() - started,
        'teacher_selection': {'schema': 'golem-teacher-repair-selection-v1',
            'source_report_name': report_path.name, 'source_report_sha256': hashlib.sha256(raw_report).hexdigest(),
            'source_partition': 'train', 'focus_acts': list(focus_acts),
            'selection': 'Recomputed reply failure with correct predicted meaning; deterministic act-balanced unique contexts.',
            'multiplicity_per_unique_context': 1, 'per_act': dict(Counter(row['act'] for row in selected)),
            'evidence': [evidence[row['id']] for row in selected],
            'limitations': 'Phrase-based diagnostic failures guide sampling, not universal semantic judgment. First-turn or unchanged-history failures are omitted. Provenance records supplied learner traces; generation is not rerun by this builder.'}}
    artifact['content_sha256'] = replay._digest(artifact)
    encoded = (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(encoded) > replay.MAX_OVERLAY_BYTES:
        raise ValueError('teacher overlay exceeds the replay artifact size bound')
    check_stop()
    if _sha(checkpoint) != checkpoint_sha or report_path.read_bytes() != raw_report:
        raise ValueError('source checkpoint or diagnostic changed during selection')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        augmented = replay.apply_repair_overlay(base, temporary)
        assert augmented['samples']['validation'] is base['samples']['validation']
        assert augmented['samples']['test'] is base['samples']['test']
        check_stop()
        # Same-directory hard linking publishes atomically without overwriting.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output.resolve()), 'sha256': _sha(output),
            'base_identity': identity, 'source_checkpoint_sha256': checkpoint_sha,
            'source_report_sha256': artifact['teacher_selection']['source_report_sha256'],
            'per_act': artifact['teacher_selection']['per_act'], **dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='Saved explicitly labelled TRAIN rollout JSON')
    parser.add_argument('--data', required=True, help='Pristine frozen conversation corpus')
    parser.add_argument('--checkpoint', required=True, help='Frozen source checkpoint matching the report hash')
    parser.add_argument('--output', required=True, help='New immutable overlay JSON path')
    parser.add_argument('--max-samples', type=int, default=MAX_TEACHER_SAMPLES)
    parser.add_argument('--acts', nargs='+', choices=ACTS, default=list(FOCUS_ACTS))
    args = parser.parse_args()
    print(json.dumps(build_teacher_repairs(args.report, args.data, args.checkpoint, args.output,
        max_samples=args.max_samples, focus_acts=args.acts), indent=2))


if __name__ == '__main__':
    main()
