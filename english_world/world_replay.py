"""TRAIN-only repair contexts from learner rollouts and verified simulator labels.

Learner replies are untrusted history, never training targets or facts. Source
USER turns, target replies, acts and frames must match the frozen TRAIN episodes.
An overlay is a small separate artifact; applying it does not rewrite the corpus.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .dialogue_state import extract_user_facts
from .semantic_model import ACTS, encode_history, generate_semantic, load_semantic_checkpoint
from .neural_training import _json, _checkpoint_hash


SCHEMA = 'golem-world-repair-overlay-v1'
MAX_SAMPLES = 512
MAX_OVERLAY_BYTES = 16 * 1024 * 1024


def _data(value):
    if isinstance(value, dict):
        data = value
    else:
        from .conversation_sim import load_conversation_sim
        data = load_conversation_sim(value)
    if data.get('repair_overlay_provenance'):
        raise ValueError('repair generation/application requires the pristine base corpus, not an accumulated overlay')
    if data.get('schema') != 'golem-conversation-sim-v1':
        raise ValueError('repairs require a frozen conversation-simulation corpus')
    return data


def _digest(value):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def _base_identity(data):
    # Only TRAIN records are inspected for the repair identity. The declared
    # whole-corpus digest comes from the already verified frozen corpus manifest.
    training = {'schema': data.get('schema'), 'meaning_vocab': data.get('meaning_vocab', []),
                'episodes': data.get('episodes', {}).get('train', []),
                'samples': data.get('samples', {}).get('train', [])}
    manifest = data.get('manifest', {})
    return {'train_sha256': _digest(training),
            'declared_content_sha256': manifest.get('content_sha256', manifest.get('content_hash'))}


def _episode_index(data):
    episodes = data.get('episodes', {}).get('train')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('frozen TRAIN episodes are required for repairs')
    result = {}
    for episode in episodes:
        identity = episode.get('id')
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError('TRAIN episode identities must be unique strings')
        if episode.get('split', 'train') != 'train':
            raise ValueError('non-TRAIN episode found in TRAIN partition')
        result[identity] = episode
    return result


def _frames(episode):
    frames = {}
    for frame in episode.get('target_frames', []):
        index = frame.get('turn_index')
        if type(index) is not int or index in frames:
            raise ValueError('episode target frame indices must be unique integers')
        frames[index] = frame
    return frames


def _expected_target(episode, turn_index, history):
    if episode.get('known_regression'):
        raise ValueError('explicit regression episodes are excluded from self-generated repairs')
    turns, frames = episode.get('turns'), _frames(episode)
    if not isinstance(turns, list) or type(turn_index) is not int or turn_index not in frames or not 0 < turn_index < len(turns):
        raise ValueError('repair target is not a verified episode assistant frame')
    if not isinstance(history, list) or not history or any(not isinstance(turn, dict) for turn in history):
        raise ValueError('repair history must be a nonempty list of turns')
    if turns[turn_index].get('role') != 'assistant' or len(history) != turn_index or history[-1].get('role') != 'user':
        raise ValueError('repair history must stop immediately before its verified assistant target')
    initial = episode.get('initial_history', [])
    initial_count = episode.get('initial_history_count', len(initial))
    if initial_count != len(initial) or turns[:initial_count] != initial or history[:initial_count] != initial:
        raise ValueError('repair changed the declared initial history')
    for index, (actual, original) in enumerate(zip(history, turns[:turn_index])):
        if not isinstance(actual, dict) or set(actual) != {'role', 'content'} or not isinstance(actual['content'], str):
            raise ValueError('repair history contains malformed turns')
        if actual['role'] != original.get('role'):
            raise ValueError('repair changed a source turn role')
        if actual['role'] == 'user' and actual['content'] != original.get('content'):
            raise ValueError('repair changed a verified USER request or fact')
        if actual['role'] not in ('user', 'assistant'):
            raise ValueError('invalid history role')
        if actual['role'] == 'assistant' and len(actual['content'].encode('utf-8')) > 512:
            raise ValueError('generated assistant context exceeds its byte bound')
    frame = frames[turn_index]
    act = frame.get('act')
    if act not in ACTS or type(frame.get('meaning_id')) is not int or frame['meaning_id'] != ACTS.index(act):
        raise ValueError('repair frame has inconsistent meaning supervision')
    facts = extract_user_facts(history)
    if facts != {'user_name': frame.get('user_name'), 'source_user_turn': frame.get('source_user_turn')}:
        raise ValueError('repair frame contradicts causal USER-only facts')
    reply = turns[turn_index].get('content')
    if not isinstance(reply, str) or not reply or len(reply.encode('utf-8')) + 1 > 128:
        raise ValueError('verified reply does not fit its bound')
    encode_history(history, max_context_tokens=512, trim_history=False)
    generated_indices = [index for index in range(initial_count, turn_index) if turns[index]['role'] == 'assistant']
    if not generated_indices or not any(history[index]['content'] != turns[index]['content'] for index in generated_indices):
        raise ValueError('repair requires a changed learner-generated prior assistant context')
    return reply, act, frame, generated_indices


def _entry(episode, turn_index, history, checkpoint_sha):
    reply, act, frame, generated_indices = _expected_target(episode, turn_index, history)
    provenance = {'episode_id': episode['id'], 'target_turn_index': turn_index,
                  'source_checkpoint_sha256': checkpoint_sha, 'turns': history}
    return {'id': 'repair-' + _digest(provenance)[:24], 'episode_id': episode['id'],
            'family_id': episode.get('family_id'), 'target_turn_index': turn_index,
            'turns': copy.deepcopy(history), 'reply': reply, 'reference': reply, 'act': act,
            'intent': act, 'meaning_id': frame['meaning_id'], 'frame': copy.deepcopy(frame),
            'generated_assistant_turns': generated_indices,
            'untrusted_assistant_turns': [index for index, turn in enumerate(history) if turn['role'] == 'assistant'],
            'source_checkpoint_sha256': checkpoint_sha,
            'supervision': 'Only verified original simulator reply/act; all learner text is untrusted context.'}


def _rollout_index(artifact, episodes):
    rollouts = artifact.get('rollouts')
    if not isinstance(rollouts, list) or len(rollouts) > 256:
        raise ValueError('invalid bounded repair rollout provenance')
    generated_by_episode = {}
    for rollout in rollouts:
        episode_id = rollout.get('episode_id') if isinstance(rollout, dict) else None
        if episode_id not in episodes or episode_id in generated_by_episode or episodes[episode_id].get('known_regression'):
            raise ValueError('rollout provenance must reference unique non-regression TRAIN episodes')
        generated = {}
        for item in rollout.get('generated', []):
            index = item.get('turn_index') if isinstance(item, dict) else None
            if type(index) is not int or index in generated or index not in _frames(episodes[episode_id]):
                raise ValueError('invalid learner rollout turn provenance')
            if not isinstance(item.get('text'), str) or len(item['text'].encode('utf-8')) > 512:
                raise ValueError('invalid learner rollout text')
            generated[index] = item['text']
        generated_by_episode[episode_id] = generated
    return generated_by_episode


def build_repair_overlay(checkpoint, data, output, seed, max_episodes=32, max_samples=256, should_stop=None):
    """Build one bounded overlay; cancellation never publishes a partial artifact."""
    if type(max_episodes) is not int or not 1 <= max_episodes <= 256:
        raise ValueError('max_episodes must be in 1..256')
    if type(max_samples) is not int or not 1 <= max_samples <= MAX_SAMPLES:
        raise ValueError('max_samples must be in 1..512')
    if type(seed) is not int or (should_stop is not None and not callable(should_stop)):
        raise ValueError('seed must be an integer and should_stop a callable or None')
    output = Path(output)
    if output.exists():
        raise ValueError('repair output must be a new immutable artifact')
    def check_stop():
        if should_stop is not None and should_stop():
            raise InterruptedError('repair rollout cancelled; no partial overlay published')
    check_stop()
    base = _data(data)
    identity = _base_identity(base)
    episodes = _episode_index(base)
    selected = [episode for _, episode in sorted(episodes.items()) if not episode.get('known_regression')]
    random.Random(seed).shuffle(selected)
    selected = selected[:max_episodes]
    checkpoint_sha = _checkpoint_hash(checkpoint)
    check_stop()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    samples, transcripts = [], []
    skipped_too_long = skipped_unchanged = 0
    try:
        model, _ = load_semantic_checkpoint(checkpoint, map_location='cpu')
        for episode in selected:
            check_stop()
            turns = episode['turns']
            frames = _frames(episode)
            initial = episode.get('initial_history', [])
            first = episode.get('initial_history_count', len(initial))
            history = copy.deepcopy(initial)
            generated = []
            for index in range(first, len(turns)):
                check_stop()
                turn = turns[index]
                if turn['role'] == 'user':
                    history.append({'role': 'user', 'content': turn['content']})
                    continue
                if turn['role'] != 'assistant' or index not in frames:
                    raise ValueError('TRAIN rollout has an unframed assistant target')
                changed = any(history[j]['role'] == 'assistant' and history[j]['content'] != turns[j]['content']
                              for j in range(first, index))
                if changed:
                    try:
                        encode_history(history, max_context_tokens=512, trim_history=False)
                    except ValueError:
                        skipped_too_long += 1
                    else:
                        samples.append(_entry(episode, index, history, checkpoint_sha))
                        if len(samples) >= max_samples:
                            break
                else:
                    skipped_unchanged += 1
                response = generate_semantic(model, history, max_new_tokens=120, temperature=0.,
                                             seed=seed + index, max_context_tokens=512)
                text = response['text']
                if not isinstance(text, str) or len(text.encode('utf-8')) > 512:
                    raise ValueError('learner response exceeds replay context bound')
                history.append({'role': 'assistant', 'content': text})
                generated.append({'turn_index': index, 'text': text, 'predicted_meaning': response.get('meaning_id')})
            transcripts.append({'episode_id': episode['id'], 'generated': generated})
            if len(samples) >= max_samples:
                break
        check_stop()
        if _checkpoint_hash(checkpoint) != checkpoint_sha:
            raise ValueError('source checkpoint changed during repair generation')
        artifact = {'schema': SCHEMA, 'base_identity': identity,
            'source_checkpoint_sha256': checkpoint_sha, 'source_checkpoint_name': Path(checkpoint).name,
            'seed': seed, 'max_episodes': max_episodes, 'max_samples': max_samples,
            'samples': samples, 'rollouts': transcripts,
            'policy': 'TRAIN non-regression episodes only. Original USER turns and verified simulator targets remain unchanged. Model-generated assistants are untrusted context, never truths or targets.',
            'counts': {'episodes': len(transcripts), 'samples': len(samples), 'skipped_too_long': skipped_too_long,
                       'skipped_unchanged': skipped_unchanged}, 'wall_seconds': time.monotonic() - started}
        artifact['content_sha256'] = _digest(artifact)
        output.parent.mkdir(parents=True, exist_ok=True)
        _json(output, artifact)
        return {'path': str(output.resolve()), 'sha256': _checkpoint_hash(output),
                'base_identity': identity, 'source_checkpoint_sha256': checkpoint_sha,
                **artifact['counts'], 'wall_seconds': artifact['wall_seconds']}
    finally:
        torch.set_num_threads(old_threads)


def apply_repair_overlay(data, path):
    """Validate every teacher label and USER fact, then copy only the TRAIN list."""
    base = _data(data)
    path = Path(path)
    if path.stat().st_size > MAX_OVERLAY_BYTES:
        raise ValueError('repair overlay exceeds 16 MiB')
    with path.open(encoding='utf-8') as stream:
        artifact = json.load(stream)
    if not isinstance(artifact, dict) or artifact.get('schema') != SCHEMA:
        raise ValueError('unsupported repair overlay schema')
    material = {key: value for key, value in artifact.items() if key != 'content_sha256'}
    if artifact.get('content_sha256') != _digest(material):
        raise ValueError('repair overlay content checksum mismatch')
    if artifact.get('base_identity') != _base_identity(base):
        raise ValueError('repair overlay belongs to another base TRAIN corpus')
    checkpoint_sha = artifact.get('source_checkpoint_sha256')
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha):
        raise ValueError('repair source checkpoint hash is invalid')
    rows = artifact.get('samples')
    if not isinstance(rows, list) or len(rows) > MAX_SAMPLES:
        raise ValueError('repair overlay must have at most 512 samples')
    episodes = _episode_index(base)
    generated_by_episode = _rollout_index(artifact, episodes)
    valid, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get('episode_id') not in episodes:
            raise ValueError('repair references an episode outside TRAIN')
        episode = episodes[row['episode_id']]
        expected = _entry(episode, row.get('target_turn_index'), row.get('turns'), checkpoint_sha)
        if row != expected:
            raise ValueError('repair sample differs from verified simulator targets or context provenance')
        generated = generated_by_episode.get(row['episode_id'], {})
        if any(generated.get(index) != row['turns'][index]['content'] for index in row['generated_assistant_turns']):
            raise ValueError('repair context differs from its recorded learner rollout')
        if row['id'] in seen:
            raise ValueError('duplicate repair sample')
        seen.add(row['id'])
        valid.append(expected)
    result = dict(base)
    result['samples'] = dict(base['samples'])
    result['samples']['train'] = list(base['samples']['train']) + valid
    result['repair_overlay_provenance'] = {'path': str(path.resolve()), 'sha256': _checkpoint_hash(path),
        'added_train_samples': len(valid), 'base_identity': artifact['base_identity'],
        'source_checkpoint_sha256': checkpoint_sha, 'policy': artifact['policy']}
    # Validation, test, regression episodes and samples remain the original
    # shared objects. The base TRAIN list and every base sample are untouched.
    return result


def apply_repair_overlays(data, paths):
    """Combine at most two independently verified overlays on the pristine base.

    The optional teacher lesson and current learner replay retain separate
    checkpoint provenance. Neither may redefine targets or accumulate an
    already augmented corpus. Validation/test objects remain unchanged.
    """
    base = _data(data)
    paths = list(paths)
    if not 1 <= len(paths) <= 2:
        raise ValueError('provide one or two bounded repair overlays')
    if len(paths) == 1:
        return apply_repair_overlay(base, paths[0])
    added, seen, provenance = [], set(), []
    original_count = len(base['samples']['train'])
    for path in paths:
        verified = apply_repair_overlay(base, path)
        provenance.append(verified['repair_overlay_provenance'])
        for row in verified['samples']['train'][original_count:]:
            if row['id'] not in seen:
                seen.add(row['id'])
                added.append(row)
    result = dict(base)
    result['samples'] = dict(base['samples'])
    result['samples']['train'] = list(base['samples']['train']) + added
    result['repair_overlay_provenance'] = {
        'schema': 'golem-verified-repair-combination-v1',
        'overlays': provenance, 'added_train_samples': len(added),
        'policy': 'Each overlay separately verified against pristine TRAIN; duplicate IDs included once.'}
    return result
