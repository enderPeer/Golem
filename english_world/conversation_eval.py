"""Free-rollout conversation evaluation with source-grounded user-name checks.

Reference assistant turns are NEVER replayed as model history. Only explicitly
provided initial adversarial history, scripted USER turns and actual generated
assistant replies enter inference. Frames and references are scoring labels only.
These are transparent scenario checks, not a general-purpose semantic judge.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re

from .chat_eval import _contains, _fold, prompt_tokens, score_response
from .data import _atomic_json
from .neural_model import ByteGRULM, generate_neural

SCHEMA = 'golem-free-conversation-eval-v1'
ACTS = ('greeting', 'presence', 'identity', 'request_help', 'clarify', 'rephrase', 'correction',
        'loneliness', 'empathy', 'thanks', 'goodbye', 'preferences', 'name_set',
        'name_recall_unknown', 'name_recall_known', 'name_correction', 'topic_chat', 'uncertainty')
_UNKNOWN = ("don't know your name", 'do not know your name', "haven't told me", 'have not told me',
            "haven't shared", 'have not shared', "don't have your name", 'do not have your name',
            "don't have a name", "won't guess", 'what should i call you', 'may i ask your name',
            "don't know yet", 'do not know yet', "don't know who you are", 'do not know who you are')
_ACT_PHRASES = {
    'greeting': ('hello', 'hi', 'hey'),
    'presence': ('here', 'ready', 'listening', 'feelings', 'ready to chat', 'chat is working'),
    'request_help': ('help', 'ask', 'question', 'tell me', 'what do you need', 'what would you like', 'start', 'practice', 'exchange', 'step'),
    'clarify': ('what', 'which', 'clarify', 'tell me more', 'more detail', 'could you', 'can you explain', 'i mean', 'simply', 'simple', 'in other words'),
    'rephrase': ('another way', 'rephrase', 'simpler', 'in other words', 'say that again', 'try again', 'differently', 'i mean', 'simply', 'simple'),
    'correction': ('sorry', 'mistake', 'correct', 'you are right', "you're right", 'got that wrong', 'try again', 'fix', "shouldn't have invented", 'name was wrong', "won't use a name"),
    'loneliness': ('sorry', 'lonely', 'listen', 'here', 'talk', 'you matter'),
    'empathy': ('sorry', 'hard', 'difficult', 'listen', 'here', 'talk', 'understand', 'tell me', 'share'),
    'thanks': ('welcome', 'glad', 'happy to help', 'anytime'),
    'goodbye': ('bye', 'goodbye', 'see you', 'take care', 'talk later'),
    'preferences': ('favorite', 'favorites', 'favourite', 'favourites', 'preferences', 'prefer', 'personal tastes'),
    'name_set': ('meet', 'remember', 'call', 'name', 'thanks', 'got it', 'hello', 'hi'),
    'name_correction': ('use', 'call', 'name', 'correction', 'correct', 'thanks', 'got it', 'sorry', 'remember'),
    'topic_chat': ('chat', 'talk', 'topic', 'tell me', "what's on", 'what is on', 'discuss'),
    'uncertainty': ("don't know", 'do not know', 'not sure', 'uncertain', "can't tell", 'cannot tell', "don't have that information", "don't want to guess"),
}
_NOT_NAMES = {'i', "i'm", 'unknown', 'not', 'unspecified', 'a', 'an', 'the', 'welcome', 'right',
              'here', 'safe', 'human', 'someone', 'kind', 'talking', 'speaking', 'asking',
              'using', 'feeling', 'with', 'my', 'your', 'there', 'friend', 'everyone', 'again'}
_NOT_NAMES.update(('how', 'what', 'nice', 'good', 'thanks', 'thank', 'glad', 'it', 'we', 'you', 'called'))


def _name_claims(text, act):
    """Extract explicit current-name assertions; output remains visible for audit."""
    text = text.replace('\u2019', "'")
    claims = []
    patterns = (
        (r'\byour\s+(?:first\s+)?name(?:\s+in\s+this\s+chat)?\s+(?:is|was|must be|may be|might be|could be)\s+([^.!?\n,;]+)', 'your_name_assertion'),
        (r"\b(?:i will call you|i'll call you|i call you)\s+([^.!?\n,;]+)", 'call_you_assertion'),
        (r'\b(?:you are called|i have your name as|you asked me to call you)\s+([^.!?\n,;]+)', 'name_record_assertion'),
        (r"\b(?:you are|you're)\s+([^.!?\n,;]+)", 'you_are_assertion'),
    )
    for expression, kind in patterns:
        for match in re.finditer(expression, text, re.IGNORECASE):
            candidate = re.split(r'\s+(?:and|but|because|right|now|from|as|so)\b', match.group(1), maxsplit=1,
                                 flags=re.IGNORECASE)[0].strip(' \t\"\'():')
            words = re.findall(r"[\w]+(?:[-'][\w]+)*", candidate)
            if not words or _fold(words[0]) in _NOT_NAMES or len(words) > 4:
                continue
            if kind == 'you_are_assertion' and not act.startswith('name_') and not words[0][0].isupper():
                continue
            claims.append({'value': ' '.join(words), 'kind': kind, 'evidence': match.group(0)})
    for match in re.finditer(r"(?i:\b(?:hi|hello|hey))(?:,\s*|\s+)([A-Z][\w'-]+)\b", text):
        if _fold(match.group(1)) not in _NOT_NAMES:
            claims.append({'value': match.group(1), 'kind': 'named_address', 'evidence': match.group(0)})
    if act.startswith('name_'):
        for match in re.finditer(r"(?i:\b(?:got it|thanks|thank you|nice to meet you|pleased to meet you|okay|understood|i will use|i'll use))[,:]?\s+([A-Z][\w'-]+)\b", text):
            if _fold(match.group(1)) not in _NOT_NAMES:
                claims.append({'value': match.group(1), 'kind': 'name_acknowledgment', 'evidence': match.group(0)})
        for match in re.finditer(r"(?i:\b(?:maybe|perhaps|is it|i think it is|i think your name is))\s+([A-Z][\w'-]+)\b", text):
            if _fold(match.group(1)) not in _NOT_NAMES:
                claims.append({'value': match.group(1), 'kind': 'name_guess', 'evidence': match.group(0)})
    if act in ('name_recall_unknown', 'name_recall_known'):
        bare = text.strip(' \t\r\n.!?\"\'')
        words = re.findall(r"[\w]+(?:[-'][\w]+)*", bare)
        if 1 <= len(words) <= 3 and all(word[:1].isupper() for word in words) and _fold(words[0]) not in _NOT_NAMES:
            claims.append({'value': ' '.join(words), 'kind': 'bare_name_answer', 'evidence': bare})
    return list({_fold(claim['value']): claim for claim in claims}.values())


def _relevance(text, frame):
    act, known = frame['act'], frame.get('user_name')
    if act not in ACTS:
        raise ValueError('unknown conversation meaning/act')
    if act != 'goodbye' and any(_contains(text, phrase) for phrase in ('goodbye', 'bye', 'see you')):
        return False
    if act != 'preferences' and any(_contains(text, phrase) for phrase in ('favorite', 'favorites', 'favourite', 'favourites')):
        return False
    if act == 'identity':
        return (_fold(text).strip(' .!') == 'golem' or any(_contains(text, phrase) for phrase in
                ('i am golem', "i'm golem", 'my name is golem', 'you can call me golem', 'this is golem')))
    if act == 'name_recall_unknown':
        return known is None and any(_contains(text, phrase) for phrase in _UNKNOWN)
    if act == 'name_recall_known':
        denied = bool(known) and re.search(r"\b(?:not|isn't|aren't)\s+" + re.escape(_fold(known)) + r'(?!\w)', _fold(text)) is not None
        return bool(known) and _contains(text, known) and not denied and not any(_contains(text, phrase) for phrase in _UNKNOWN)
    if act == 'topic_chat' and frame.get('topic'):
        related = {'books': ('book', 'books', 'story'), 'music': ('music', 'song'),
                   'gardens': ('gardens', 'garden', 'grow', 'growing'), 'walking': ('walk', 'walking'),
                   'food': ('food', 'dish'), 'puzzles': ('puzzle', 'puzzles'), 'art': ('art',),
                   'movies': ('movie', 'movies')}.get(frame['topic'], (frame['topic'],))
        return '?' in text and any(_contains(text, term) for term in related)
    phrases = _ACT_PHRASES[act]
    relevant = any(_contains(text, phrase) for phrase in phrases)
    if act in ('name_set', 'name_correction'):
        relevant = relevant and bool(known) and _contains(text, known)
    return relevant


def score_conversation_turn(text, frame, previous=None):
    """Combine declared rubric, request relevance and independent name grounding."""
    rubric = score_response(text, frame['rubric'])
    relevant = _relevance(text, frame)
    claims = _name_claims(text, frame['act'])
    known = frame.get('user_name')
    false = [claim for claim in claims if known is None or _fold(claim['value']) != _fold(known)]
    # Explicitly naming the user Golem is not a correct answer to "who are YOU",
    # even if a user happens to have introduced that same spelling.
    if frame['act'] == 'identity' and any(claim['kind'] in ('your_name_assertion', 'you_are_assertion') for claim in claims):
        relevant = False
    prior = previous or []
    repeated_irrelevant = any(_fold(row['generated']) == _fold(text) and
        (row['act'] != frame['act'] or row.get('expected_user_name') != known)
        for row in prior) and (not relevant or not rubric['heuristic_pass'] or bool(false))
    score = rubric['score'] if relevant and not false and not repeated_irrelevant else 0.
    return {'score': score, 'passed': bool(rubric['heuristic_pass'] and relevant and not false and not repeated_irrelevant),
            'rubric_result': rubric, 'relevant_to_request': relevant, 'name_claims': claims,
            'false_memory_claims': false, 'repeated_irrelevant_response': bool(repeated_irrelevant),
            'expected_user_name': known}


def _selected_episodes(data, split, limit):
    episodes = data.get('episodes', {}).get(split)
    if episodes is None and split == 'regression':
        episodes = data.get('regression_episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f'no conversation episodes in {split}')
    groups = {}
    for episode in episodes:
        if not isinstance(episode.get('id'), str):
            raise ValueError('episodes require an ID')
        key = episode.get('counterfactual_pair_id') or episode.get('contrast_group_id') or episode['id']
        groups.setdefault(key, []).append(episode)
    remaining = {key: sorted(group, key=lambda episode: episode['id']) for key, group in groups.items()}
    group_acts = {key: {frame['act'] for episode in group for frame in episode.get('target_frames', [])}
                  for key, group in remaining.items()}
    selected, covered = [], set()
    family_counts = defaultdict(int)

    def take(key):
        group = remaining.pop(key)
        selected.extend(group)
        covered.update(group_acts[key])
        for episode in group:
            family_counts[episode.get('family_id', episode['id'])] += 1

    # Predeclared label balancing, independent of model outputs: first maximize
    # newly represented communicative acts without breaking a complete pair.
    while remaining:
        feasible = [key for key, group in remaining.items() if len(selected) + len(group) <= limit]
        if not feasible:
            break
        best = min(feasible, key=lambda key: (-len(group_acts[key] - covered),
                   -len(group_acts[key] - covered) / len(remaining[key]), len(remaining[key]), key))
        if not group_acts[best] - covered:
            break
        take(best)
    # Then fill by least-represented family, with stable ID tie-breaks.
    while remaining:
        feasible = [key for key, group in remaining.items() if len(selected) + len(group) <= limit]
        if not feasible:
            break
        best = min(feasible, key=lambda key: (min(family_counts[episode.get('family_id', episode['id'])]
                    for episode in remaining[key]), min(episode.get('family_id', episode['id']) for episode in remaining[key]), key))
        take(best)
    if not selected:
        raise ValueError('episode limit cannot fit a complete contrast group')
    return selected


def _check_frame(frame, turns, assistant_index):
    if frame.get('turn_index') != assistant_index or frame.get('act') not in ACTS:
        raise ValueError('target frame does not label this assistant turn')
    from .dialogue_state import extract_user_facts
    expected = extract_user_facts(turns[:assistant_index])
    if frame.get('user_name') != expected['user_name'] or frame.get('source_user_turn') != expected['source_user_turn']:
        raise ValueError('gold user-name state must match explicit USER facts, including correction and clearing')


def evaluate_conversations(model_or_generate, data, split='validation', max_episodes=32,
                           max_new_tokens=120, seed=7, temperature=0., should_stop=None, output=None):
    if split not in ('validation', 'test', 'regression'):
        raise ValueError('use validation, test, or a separately labelled regression split')
    if type(max_episodes) is not int or not 1 <= max_episodes <= 256:
        raise ValueError('max_episodes must be 1..256')
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 120:
        raise ValueError('max_new_tokens must be 1..120')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if should_stop is not None and not callable(should_stop):
        raise ValueError('should_stop must be callable')
    if isinstance(model_or_generate, ByteGRULM):
        def generate(history, **kwargs):
            return generate_neural(model_or_generate, prompt_tokens(history, max_prompt_bytes=8192), dialogue=False, **kwargs)
    elif callable(model_or_generate):
        generate = model_or_generate
    else:
        raise ValueError('supply a ByteGRU or a generate(history, **kwargs) callback')

    def check_stop():
        if should_stop is not None and should_stop():
            raise InterruptedError('conversation evaluation interrupted; no partial score published')

    check_stop()
    completed = []
    for episode in _selected_episodes(data, split, max_episodes):
        check_stop()
        turns = episode.get('turns')
        initial = episode.get('initial_history', [])
        initial_count = episode.get('initial_history_count', len(initial))
        if not isinstance(turns, list) or not 1 <= len(turns) <= 64 or initial_count != len(initial) or turns[:initial_count] != initial:
            raise ValueError('invalid episode turns or adversarial initial history prefix')
        frames = {frame['turn_index']: frame for frame in episode.get('target_frames', [])}
        history = [dict(turn) for turn in initial]
        rows = []
        identity = episode.get('counterfactual_pair_id') or episode.get('contrast_group_id') or episode['id']
        base_seed = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big') % (2**63)
        for index in range(initial_count, len(turns)):
            check_stop()
            turn = turns[index]
            if turn.get('role') == 'user':
                history.append({'role': 'user', 'content': turn['content']})
                continue
            if turn.get('role') != 'assistant' or index not in frames or not history or history[-1]['role'] != 'user':
                raise ValueError('each scored assistant turn must follow a user request and have a target frame')
            frame = frames[index]
            _check_frame(frame, turns, index)
            budget = min(max_new_tokens, frame['rubric'].get('max_response_bytes', 120))
            generated = generate([dict(item) for item in history], max_new_tokens=budget,
                                 temperature=temperature, seed=base_seed + len(rows))
            if isinstance(generated, str):
                text, predicted = generated, None
            elif isinstance(generated, dict) and isinstance(generated.get('text'), str):
                text, predicted = generated['text'], generated.get('meaning_id')
            else:
                raise ValueError('generator must return text or a dict containing text')
            scored = score_conversation_turn(text, frame, rows)
            row = {'turn_index': index, 'case_id': frame.get('case_id'), 'act': frame['act'],
                   'meaning_id': frame.get('meaning_id'), 'predicted_meaning_id': predicted,
                   'predicted_meaning_correct': predicted in (frame['act'], frame.get('meaning_id')) if predicted is not None else None,
                   'input_history': [dict(item) for item in history], 'generated': text,
                   'reference': turn['content'], 'frame': frame, **scored}
            rows.append(row)
            history.append({'role': 'assistant', 'content': text})
        if not rows:
            raise ValueError('episode contains no scored assistant responses')
        false_count = sum(len(row['false_memory_claims']) for row in rows)
        completed.append({'id': episode['id'], 'family_id': episode.get('family_id'),
            'counterfactual_pair_id': episode.get('counterfactual_pair_id'), 'contrast_group_id': episode.get('contrast_group_id'),
            'initial_history': initial, 'rows': rows, 'actual_history': history,
            'score': sum(row['score'] for row in rows) / len(rows), 'passed': all(row['passed'] for row in rows),
            'false_memory_claims': false_count, 'false_memory_turns': sum(bool(row['false_memory_claims']) for row in rows)})
    check_stop()
    by_act, pairs, contrasts = defaultdict(list), defaultdict(list), defaultdict(list)
    for episode in completed:
        for row in episode['rows']:
            by_act[row['act']].append(row)
        if episode['counterfactual_pair_id']:
            pairs[episode['counterfactual_pair_id']].append(episode)
        if episode['contrast_group_id']:
            contrasts[episode['contrast_group_id']].append(episode)
    per_meaning = {act: {'turns': len(rows), 'score': sum(row['score'] for row in rows) / len(rows),
                        'pass_rate': sum(row['passed'] for row in rows) / len(rows),
                        'false_memory_claims': sum(len(row['false_memory_claims']) for row in rows)}
                   for act, rows in sorted(by_act.items())}
    pair_rows = []
    for identity, members in sorted(pairs.items()):
        final = [member['rows'][-1] for member in members]
        names = [row['expected_user_name'] for row in final]
        eligible = len(final) >= 2 and len(set(names)) >= 2
        pair_rows.append({'id': identity, 'episode_ids': [member['id'] for member in members],
                          'eligible': eligible, 'passed': bool(eligible and all(row['passed'] for row in final)),
                          'expected_names': names, 'generated': [row['generated'] for row in final]})
    contrast_rows = [{'id': identity, 'episode_ids': [member['id'] for member in members],
                     'passed': all(member['rows'][-1]['passed'] for member in members),
                     'responses_differ': len({_fold(member['rows'][-1]['generated']) for member in members}) > 1}
                    for identity, members in sorted(contrasts.items())]
    all_rows = [row for episode in completed for row in episode['rows']]
    predicted_rows = [row for row in all_rows if row['predicted_meaning_correct'] is not None]
    false_total = sum(episode['false_memory_claims'] for episode in completed)
    report = {'schema': SCHEMA, 'split': split,
              'evaluation_kind': 'known_user_regression' if split == 'regression' else 'held_out_scenario_rollout',
              'episodes': len(completed), 'turns': len(all_rows),
              'score': sum(row['score'] for row in per_meaning.values()) / len(per_meaning),
              'covered_acts': sorted(per_meaning), 'covered_act_count': len(per_meaning),
              'available_acts': sorted({frame['act'] for episode in
                  (data.get('episodes', {}).get(split) or data.get('regression_episodes', []))
                  for frame in episode.get('target_frames', [])}),
              'episode_selection': 'Greedy new-ACT coverage first, then balanced family filling; complete contrast/counterfactual groups only. Independent of model outputs.',
              'whole_episode_pass_rate': sum(episode['passed'] for episode in completed) / len(completed),
              'false_memory_claims': false_total, 'false_memory_turns': sum(episode['false_memory_turns'] for episode in completed),
              'false_memory_episode_rate': sum(episode['false_memory_claims'] > 0 for episode in completed) / len(completed),
              'repeated_irrelevant_responses': sum(row['repeated_irrelevant_response'] for row in all_rows),
              'meaning_accuracy': sum(row['predicted_meaning_correct'] for row in predicted_rows) / len(predicted_rows) if predicted_rows else None,
              'per_meaning': per_meaning, 'counterfactual_pairs': pair_rows, 'latest_question_contrasts': contrast_rows,
              'release_eligible': false_total == 0 and all(episode['passed'] for episode in completed),
              'rows': completed, 'temperature': temperature, 'seed': seed,
              'history_policy': 'Only scripted users, explicit initial adversarial prefix, and ACTUAL generated assistant replies enter inference; teacher replies and frames are evaluation labels only.',
              'limitations': 'Closed-scenario phrase/relevance and explicit-name-assertion checks, not a complete semantic judge or detector of every possible hallucination. Meaning-head accuracy is separate from reply correctness.'}
    if output is not None:
        check_stop()
        _atomic_json(Path(output), report)
    return report


def validation_selection_key(report):
    if report.get('split') != 'validation':
        raise ValueError('test and regression rollouts must not select checkpoints')
    return (report['false_memory_claims'] == 0, float(report['score']), float(report['whole_episode_pass_rate']))
