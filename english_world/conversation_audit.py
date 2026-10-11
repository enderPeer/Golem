"""Posthoc name-grounding audit of saved conversation reports.

This is a separate review gate, not new training fitness or a general semantic
judge. It never generates answers, changes legacy scores, or releases a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from .conversation_eval import ACTS, SCHEMA as EVAL_SCHEMA, _name_claims
from .dialogue_state import extract_user_facts


SCHEMA = 'golem-conversation-name-audit-v1'
MAX_REPORT_BYTES = 64 * 1024 * 1024
_EXTRA_KINDS = {'name_acknowledgment', 'name_guess'}
_ADDRESS_KINDS = {'named_address', 'name_acknowledgment', 'name_guess', 'bare_name_answer'}
_COMMON_ACKNOWLEDGMENTS = {
    'that', 'this', 'these', 'those', 'it', 'its', "it's", 'i', "i'm", 'you', 'your', 'we',
    'they', 'them', 'he', 'she', 'me', 'my', 'our', 'their', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'and', 'but', 'so', 'then', 'there', 'here', 'well', 'okay', 'ok',
    'alright', 'sure', 'yes', 'no', 'right', 'good', 'great', 'fine', 'thanks', 'thank',
    'please', 'sorry', 'understood', 'welcome', 'nice', 'glad', 'happy', 'hello', 'hi', 'hey',
    'maybe', 'perhaps', 'not', 'nothing', 'something', 'anything', 'everything', 'everyone',
    'someone', 'anyone', 'let', "let's", 'lets', 'now', 'again', 'also', 'just', 'instead',
    "i'll", "i've", "i'd", "we'll", "we're", "we've", "we'd", "you're", "you'll", "you've", "you'd",
    "that's", "there's", "here's", "what's", "he's", "she's", "they're", "they'll", "they've", "they'd",
}
_NAME_REINFORCEMENT = re.compile(
    r"\b(?:(?:i will|i'll)\s+(?:use|remember)\s+(?:that|your)\s+name|"
    r"(?:that|this)\s+is\s+your\s+name|(?:i will|i'll)\s+call\s+you)\b", re.IGNORECASE)


def _fold(value):
    return ' '.join(value.replace('\u2019', "'").casefold().split())


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def audit_turn(row):
    """Inspect existing text; gold frames and prior assistant claims are not facts."""
    act, text, history = row.get('act'), row.get('generated'), row.get('input_history')
    if act not in ACTS or not isinstance(text, str) or len(text.encode('utf-8')) > 65536:
        raise ValueError('audit rows require a declared act and bounded generated text')
    if not isinstance(history, list) or not history or len(history) > 512:
        raise ValueError('audit rows require their actual bounded input_history')
    if sum(len(turn.get('content', '').encode('utf-8')) for turn in history if isinstance(turn, dict)
           and isinstance(turn.get('content'), str)) > 1024 * 1024:
        raise ValueError('audit input_history exceeds its byte limit')
    facts = extract_user_facts(history)
    if history[-1]['role'] != 'user':
        raise ValueError('audited input_history must end with the triggering USER request')
    known = facts['user_name']
    original = _name_claims(text, act)
    # Crucially do not import forced-name-act you_are_assertion or bare-answer
    # interpretations: "you are lonely" is not a newly inferred user name.
    additional = [claim for claim in _name_claims(text, 'name_set') if claim['kind'] in _EXTRA_KINDS]
    claims, flags, suppressed = [], [], []
    seen = set()
    for claim, origin in [(claim, 'legacy_actual_act') for claim in original] + [
            (claim, 'cross_act_acknowledgment_or_guess') for claim in additional]:
        key = (_fold(claim['value']), claim['kind'], claim['evidence'])
        if key in seen:
            continue
        seen.add(key)
        supported = known is not None and _fold(claim['value']) == _fold(known)
        if (not supported and claim['kind'] in _ADDRESS_KINDS and
                _fold(claim['value']) in _COMMON_ACKNOWLEDGMENTS):
            suppressed.append({**claim, 'reason': 'common acknowledgment/pronoun, not enough evidence of a name'})
            continue
        recorded = {**claim, 'extractor_origin': origin, 'supported_by_user_history': supported}
        claims.append(recorded)
        if supported:
            continue
        ambiguous = (origin == 'cross_act_acknowledgment_or_guess' and
                     not _NAME_REINFORCEMENT.search(text.replace('\u2019', "'")))
        finding = ('suspected_name_address' if ambiguous else
                   'unsupported_name_address' if claim['kind'] in _ADDRESS_KINDS else 'unsupported_name_assertion')
        flags.append({**recorded, 'finding': finding, 'review_required': True,
                      'reason': ('Name-like wording needs human review; the USER history does not support this value.'
                                 if ambiguous else 'The extracted user-name claim/address is unsupported by explicit USER facts.')})
    return {'case_id': row.get('case_id'), 'turn_index': row.get('turn_index'), 'act': act,
            'generated': text, 'verified_user_facts': facts, 'input_history_sha256': _canonical_hash(history),
            'claims': claims, 'flags': flags, 'suppressed_common_acknowledgments': suppressed,
            'review_required': bool(flags)}


def audit_report(report, *, source_report_sha256=None):
    """Audit a v1 report without mutating it; optional hash identifies source bytes."""
    if not isinstance(report, dict) or report.get('schema') != EVAL_SCHEMA:
        raise ValueError('supply a saved golem-free-conversation-eval-v1 report')
    if source_report_sha256 is not None and (not isinstance(source_report_sha256, str) or
                                           not re.fullmatch(r'[0-9a-f]{64}', source_report_sha256)):
        raise ValueError('source_report_sha256 must be a lowercase SHA-256 digest')
    episodes = report.get('rows')
    if not isinstance(episodes, list) or not 1 <= len(episodes) <= 256:
        raise ValueError('audit expects 1..256 saved episode rows')
    audited, total = [], 0
    for episode in episodes:
        rows = episode.get('rows') if isinstance(episode, dict) else None
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64 or any(not isinstance(row, dict) for row in rows):
            raise ValueError('audit expects 1..64 saved turn rows per episode')
        evaluated = [audit_turn(row) for row in rows]
        total += len(evaluated)
        audited.append({'id': episode.get('id'), 'rows': evaluated,
                        'review_required': any(row['review_required'] for row in evaluated)})
    flags = [flag for episode in audited for row in episode['rows'] for flag in row['flags']]
    canonical = _canonical_hash(report)
    return {'schema': SCHEMA, 'source_report_sha256': source_report_sha256 or canonical,
            'source_hash_scope': 'caller-supplied source bytes' if source_report_sha256 else 'canonical parsed JSON',
            'source_canonical_json_sha256': canonical,
            'legacy_report_summary': {key: report.get(key) for key in
                                      ('schema', 'split', 'score', 'false_memory_claims', 'release_eligible')},
            'episodes': len(audited), 'turns': total, 'flags': len(flags),
            'unsupported_name_claims_or_addresses': sum(flag['finding'] != 'suspected_name_address' for flag in flags),
            'ambiguous_suspected_name_addresses': sum(flag['finding'] == 'suspected_name_address' for flag in flags),
            'flagged_turns': sum(row['review_required'] for episode in audited for row in episode['rows']),
            'review_required': bool(flags), 'blocks_live_release': bool(flags), 'automatic_release': False,
            'release_policy': 'Any flags block live release pending review. A clean audit is not a release authorization; all other release requirements still apply.',
            'method': 'Original actual-act _name_claims plus only name_acknowledgment/name_guess extras across acts; facts come solely from each recorded input_history USER message.',
            'limitations': 'A bounded posthoc phrase audit, not a general semantic judge. Ambiguous addresses require review; unsupported names and other hallucinations can still escape these patterns. Legacy training scores/protocols are unchanged.',
            'rows': audited}


def audit_report_file(path, output=None):
    """Hash the exact saved source and optionally write a separate immutable audit."""
    path = Path(path)
    destination = Path(output) if output is not None else None
    if destination is not None and (destination.resolve() == path.resolve() or destination.exists()):
        raise ValueError('audit output must be a fresh file separate from the source report')
    with path.open('rb') as stream:
        content = stream.read(MAX_REPORT_BYTES + 1)
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError('source report exceeds the 64 MiB audit limit')
    result = audit_report(json.loads(content), source_report_sha256=hashlib.sha256(content).hexdigest())
    result['source_hash_scope'] = 'exact source file bytes'
    result['source_report_path'] = str(path.resolve())
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='Existing saved free-conversation v1 report; no model is evaluated')
    parser.add_argument('--output', help='Fresh JSON audit sidecar; never overwrites the source')
    args = parser.parse_args(argv)
    result = audit_report_file(args.report, args.output)
    print(json.dumps({key: result[key] for key in ('schema', 'source_report_sha256', 'turns', 'flags',
        'flagged_turns', 'review_required', 'blocks_live_release', 'automatic_release')}, indent=2))
    return 2 if result['blocks_live_release'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
