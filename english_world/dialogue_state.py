"""Conservative, explicit USER fact memory for simulated and real conversations.

This is visible symbolic preprocessing, not a learned memory claim. Assistant
text never creates a user fact. No answers or response templates are returned.
"""
from __future__ import annotations

import re

_NAME = re.compile(
    r"^(?:(?:actually|no)[,:]?\s+)?(?:my name is|call me|i am called|i'm called|i go by)\s+"
    r"([^\W\d_](?:[^\W\d_]|['\u2019-]){0,31})[.!]?$", re.IGNORECASE)
_FORGET = re.compile(r"^(?:please\s+)?(?:forget my name|don't remember my name|do not remember my name)[.!]?$", re.IGNORECASE)
_NON_NAMES = frozenset(('not', 'unknown', 'none', 'lonely', 'sad', 'happy', 'fine', 'here',
                        'there', 'no', 'yes', 'someone', 'something', 'user', 'assistant', 'you', 'me'))


def extract_user_facts(turns: list[dict]) -> dict:
    """Last explicit self-introduction wins; unsupported/ambiguous text does not.

    Source positions index the supplied role history. The supported grammar is
    deliberately narrow: one-word names in explicit first-person assertions.
    Questions, quotations, hypothetical statements and assistant claims cannot
    fill this slot. Broader entity extraction is a separate capability.
    """
    if not isinstance(turns, list):
        raise ValueError('turns must be a list of role/content objects')
    facts = {'user_name': None, 'source_user_turn': None}
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get('role') not in ('user', 'assistant') or not isinstance(turn.get('content'), str):
            raise ValueError('turns require user/assistant roles and string content')
        if turn['role'] != 'user':
            continue
        content = turn['content'].strip()
        if _FORGET.fullmatch(content):
            facts = {'user_name': None, 'source_user_turn': index}
            continue
        match = _NAME.fullmatch(content)
        if match:
            name = match.group(1)
            if name.casefold() not in _NON_NAMES and len(name.encode('utf-8')) <= 32 and name[-1].isalpha():
                facts = {'user_name': name, 'source_user_turn': index}
    return facts


def render_verified_facts(turns: list[dict]) -> str:
    facts = extract_user_facts(turns)
    return '[verified user name: ' + (facts['user_name'] or 'UNKNOWN') + ']\n'
