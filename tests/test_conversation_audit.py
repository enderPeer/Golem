"""Posthoc flags do not rewrite frozen evaluation or accept assistant facts."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from english_world import conversation_audit as audit


def row(text, act='loneliness', history=None):
    return {'case_id': 'saved-case', 'turn_index': 1, 'act': act, 'generated': text,
            'input_history': history or [{'role': 'user', 'content': 'I am lonely.'}],
            'frame': {'user_name': 'FORGED GOLD NAME'}, 'expected_user_name': 'FORGED GOLD NAME',
            'false_memory_claims': [], 'score': 1., 'passed': True}


def report(*rows):
    return {'schema': 'golem-free-conversation-eval-v1', 'split': 'regression',
            'score': 1., 'false_memory_claims': 0, 'release_eligible': True,
            'rows': [{'id': 'saved-episode', 'rows': list(rows)}]}


class ConversationAuditTests(unittest.TestCase):
    def test_cross_act_kalen_and_bales_are_review_flags_despite_legacy_zero(self):
        source = report(row('Okay, Kalen. We can talk.'),
                        row("Okay, Bales. I'll use that name for this conversation.", 'clarify'))
        original = copy.deepcopy(source)
        result = audit.audit_report(source)
        self.assertEqual(source, original)
        self.assertEqual(result['legacy_report_summary']['false_memory_claims'], 0)
        self.assertEqual(result['flags'], 2)
        self.assertEqual(result['flagged_turns'], 2)
        self.assertEqual(result['ambiguous_suspected_name_addresses'], 1)
        self.assertEqual(result['unsupported_name_claims_or_addresses'], 1)
        self.assertTrue(result['blocks_live_release'])
        self.assertTrue(result['review_required'])
        self.assertFalse(result['automatic_release'])

    def test_current_known_user_name_is_supported_and_assistant_fact_is_not(self):
        history = [{'role': 'user', 'content': 'My name is Kalen'},
                   {'role': 'assistant', 'content': 'Your name is Bales.'},
                   {'role': 'user', 'content': 'Are you there?'}]
        result = audit.audit_turn(row("Okay, Kalen. I'll use that name.", 'presence', history))
        self.assertFalse(result['flags'])
        self.assertEqual(result['verified_user_facts']['user_name'], 'Kalen')
        self.assertTrue(result['claims'][0]['supported_by_user_history'])
        history[0]['content'] = 'hello'
        result = audit.audit_turn(row("Okay, Bales. I'll use that name.", 'presence', history))
        self.assertTrue(result['flags'])
        self.assertIsNone(result['verified_user_facts']['user_name'])

    def test_forgetting_and_correction_override_old_names(self):
        history = [{'role': 'user', 'content': 'Call me Kalen'},
                   {'role': 'assistant', 'content': 'Hello Kalen.'},
                   {'role': 'user', 'content': 'Forget my name'}]
        forgotten = audit.audit_turn(row('Okay, Kalen.', 'correction', history))
        self.assertTrue(forgotten['review_required'])
        self.assertEqual(forgotten['verified_user_facts'], {'user_name': None, 'source_user_turn': 2})
        history[-1]['content'] = 'Actually, my name is Bales'
        self.assertTrue(audit.audit_turn(row('Okay, Kalen.', 'name_correction', history))['flags'])
        self.assertFalse(audit.audit_turn(row('Okay, Bales.', 'name_correction', history))['flags'])

    def test_pronouns_common_acknowledgments_and_forced_you_are_are_not_new_names(self):
        texts = ['Okay, That sounds difficult.', 'Okay, This may help.', "Okay, I'll listen.",
                 "Understood, You're welcome to talk.", 'Maybe We can talk.', 'Perhaps It would help.',
                 'You are lonely, and you are not forgotten.', 'Hello! How are you?']
        for text in texts:
            with self.subTest(text=text):
                result = audit.audit_turn(row(text))
                self.assertFalse(result['flags'])
                self.assertFalse(any(claim['kind'] == 'you_are_assertion' for claim in result['claims']))
        # An explicit user fact wins over the conservative common-word filter.
        known = audit.audit_turn(row('Okay, That.', 'greeting', [{'role': 'user', 'content': 'Call me That'}]))
        self.assertTrue(known['claims'][0]['supported_by_user_history'])

    def test_explicit_actual_act_assertion_and_cross_act_guess_remain_visible(self):
        result = audit.audit_report(report(row('Your name is Kalen.', 'identity'),
                                           row('Perhaps Bales.', 'topic_chat')))
        claims = [flag for episode in result['rows'] for turn in episode['rows'] for flag in turn['flags']]
        self.assertEqual([claim['kind'] for claim in claims], ['your_name_assertion', 'name_guess'])
        self.assertEqual(claims[0]['extractor_origin'], 'legacy_actual_act')
        self.assertEqual(claims[1]['finding'], 'suspected_name_address')

    def test_report_file_hash_and_cli_sidecar_preserve_original_and_block_release(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'legacy.json', Path(directory) / 'audit.json'
            content = json.dumps(report(row('Okay, Kalen.')), indent=1).encode() + b'\n'
            source.write_bytes(content)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = audit.main([str(source), '--output', str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(code, 2)
            self.assertEqual(source.read_bytes(), content)
            self.assertEqual(result['source_report_sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(result['source_hash_scope'], 'exact source file bytes')
            self.assertTrue(json.loads(stdout.getvalue())['blocks_live_release'])
            with self.assertRaises(ValueError):
                audit.audit_report_file(source, source)
            with self.assertRaises(ValueError):
                audit.audit_report_file(source, output)

    def test_missing_actual_history_cannot_be_replaced_with_gold_or_teacher_history(self):
        bad = row('Your name is Maya.', 'name_recall_known')
        bad.pop('input_history')
        bad['frame'] = {'user_name': 'Maya'}
        bad['reference'] = 'Your name is Maya.'
        with self.assertRaisesRegex(ValueError, 'input_history'):
            audit.audit_report(report(bad))
        clean = audit.audit_report(report(row("I'm here to listen.")))
        self.assertFalse(clean['blocks_live_release'])
        self.assertFalse(clean['automatic_release'])
        self.assertIn('not a release authorization', clean['release_policy'])


if __name__ == '__main__':
    unittest.main()
