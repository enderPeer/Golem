import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.neural_model import decode_bytes

SPEC = importlib.util.spec_from_file_location('chat_release_gate', Path(__file__).resolve().parents[1] / 'ops/chat_release_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class ChatReleaseGateTests(unittest.TestCase):
    def test_wrong_favorites_and_identity_are_rejected_without_substitution(self):
        response = 'I am a program so I do not have personal favorites.'
        with patch.object(gate, 'generate_neural', return_value=response):
            report = gate.evaluate_release(object())
        status = next(row for row in report['single_turns'] if row['id'] == 'whats_up')
        self.assertEqual(status['generated'], response)
        self.assertFalse(status['passed'])
        self.assertFalse(report['passed'])
        with patch.object(gate, 'generate_neural', return_value='My name is Golem.'):
            report = gate.evaluate_release(object())
        self.assertFalse(next(row for row in report['single_turns'] if row['id'] == 'wellbeing')['passed'])

    def test_real_generated_history_is_forwarded_to_multiturn_probe(self):
        answers = ['I am Golem.', 'I am ready to chat.', 'I am here to help.', 'Hello!',
                   'I can answer questions.', "You're welcome!", 'Hello from this model!',
                   'I am Golem, your assistant.', 'I am here and ready to chat.']
        prefixes = []

        def generated(model, tokens, **kwargs):
            prefixes.append(decode_bytes(tokens))
            self.assertEqual(kwargs['temperature'], 0.)
            self.assertLessEqual(kwargs['max_new_tokens'], 120)
            return answers[len(prefixes) - 1]

        with patch.object(gate, 'generate_neural', side_effect=generated):
            report = gate.evaluate_release(object())
        self.assertTrue(report['passed'])
        self.assertEqual(report['pass_count'], 7)
        self.assertIn('Hello from this model!', prefixes[-1])
        self.assertIn('I am Golem, your assistant.', prefixes[-1])
        self.assertNotIn('I am here and ready to chat.', prefixes[-1])
        self.assertEqual(report['multi_turns'][-1]['generated'], answers[-1])

    def test_no_all_pass_candidate_means_no_selection_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'candidate.pt'
            checkpoint.write_bytes(b'fixture checkpoint; loader mocked')
            report_path = root / 'report.json'
            report_path.write_text(json.dumps({'schema': 'golem-chat-collector-v1', 'candidates': [
                {'id': 'island-a', 'host': 'specht32', 'run_dir': 'runs/chat-sweep/island-a',
                 'strategy': 'ordinary', 'updates': 20, 'score': .5, 'reference_loss': 2.,
                 'local_path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}))
            with patch.object(gate, 'load_neural_checkpoint', return_value=(object(), {})), \
                 patch.object(gate, 'generate_neural', return_value='I have no personal favorites.'):
                report = gate.run_release_gate(report_path, root / 'release-report.json')
            self.assertIsNone(report['chosen'])
            self.assertFalse(report['published'])
            self.assertEqual(report['checked_candidates'], 1)
            self.assertEqual(report['rows'][0]['single_turns'][0]['generated'], 'I have no personal favorites.')
            self.assertIsNone(json.loads((root / 'release-report.json').read_text())['chosen'])


if __name__ == '__main__':
    unittest.main()
