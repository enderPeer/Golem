"""Code-domain split discipline, diagnostic isolation and legacy resume checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world import evolution
from english_world.evolution import Trainer, _configuration, _fingerprint, train


def code_dataset():
    return {
        'domain': 'code',
        'alphabet': ''.join(chr(c) for c in range(32, 127)) + '\n\t',
        'train': [{'context': 'def Example():\n\treturn ', 'target': 'X'}] * 12,
        'validation': [{'context': 'class HeldOut:\n    result = ', 'target': 'Y'}] * 4,
        'test': [{'context': 'TEST_ONLY = ', 'target': 'Z'}] * 4,
        'dictionary': [],
        'manifest': {'source': 'self-code', 'domain': 'code', 'source_commit': 'fixture',
                     'source_sha256': 'a' * 64,
                     'source_files': [{'path': 'english_world/train_fixture.py', 'sha256': 'b' * 64,
                                       'normalized_sha256': 'c' * 64, 'split': 'train'}]},
    }


def config(generations=2):
    return {'domain': 'code', 'population_size': 4, 'generations': generations,
            'seed': 11, 'max_order': 8, 'max_contexts': 32, 'elite_count': 1,
            'mutation_rate': 1., 'guided_probability': 1., 'mutation_events': 2,
            'eval_examples': 8, 'validation_examples': 4, 'generated_chars': 0,
            'max_seconds': 0, 'checkpoint_seconds': 600.}


class CodeEvolutionTests(unittest.TestCase):
    def test_only_training_examples_reach_guided_mutation(self):
        data = code_dataset()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evolution, 'reproduce', wraps=evolution.reproduce) as calls:
                train(config(), data, directory)
            self.assertGreater(calls.call_count, 0)
            for call in calls.call_args_list:
                self.assertIs(call.kwargs['train_examples'], data['train'])
            state = json.loads((Path(directory) / 'checkpoint.json').read_text())
            for member in state['population']:
                self.assertEqual(member['genome']['domain'], 'code')
                for row in member['genome']['rules'].values():
                    self.assertEqual(set(row), {'X'})

    def test_source_identity_change_rejects_resume_without_sample_change(self):
        data = code_dataset()
        with tempfile.TemporaryDirectory() as directory:
            train(config(1), data, directory)
            changed = copy.deepcopy(data)
            changed['manifest']['source_files'][0]['sha256'] = 'd' * 64
            self.assertNotEqual(_fingerprint(data), _fingerprint(changed))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                train(config(2), changed, directory, resume=True)

    def test_domain_mismatch_and_invalid_context_order_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'domain'):
                train(dict(config(), domain='english', max_order=5), code_dataset(), directory)
        self.assertEqual(_configuration(dict(config(), max_order=16))['max_order'], 16)
        with self.assertRaises(ValueError):
            _configuration(dict(config(), max_order=17))

    def test_self_prediction_diagnostic_does_not_affect_selection(self):
        states = []
        for artificial_loss in (0., 1000.):
            with tempfile.TemporaryDirectory() as directory:
                with patch('english_world.model.self_prediction_metrics',
                           return_value={'applicable': True, 'loss': artificial_loss}):
                    result = train(config(), code_dataset(), directory)
                state = json.loads((Path(directory) / 'checkpoint.json').read_text())
                status = json.loads((Path(directory) / 'status.json').read_text())
                self.assertEqual(status['champion_self_prediction']['loss'], artificial_loss)
                self.assertEqual(result['source_manifest']['source'], 'self-code')
                self.assertNotIn('self_prediction', state['champion']['metrics'])
                states.append(state)
        for key in ('population', 'rng_state', 'history', 'ancestry', 'total_births'):
            self.assertEqual(states[0][key], states[1][key], key)
        self.assertEqual(states[0]['champion']['genome_hash'], states[1]['champion']['genome_hash'])

    def test_code_mid_generation_resume_is_deterministic(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            train(config(3), code_dataset(), whole)
            partial = Trainer(config(3), code_dataset(), split)
            for _ in range(8):
                partial._step()
            self.assertEqual(partial.state['stage'], 'reproduce')
            partial.checkpoint()
            train(config(3), code_dataset(), split, resume=True)
            a = json.loads((Path(whole) / 'checkpoint.json').read_text())
            b = json.loads((Path(split) / 'checkpoint.json').read_text())
            for key in ('population', 'rng_state', 'champion', 'history', 'ancestry'):
                self.assertEqual(a[key], b[key], key)

    def test_legacy_english_fingerprint_and_missing_domain_resume(self):
        data = {'train': [{'context': 'hi ', 'target': 'a'}] * 8,
                'validation': [{'context': 'there ', 'target': 'a'}] * 4,
                'test': [{'context': 'bye ', 'target': 'b'}], 'dictionary': ['a']}
        legacy_digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True,
                                                  separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(_fingerprint(data), legacy_digest)
        self.assertEqual(_fingerprint(dict(data, domain='english')), legacy_digest)
        english_config = dict(config(2), domain='english', max_order=3)
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            train(english_config, data, whole)
            train(dict(english_config, generations=1), data, split)
            path = Path(split) / 'checkpoint.json'
            old = json.loads(path.read_text())
            old['config'].pop('domain')
            path.write_text(json.dumps(old), encoding='utf-8')
            train(english_config, data, split, resume=True)
            a = json.loads((Path(whole) / 'checkpoint.json').read_text())
            b = json.loads(path.read_text())
            for key in ('population', 'rng_state', 'champion', 'history', 'ancestry'):
                self.assertEqual(a[key], b[key], key)

    def test_exception_is_recorded_as_failed_in_checkpoint_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config(), code_dataset(), directory)
            with patch.object(trainer, '_step', side_effect=RuntimeError('deliberate test failure')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                    trainer.run()
            for name in ('checkpoint.json', 'status.json'):
                state = json.loads((Path(directory) / name).read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['error']['type'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
