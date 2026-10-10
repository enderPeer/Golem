"""Meaningful budget, inheritance, split and replay tests for neural experiments."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from english_world import neural_training as training


def dataset():
    return {'texts': {'train': ['alpha beta gamma. ' * 5, 'a small training document.'],
                      'validation': ['held out validation document.'],
                      'test': ['final unseen evaluation document.']}}


def config():
    return {'max_total_updates': 8, 'rounds': 2, 'batch_size': 2, 'sequence_length': 12,
            'seed': 5, 'threads': 1, 'validation_batches': 1, 'test_batches': 1,
            'checkpoint_every': 2, 'max_seconds': 0,
            'model': {'embedding_dim': 8, 'hidden_size': 12, 'num_layers': 1, 'dropout': .1}}


class NeuralTrainingTests(unittest.TestCase):
    def test_equal_operation_budgets_and_actual_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            result = training.compare(config(), dataset(), directory)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(all(result['matched_operation_counts'].values()))
            for arm in result['arms'].values():
                self.assertEqual(arm['updates'], 8)
                self.assertEqual(arm['processed_training_tokens'], 8 * 2 * 12)
                self.assertEqual(arm['validation_forward_batches'], 8)
                self.assertEqual(arm['test']['forward_batches'], 1)
                self.assertGreater(arm['cpu_seconds'], 0)
            inheritance = result['arms']['evolution']['inheritance']
            self.assertEqual(len(inheritance), 1)
            self.assertTrue(inheritance[0]['copied_optimizer_state'])
            self.assertEqual(inheritance[0]['global_updates'], 4)
            self.assertNotEqual(inheritance[0]['parent'], inheritance[0]['child'])
            self.assertTrue((Path(directory) / 'evolution_champion.pt').is_file())

    def test_update_batches_identical_and_heldout_not_used_for_gradient(self):
        received = []
        original = training._loss

        def observe(model, batch):
            if model.training:
                received.append((batch['inputs'].clone(), batch['targets'].clone(), batch['loss_mask'].clone()))
            return original(model, batch)

        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_loss', side_effect=observe):
            training.compare(config(), dataset(), directory)
        self.assertEqual(len(received), 16)
        for a, b in zip(received[:8], received[8:]):
            for left, right in zip(a, b):
                self.assertTrue(torch.equal(left, right))

    def test_partial_resume_matches_uninterrupted_weights_and_optimizer(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            training.compare(config(), dataset(), whole)
            original = training._step_arm

            def stop_after_step(*args):
                original(*args)
                args[-1]['max_seconds'] = 1e-12

            with patch.object(training, '_step_arm', side_effect=stop_after_step):
                partial = training.compare(config(), dataset(), split)
            self.assertEqual(partial['status'], 'time_limit')
            self.assertGreater(partial['arms']['ordinary']['updates'], 0)
            self.assertLess(partial['arms']['ordinary']['updates'], config()['max_total_updates'])
            training.compare(config(), dataset(), split, resume=True)
            a = torch.load(Path(whole) / 'training.pt', weights_only=True)
            b = torch.load(Path(split) / 'training.pt', weights_only=True)
            for name in training.ARM_NAMES:
                left, right = a['arms'][name], b['arms'][name]
                self.assertEqual(left['validation_history'], right['validation_history'])
                self.assertEqual(left['inheritance'], right['inheritance'])
                self.assertEqual(left['best']['weight_sha256'], right['best']['weight_sha256'])
                self.assertTrue(torch.equal(left['torch_rng_state'], right['torch_rng_state']))
                for lc, rc in zip(left['candidates'], right['candidates']):
                    self.assertEqual(training._hash_weights(lc['model']), training._hash_weights(rc['model']))
                    for parameter, optimizer_state in lc['optimizer']['state'].items():
                        for key, value in optimizer_state.items():
                            other = rc['optimizer']['state'][parameter][key]
                            self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)

    def test_test_evaluation_waits_until_training_finishes(self):
        calls = []
        original = training._evaluate

        def record(model, docs, cfg, batches, seed):
            calls.append(seed)
            return original(model, docs, cfg, batches, seed)

        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_evaluate', side_effect=record):
            cfg = dict(config(), max_seconds=1e-12)
            result = training.compare(cfg, dataset(), directory)
        self.assertEqual(result['status'], 'time_limit')
        self.assertNotIn(config()['seed'] + 300000, calls)
        self.assertIsNone(result['arms']['ordinary']['test'])

    def test_resume_rejects_changed_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            training.compare(config(), dataset(), directory)
            changed = copy.deepcopy(dataset())
            changed['texts']['train'][0] += 'changed'
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                training.compare(config(), changed, directory, resume=True)

    def test_invalid_or_misrepresented_budgets_rejected(self):
        for overrides in ({'max_total_updates': 9}, {'budget_mode': 'cpu_seconds'}, {'threads': 3}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                training._config({**config(), **overrides})

    def test_memory_smoke_reports_failure_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            result = training.train_memory({'max_updates': 1, 'max_seconds': 10,
                'test_documents': 20, 'batch_size': 4, 'threads': 1,
                'model': {'embedding_dim': 8, 'hidden_size': 12, 'num_layers': 1, 'dropout': 0.}}, directory)
            self.assertFalse(result['goal_achieved'])
            self.assertEqual(set(result['recall']), {'32', '64'})
            for row in result['recall'].values():
                self.assertEqual(row['chance'], .1)
                self.assertGreaterEqual(row['accuracy'], 0)
                self.assertLessEqual(row['accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
