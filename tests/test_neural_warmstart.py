"""Immutable shared-checkpoint initialization and legacy resume compatibility."""
import tempfile
from pathlib import Path
import unittest

import torch

from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint
from english_world import neural_training as training


def config():
    return {'max_total_updates': 8, 'rounds': 2, 'batch_size': 2, 'sequence_length': 12,
            'threads': 1, 'max_seconds': 0, 'validation_batches': 1, 'test_batches': 1,
            'model': {'embedding_dim': 8, 'hidden_size': 12, 'num_layers': 1, 'dropout': .1}}


def data():
    return {'texts': {'train': ['A dog is an animal. ' * 5],
                      'validation': ['A cat is a mammal.'], 'test': ['A bird is a vertebrate.']}}


class NeuralWarmstartTests(unittest.TestCase):
    def seed_checkpoint(self, directory):
        torch.manual_seed(49)
        source = ByteGRULM(NeuralConfig(**config()['model']))
        checkpoint = Path(directory) / 'seed.pt'
        save_neural_checkpoint(checkpoint, source, {'global_updates': 320, 'selection_split': 'validation'})
        return checkpoint, source

    def test_every_candidate_starts_with_identical_source_weights_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, source = self.seed_checkpoint(directory)
            cfg = training._config({**config(), 'init_checkpoint': str(checkpoint)})
            expected = training._hash_weights(source.state_dict())
            for name in training.ARM_NAMES:
                arm = training._new_arm(name, cfg)
                self.assertEqual(arm['initialization']['weight_sha256'], expected)
                self.assertEqual(arm['initialization']['source_metadata']['global_updates'], 320)
                for candidate in arm['candidates']:
                    self.assertEqual(training._hash_weights(candidate['model']), expected)
                    self.assertFalse(candidate['optimizer']['state'])
                    self.assertEqual(candidate['ancestral_updates'], 0)
                if name == 'evolution':
                    first = next(iter(arm['candidates'][0]['model'].values()))
                    second = next(iter(arm['candidates'][1]['model'].values()))
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_source_is_immutable_and_all_operation_budgets_still_match(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.seed_checkpoint(directory)
            original = checkpoint.read_bytes()
            cfg = {**config(), 'init_checkpoint': str(checkpoint)}
            result = training.compare(cfg, data(), Path(directory) / 'run')
            self.assertTrue(all(result['matched_operation_counts'].values()))
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(result['arms']['ordinary']['initialization'], result['arms']['evolution']['initialization'])
            self.assertEqual(result['arms']['ordinary']['updates'], 8)

    def test_changed_source_at_same_path_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, source = self.seed_checkpoint(directory)
            cfg = {**config(), 'init_checkpoint': str(checkpoint)}
            training.compare(cfg, data(), Path(directory) / 'run')
            with torch.no_grad():
                source.output.bias[0] += 1
            save_neural_checkpoint(checkpoint, source)
            with self.assertRaisesRegex(ValueError, 'init_checkpoint_sha256'):
                training.compare(cfg, data(), Path(directory) / 'run', resume=True)

    def test_architecture_mismatch_rejected_but_explicit_dropout_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self.seed_checkpoint(directory)
            cfg = training._config({**config(), 'init_checkpoint': str(checkpoint),
                                    'model': {**config()['model'], 'dropout': .2}})
            arm = training._new_arm('ordinary', cfg)
            self.assertEqual(arm['candidates'][0]['dropout'], .2)
            cfg['model']['hidden_size'] = 13
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                training._new_arm('ordinary', cfg)

    def test_legacy_checkpoint_without_initialization_fields_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = training.compare(config(), data(), directory)
            state = torch.load(Path(directory) / 'training.pt', weights_only=True)
            self.assertNotIn('init_checkpoint', state['config'])
            self.assertNotIn('initialization', state['arms']['ordinary'])
            resumed = training.compare({**config(), 'init_checkpoint': None}, data(), directory, resume=True)
            self.assertEqual(resumed['arms']['ordinary']['best_validation'], result['arms']['ordinary']['best_validation'])
            self.assertTrue(all(resumed['matched_operation_counts'].values()))


if __name__ == '__main__':
    unittest.main()
