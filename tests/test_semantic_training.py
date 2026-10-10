"""Current-reply-only supervision, genetic controls and deterministic replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world import semantic_training as training
from english_world.semantic_model import ACTS


def dataset():
    data = {'samples': {}, 'episodes': {}}
    for split, word in (('train', 'hello'), ('validation', 'hi'), ('test', 'hey')):
        data['samples'][split] = [
            {'id': split + '-greeting', 'turns': [{'role': 'user', 'content': word}],
             'reply': 'Hello.', 'act': 'greeting'},
            {'id': split + '-unknown', 'turns': [{'role': 'assistant', 'content': 'Your name is Noah.'},
                                               {'role': 'user', 'content': 'What is my name?'}],
             'reply': 'I do not know your name.', 'act': 'name_recall_unknown',
             'frame': {'user_name': 'DO NOT FEED THIS LABEL TO MODEL'}}]
        data['episodes'][split] = []
    return data


def config():
    return {'strategies': ['ordinary', 'evolution'], 'max_total_updates': 8, 'rounds': 2,
            'batch_size': 2, 'threads': 1, 'max_seconds': 0, 'validation_batches': 1,
            'checkpoint_every': 2, 'max_context_tokens': 128, 'max_reply_tokens': 48,
            'validation_episodes': 2, 'test_episodes': 2, 'max_new_tokens': 8,
            'model': {'embedding_dim': 8, 'hidden_size': 12, 'num_layers': 1, 'dropout': .1}}


def rollout(model, data, split, config, should_stop):
    if should_stop():
        raise InterruptedError('test cancellation')
    return {'score': .1, 'false_memory_claims': 0, 'whole_episode_pass_rate': 0., 'split': split}


class SemanticTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_targets_exclude_wrong_prior_assistant_and_frames_are_not_inputs(self):
        encoded = training.encode_samples(dataset(), 'train', training._config(config()))
        unknown = encoded[1]
        self.assertIn(b'Noah', bytes(token for token in unknown['prefix'] if token < 256))
        self.assertNotIn(b'Noah', bytes(token for token in unknown['targets'] if token < 256))
        self.assertNotIn(b'DO NOT FEED', bytes(token for token in unknown['prefix'] if token < 256))
        self.assertIn(b'UNKNOWN', bytes(token for token in unknown['prefix'] if token < 256))
        batch = training.make_batch(encoded)
        self.assertEqual(int(batch['mask'].sum()), sum(len(row['targets']) for row in encoded))
        self.assertEqual(batch['acts'][1].item(), ACTS.index('name_recall_unknown'))

    def test_joint_training_controls_and_bounded_recipe_inheritance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_rollout', side_effect=rollout):
            report = training.train_semantic(config(), dataset(), directory)
            self.assertEqual(report['status'], 'complete')
            for strategy in report['strategies'].values():
                self.assertEqual(strategy['updates'], 8)
                self.assertEqual(strategy['validation_calls'], 8)
                self.assertGreater(strategy['reply_tokens'], 0)
                self.assertEqual(strategy['test']['rollout']['split'], 'test')
            self.assertEqual(len(report['strategies']['evolution']['births']), 2)
            self.assertTrue(all(row['elite_preserved'] for row in report['strategies']['evolution']['births']))
            self.assertTrue((Path(directory) / 'best.pt').is_file())

    def test_partial_resume_preserves_joint_model_and_optimizer(self):
        cfg = {**config(), 'strategies': ['evolution']}
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split, \
             patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), whole)
            original = training._step

            def stop_after_one(*args):
                original(*args)
                args[-3]['max_seconds'] = 1e-12

            with patch.object(training, '_step', side_effect=stop_after_one):
                partial = training.train_semantic(cfg, dataset(), split)
            self.assertEqual(partial['status'], 'time_limit')
            training.train_semantic(cfg, dataset(), split, resume=True)
            a = torch.load(Path(whole) / 'semantic-training.pt', weights_only=True)['strategies']['evolution']
            b = torch.load(Path(split) / 'semantic-training.pt', weights_only=True)['strategies']['evolution']
            for key in ('history', 'births', 'genetic_rng', 'updates', 'reply_tokens', 'prefix_tokens'):
                self.assertEqual(a[key], b[key], key)
            self.assertTrue(torch.equal(a['rng']['cpu'], b['rng']['cpu']))
            for left, right in zip(a['population'], b['population']):
                self.assertEqual(training._hash_weights(left['weights']), training._hash_weights(right['weights']))
                for index, fields in left['optimizer']['state'].items():
                    for key, value in fields.items():
                        other = right['optimizer']['state'][index][key]
                        self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)

    def test_cancelled_rollout_is_not_scored_and_resumes_without_partial_selection(self):
        cfg = {**config(), 'strategies': ['ordinary']}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, '_rollout', side_effect=InterruptedError('stop')):
                report = training.train_semantic(cfg, dataset(), directory)
            self.assertEqual(report['status'], 'time_limit')
            state = torch.load(Path(directory) / 'semantic-training.pt', weights_only=True)['strategies']['ordinary']
            self.assertTrue(state['evaluation_due'])
            self.assertIsNone(state['best'])
            with patch.object(training, '_rollout', side_effect=rollout):
                resumed = training.train_semantic(cfg, dataset(), directory, resume=True)
            self.assertEqual(resumed['status'], 'complete')

    def test_meaning_integer_id_must_match_act_name(self):
        data = dataset()
        data['samples']['train'][0]['meaning_id'] = ACTS.index('identity')
        with self.assertRaisesRegex(ValueError, 'meaning_id'):
            training.encode_samples(data, 'train', training._config(config()))


if __name__ == '__main__':
    unittest.main()
