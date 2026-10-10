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

    def assert_strategy_equal(self, left, right):
        for key in ('history', 'births', 'genetic_rng', 'updates', 'reply_tokens', 'prefix_tokens'):
            self.assertEqual(left[key], right[key], key)
        self.assertTrue(torch.equal(left['rng']['cpu'], right['rng']['cpu']))
        for a, b in zip(left['population'], right['population']):
            self.assertEqual(training._hash_weights(a['weights']), training._hash_weights(b['weights']))
            self.assertEqual(a['recipe'], b['recipe'])
            self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
            for index, fields in a['optimizer']['state'].items():
                for key, value in fields.items():
                    other = b['optimizer']['state'][index][key]
                    self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)

    def test_round_robin_short_budget_trains_both_without_checkpointing_each_block(self):
        cfg = {**config(), 'max_total_updates': 16, 'max_seconds': 5.,
               'checkpoint_every': 99, 'checkpoint_seconds': 3600., 'evaluate_test_on_complete': False}
        clock, order = [100.], []
        original = training._step
        def measured(*args):
            evaluation = args[1]['evaluation_due']
            original(*args)
            clock[0] += .2 if evaluation else 1.
            if evaluation:
                order.append(args[0])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(training, '_rollout', side_effect=rollout), \
             patch.object(training, '_step', side_effect=measured), \
             patch.object(training.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(training, '_checkpoint', wraps=training._checkpoint) as checkpoints:
            report = training.train_semantic(cfg, dataset(), directory)
            self.assertEqual(report['status'], 'time_limit')
            self.assertEqual(report['config']['strategy_schedule'], 'round_robin')
            self.assertEqual(order, ['ordinary', 'evolution'])
            self.assertEqual(report['strategies']['ordinary']['updates'], 3)
            self.assertEqual(report['strategies']['evolution']['updates'], 2)
            self.assertEqual(checkpoints.call_count, 2)  # Initial and interrupted state only.

    def test_round_robin_and_sequential_preserve_identical_full_strategy_results(self):
        cfg = {**config(), 'evaluate_test_on_complete': False,
               'model': {**config()['model'], 'num_layers': 2}}
        with tempfile.TemporaryDirectory() as rr, tempfile.TemporaryDirectory() as seq, \
             patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), rr)
            training.train_semantic({**cfg, 'strategy_schedule': 'sequential'}, dataset(), seq)
            a = torch.load(Path(rr) / 'semantic-training.pt', weights_only=True)
            b = torch.load(Path(seq) / 'semantic-training.pt', weights_only=True)
            for name in cfg['strategies']:
                self.assert_strategy_equal(a['strategies'][name], b['strategies'][name])
            self.assertEqual(a['validation_protocol_hash'], b['validation_protocol_hash'])

    def test_round_robin_resume_at_block_boundary_and_mid_evaluation_is_exact(self):
        cfg = {**config(), 'evaluate_test_on_complete': False}
        original = training._step
        with tempfile.TemporaryDirectory() as whole, patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), whole)
            expected = torch.load(Path(whole) / 'semantic-training.pt', weights_only=True)
            for where in ('after_block', 'during_evaluation'):
                with self.subTest(where=where), tempfile.TemporaryDirectory() as partial:
                    def interrupt(*args):
                        evaluation = args[1]['evaluation_due']
                        if args[0] == 'evolution' and evaluation and where == 'during_evaluation':
                            raise InterruptedError('mid-validation shutdown')
                        original(*args)
                        if args[0] == 'evolution' and evaluation:
                            args[-3]['max_seconds'] = 1e-12
                    with patch.object(training, '_step', side_effect=interrupt):
                        result = training.train_semantic(cfg, dataset(), partial)
                    self.assertEqual(result['status'], 'time_limit')
                    state = torch.load(Path(partial) / 'semantic-training.pt', weights_only=True)
                    self.assertEqual(state['schedule_cursor'], 0 if where == 'after_block' else 1)
                    self.assertEqual(state['strategies']['evolution']['evaluation_due'], where == 'during_evaluation')
                    training.train_semantic(cfg, dataset(), partial, resume=True)
                    actual = torch.load(Path(partial) / 'semantic-training.pt', weights_only=True)
                    for name in cfg['strategies']:
                        self.assert_strategy_equal(expected['strategies'][name], actual['strategies'][name])

    def test_legacy_incomplete_checkpoint_resumes_sequential_without_explicit_option(self):
        cfg = {**config(), 'strategy_schedule': 'sequential', 'evaluate_test_on_complete': False}
        original = training._step
        def stop(*args):
            original(*args)
            args[-3]['max_seconds'] = 1e-12
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as partial, \
             patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), whole)
            with patch.object(training, '_step', side_effect=stop):
                training.train_semantic(cfg, dataset(), partial)
            path = Path(partial) / 'semantic-training.pt'
            state = torch.load(path, weights_only=True)
            state['config'].pop('strategy_schedule')
            state['config'].pop('meaning_condition_gain')
            state.pop('schedule_cursor')
            torch.save(state, path)
            implicit = {key: value for key, value in cfg.items() if key != 'strategy_schedule'}
            with self.assertRaisesRegex(ValueError, 'strategy_schedule'):
                training.train_semantic({**implicit, 'strategy_schedule': 'round_robin'}, dataset(), partial, resume=True)
            report = training.train_semantic(implicit, dataset(), partial, resume=True)
            self.assertEqual(report['config']['strategy_schedule'], 'sequential')
            self.assertEqual(report['config']['meaning_condition_gain'], 1.)
            expected = torch.load(Path(whole) / 'semantic-training.pt', weights_only=True)
            actual = torch.load(path, weights_only=True)
            for name in cfg['strategies']:
                self.assert_strategy_equal(expected['strategies'][name], actual['strategies'][name])

    def test_meaning_gain_is_bounded_explicit_and_preserved_in_training_exports(self):
        from english_world.semantic_model import load_semantic_checkpoint
        self.assertEqual(training._config(config())['meaning_condition_gain'], 1.)
        for invalid in (0., 65., float('nan'), float('inf')):
            with self.subTest(gain=invalid), self.assertRaisesRegex(ValueError, 'meaning_condition_gain'):
                training._config({**config(), 'meaning_condition_gain': invalid})
        cfg = {**config(), 'strategies': ['ordinary'], 'max_total_updates': 4, 'rounds': 1,
               'meaning_condition_gain': 32., 'evaluate_test_on_complete': False}
        observed = []
        def inspect(model, *args):
            observed.append(model.meaning_condition_gain)
            return rollout(model, *args)
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_rollout', side_effect=inspect):
            report = training.train_semantic(cfg, dataset(), directory)
            exported, _ = load_semantic_checkpoint(Path(directory) / 'best.pt')
            self.assertEqual(report['config']['meaning_condition_gain'], 32.)
            self.assertEqual(exported.meaning_condition_gain, 32.)
            self.assertEqual(observed, [32.] * 4)
            with self.assertRaisesRegex(ValueError, 'meaning_condition_gain'):
                training.train_semantic({**cfg, 'meaning_condition_gain': 1.}, dataset(), directory, resume=True)

    def test_explicit_teacher_and_repair_overlays_are_combined_and_hash_pinned(self):
        from english_world import world_replay as replay
        from english_world.semantic_model import SemanticGRU, save_semantic_checkpoint
        from english_world.neural_model import NeuralConfig
        from tests.test_world_replay import base_data
        data = base_data()
        for split in ('validation', 'test'):
            data['samples'][split] = dataset()['samples'][split]
            data['episodes'][split] = []
        cfg = {**config(), 'strategies': ['ordinary'], 'max_total_updates': 4, 'rounds': 1,
               'max_context_tokens': 512, 'evaluate_test_on_complete': False}
        self.assertNotIn('teacher_overlay', training._config(cfg))
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_rollout', side_effect=rollout):
            root = Path(directory)
            source = root / 'source.pt'
            save_semantic_checkpoint(source, SemanticGRU(NeuralConfig(**cfg['model'])))
            paths = []
            for kind in ('repair', 'teacher'):
                path = root / (kind + '.json')
                with patch.object(replay, 'generate_semantic', return_value={'text': kind + ' wrong answer'}):
                    replay.build_repair_overlay(source, data, path, seed=3, max_episodes=1)
                paths.append(path)
            cfg.update(repair_overlay=str(paths[0]), teacher_overlay=str(paths[1]))
            expected = replay.apply_repair_overlays(data, paths)
            self.assertEqual(len(expected['samples']['train']), len(data['samples']['train']) + 4)
            result = training.train_semantic(cfg, data, root / 'run')
            self.assertEqual(result['status'], 'complete')
            for kind, path in zip(('repair', 'teacher'), paths):
                self.assertEqual(result['config'][kind + '_overlay_sha256'], training._checkpoint_hash(path))
            # Even a semantically equivalent replacement of the teacher file
            # must be explicit rather than silently changing a resumed run.
            paths[1].write_text(paths[1].read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'teacher_overlay_sha256'):
                training.train_semantic(cfg, data, root / 'run', resume=True)

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

    def test_long_run_defers_test_completely(self):
        cfg = {**config(), 'strategies': ['ordinary'], 'evaluate_test_on_complete': False}
        data = dataset()
        data['samples']['test'] = []  # Preparing/evaluating this would fail.
        splits = []
        def record(model, data, split, config, should_stop):
            splits.append(split)
            return rollout(model, data, split, config, should_stop)
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_rollout', side_effect=record):
            result = training.train_semantic(cfg, data, directory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(splits, ['validation'] * 8)
            self.assertIsNone(result['strategies']['ordinary']['test'])
            self.assertIn('deferred', result['test_policy'])
            artifact = torch.load(Path(directory) / 'best.pt', weights_only=True)
            self.assertEqual(artifact['metadata']['validation_protocol_hash'], result['validation_protocol_hash'])

    def test_frozen_validation_protocol_ignores_training_seed_and_train_overlay(self):
        a = dataset()
        b = dataset()
        b['samples']['train'].append(dict(b['samples']['train'][0], id='new-training-repair'))
        fixed = {'validation_seed': 17, 'validation_nll_seed': 23, 'validation_batch_size': 2}
        ca = training._config({**config(), **fixed, 'seed': 1, 'batch_size': 2})
        cb = training._config({**config(), **fixed, 'seed': 9, 'batch_size': 8})
        self.assertEqual(training._validation_protocol(a, ca), training._validation_protocol(b, cb))
        b['samples']['validation'][0]['reply'] = 'Changed held-out answer.'
        self.assertNotEqual(training._validation_protocol(a, ca)[1], training._validation_protocol(b, cb)[1])

    def test_legacy_resume_defaults_remain_compatible(self):
        cfg = {**config(), 'strategies': ['ordinary']}
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), directory)
            path = Path(directory) / 'semantic-training.pt'
            state = torch.load(path, weights_only=True)
            for key in ('evaluate_test_on_complete', 'validation_seed', 'validation_nll_seed', 'validation_batch_size'):
                state['config'].pop(key)
            state.pop('validation_protocol')
            state.pop('validation_protocol_hash')
            torch.save(state, path)
            result = training.train_semantic(cfg, dataset(), directory, resume=True)
            self.assertEqual(result['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
