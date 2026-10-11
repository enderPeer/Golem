"""Current-reply-only supervision, genetic controls and deterministic replay."""
from pathlib import Path
import copy
import random
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

    def test_verified_name_spans_are_exact_grounded_and_utf8_aligned(self):
        for name, reply in (
                ('Zoë', '🌻 Your name is Zoë, not Zoël or Zoe.'),
                ('O\u2019Neil', 'Your name is O\u2019Neil, not Neil.'),
                ('Ana-Marie', 'Ana-Marie, not Jean-Ana-Marie.'),
                ('李', 'Your name is 李.')):
            turns = [{'role': 'user', 'content': 'My name is ' + name},
                     {'role': 'assistant', 'content': 'Your name is Noah.'},
                     {'role': 'user', 'content': 'What is my name?'}]
            with self.subTest(name=name):
                mask = training._verified_name_mask(turns, reply, 'name_recall_known')
                encoded = reply.encode('utf-8')
                start = len(reply[:reply.index(name)].encode('utf-8'))
                expected = list(range(start, start + len(name.encode('utf-8'))))
                self.assertEqual([i for i, value in enumerate(mask) if value], expected)
                self.assertEqual(len(mask), len(encoded) + 1)
                self.assertFalse(mask[-1])
                self.assertEqual(bytes(value for value, selected in zip(encoded, mask) if selected).decode(), name)
        turns = [{'role': 'user', 'content': 'Call me Ana'}]
        reply = "Ana, not Ana-Maria, O'Ana, O\u2019Ana, Anabel, or ana."
        self.assertEqual(sum(training._verified_name_mask(turns, reply, 'name_set')), 3)

    def test_name_mask_ignores_assistant_facts_forgotten_names_and_unrelated_acts(self):
        assistant_only = [{'role': 'assistant', 'content': 'My name is Noah. Your name is Noah.'},
                          {'role': 'user', 'content': 'What is my name?'}]
        self.assertFalse(any(training._verified_name_mask(assistant_only, 'Your name is Noah.', 'name_recall_known')))
        turns = [{'role': 'user', 'content': 'My name is Maya'},
                 {'role': 'assistant', 'content': 'Your name is Noah.'},
                 {'role': 'user', 'content': 'Actually, my name is Zoë'}]
        reply = 'Zoë, not Maya or Noah.'
        mask = training._verified_name_mask(turns, reply, 'name_correction')
        self.assertEqual(sum(mask), len('Zoë'.encode('utf-8')))
        turns.append({'role': 'user', 'content': 'Please forget my name'})
        self.assertFalse(any(training._verified_name_mask(turns, reply, 'name_recall_known')))
        for name, act, reply in (('Golem', 'identity', 'I am Golem.'), ('May', 'topic_chat', 'May I help?')):
            turns = [{'role': 'user', 'content': 'Call me ' + name}]
            self.assertFalse(any(training._verified_name_mask(turns, reply, act)))

    def test_encoded_name_mask_ignores_frame_and_batch_legacy_padding_end_are_zero(self):
        rows = [
            {'id': 'known', 'turns': [{'role': 'user', 'content': 'My name is Zoë'}],
             'reply': 'Hello Zoë.', 'act': 'name_set', 'frame': {'user_name': 'FORGED'}},
            {'id': 'unknown', 'turns': [{'role': 'assistant', 'content': 'Your name is Zoë.'},
                                     {'role': 'user', 'content': 'Who am I?'}],
             'reply': 'Zoë.', 'act': 'name_recall_known', 'frame': {'user_name': 'Zoë'}}]
        encoded = training.encode_samples({'samples': {'train': rows}}, 'train', training._config(config()))
        self.assertEqual(sum(encoded[0]['verified_name_mask']), 4)
        self.assertFalse(any(encoded[1]['verified_name_mask']))
        encoded.append({key: value for key, value in encoded[0].items() if key != 'verified_name_mask'})
        batch = training.make_batch(encoded)
        self.assertEqual(batch['verified_name_mask'].dtype, torch.bool)
        self.assertEqual(int(batch['verified_name_mask'].sum()), 4)
        self.assertFalse(batch['verified_name_mask'][2].any())
        self.assertFalse(batch['verified_name_mask'][~batch['mask']].any())
        self.assertFalse(batch['verified_name_mask'][batch['targets'] == training.END].any())
        malformed = copy.deepcopy(encoded[0])
        malformed['verified_name_mask'][-1] = True
        with self.assertRaisesRegex(ValueError, 'END'):
            training.make_batch([malformed])

    def test_default_name_weight_loss_and_gradients_are_bit_identical_to_legacy(self):
        from english_world.semantic_model import SemanticGRU
        from english_world.neural_model import NeuralConfig
        rows = [{'turns': [{'role': 'user', 'content': 'Call me Maya'}], 'reply': 'Maya.', 'act': 'name_set'}]
        encoded = training.encode_samples({'samples': {'train': rows}}, 'train', training._config(config()))
        batch = training.make_batch(encoded)
        model = SemanticGRU(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=2, dropout=.1))
        model.train()
        recipe = {'first_response_weight': 3., 'intent_loss_weight': 1.5, 'verified_name_weight': 1.}
        rng = torch.get_rng_state()
        logits, meaning = model(batch['prefix'], batch['prefix_lengths'], batch['reply_inputs'])
        errors = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
            batch['targets'].reshape(-1), reduction='none').reshape_as(batch['targets'])
        mask = batch['mask'].float()
        weights = mask.clone()
        weights[:, :3] *= 3.
        old_reply = (errors * mask).sum() / mask.sum()
        old_total = (errors * weights).sum() / weights.sum() + 1.5 * torch.nn.functional.cross_entropy(meaning, batch['acts'])
        parameters = tuple(model.parameters())
        old_gradients = torch.autograd.grad(old_total, parameters)
        torch.set_rng_state(rng)
        new_total, metrics = training.joint_loss(model, batch, recipe, first_response_bytes=3)
        new_gradients = torch.autograd.grad(new_total, parameters)
        self.assertTrue(torch.equal(old_total, new_total))
        self.assertTrue(torch.equal(old_reply, metrics['reply_loss']))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_gradients, new_gradients)))

    def test_opt_in_changes_only_relative_named_token_weights_not_reported_nll(self):
        class FixedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.arange(4 * 260, dtype=torch.float32).reshape(1, 4, 260) / 1000)
                self.meaning = torch.nn.Parameter(torch.zeros(1, len(ACTS)))
            def forward(self, *args):
                return self.logits, self.meaning
        batch = {'prefix': torch.zeros(1, 1, dtype=torch.long), 'prefix_lengths': torch.tensor([1]),
                 'reply_inputs': torch.tensor([[1, 2, training.END]]),
                 'targets': torch.tensor([[1, 2, training.END, training.PAD]]),
                 'mask': torch.tensor([[True, True, True, False]]),
                 'verified_name_mask': torch.tensor([[False, True, False, False]]), 'acts': torch.tensor([0])}
        model = FixedLogits()
        baseline = {'first_response_weight': 2., 'intent_loss_weight': 1.}
        loss1, metrics1 = training.joint_loss(model, batch, baseline, first_response_bytes=2)
        gradient1 = torch.autograd.grad(loss1, model.logits)[0]
        loss4, metrics4 = training.joint_loss(model, batch, {**baseline, 'verified_name_weight': 4.}, first_response_bytes=2)
        gradient4 = torch.autograd.grad(loss4, model.logits)[0]
        self.assertTrue(torch.equal(metrics1['reply_loss'], metrics4['reply_loss']))
        self.assertEqual(metrics1['reply_tokens'], metrics4['reply_tokens'])
        # Existing prefix weights [2,2,1,0] become [2,8,1,0]. Only the
        # name's relative factor changes; the weighted mean also rescales all.
        factor = 5. / 11.
        self.assertTrue(torch.allclose(gradient4[:, 0], gradient1[:, 0] * factor, atol=1e-8))
        self.assertTrue(torch.allclose(gradient4[:, 1], gradient1[:, 1] * factor * 4, atol=1e-8))
        self.assertTrue(torch.allclose(gradient4[:, 2], gradient1[:, 2] * factor, atol=1e-8))
        self.assertFalse(gradient4[:, 3].any())
        legacy_batch = {key: value for key, value in batch.items() if key != 'verified_name_mask'}
        loss_no_mask, _ = training.joint_loss(model, legacy_batch, {**baseline, 'verified_name_weight': 4.}, first_response_bytes=2)
        self.assertTrue(torch.equal(loss1, loss_no_mask))

    def test_name_weight_is_bounded_recorded_and_never_genetically_mutated(self):
        self.assertEqual(training._config(config())['verified_name_weight'], 1.)
        for invalid in (0, .9, 8.1, True, float('nan'), float('inf')):
            with self.subTest(weight=invalid), self.assertRaisesRegex(ValueError, 'verified_name_weight'):
                training._config({**config(), 'verified_name_weight': invalid})
        for weight in (1., 4., 8.):
            recipe = training._recipe(training._config({**config(), 'verified_name_weight': weight}), ['name_set'])
            for seed in range(8):
                self.assertEqual(training.mutate_recipe(recipe, random.Random(seed))['verified_name_weight'], weight)
        recipe = training._recipe(training._config(config()), ['name_set'])
        legacy = {key: value for key, value in recipe.items() if key != 'verified_name_weight'}
        old_rng, new_rng = random.Random(7), random.Random(7)
        old_child = training.mutate_recipe(legacy, old_rng)
        new_child = training.mutate_recipe(recipe, new_rng)
        new_child.pop('verified_name_weight')
        self.assertEqual(new_child, old_child)
        self.assertEqual(new_rng.getstate(), old_rng.getstate())

    def test_legacy_resume_without_name_weight_preserves_model_optimizer_and_history(self):
        cfg = {**config(), 'strategies': ['ordinary'], 'max_total_updates': 4, 'rounds': 1,
               'evaluate_test_on_complete': False}
        original = training._step
        def stop_after_step(*args):
            original(*args)
            args[-3]['max_seconds'] = 1e-12
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as partial, \
             patch.object(training, '_rollout', side_effect=rollout):
            training.train_semantic(cfg, dataset(), whole)
            with patch.object(training, '_step', side_effect=stop_after_step):
                training.train_semantic(cfg, dataset(), partial)
            path = Path(partial) / 'semantic-training.pt'
            state = torch.load(path, weights_only=True)
            state['config'].pop('verified_name_weight')
            for candidate in state['strategies']['ordinary']['population']:
                candidate['recipe'].pop('verified_name_weight')
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'verified_name_weight'):
                training.train_semantic({**cfg, 'verified_name_weight': 4.}, dataset(), partial, resume=True)
            report = training.train_semantic(cfg, dataset(), partial, resume=True)
            self.assertEqual(report['config']['verified_name_weight'], 1.)
            expected = torch.load(Path(whole) / 'semantic-training.pt', weights_only=True)['strategies']['ordinary']
            actual = torch.load(path, weights_only=True)['strategies']['ordinary']
            for candidate in expected['population']:
                candidate['recipe'].pop('verified_name_weight')
            self.assert_strategy_equal(expected, actual)

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
