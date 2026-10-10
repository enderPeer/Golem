"""Chat-objective selection, bounded genetics, matched updates and safe replay."""
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from english_world import chat_training as training
from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint


def dataset():
    result = {'texts': {}, 'conversations': {}, 'chat_cases': {}}
    for split, prompt in (('train', 'hi'), ('validation', 'hello'), ('test', 'hey')):
        records, cases, texts = [], [], []
        for intent, question, answer in (('greeting', prompt, 'hi'), ('identity', prompt + ' name?', 'golem')):
            identity = split + '-' + intent
            turns = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            records.append({'id': identity, 'intent': intent, 'family_id': identity, 'turns': turns})
            texts.append('\n'.join(turn['role'] + ': ' + turn['content'] for turn in turns))
            cases.append({'id': identity, 'intent': intent, 'family_id': identity,
                'turns': turns[:-1], 'reference': answer, 'rubric': {'required_any': [answer], 'max_response_bytes': 24}})
        result['texts'][split], result['conversations'][split], result['chat_cases'][split] = texts, records, cases
    return result


def config():
    return {'max_total_updates': 8, 'rounds': 2, 'batch_size': 2, 'sequence_length': 32,
            'threads': 1, 'device': 'cpu', 'seed': 3, 'max_seconds': 0,
            'validation_batches': 1, 'validation_cases': 2, 'test_cases': 2, 'max_new_tokens': 8,
            'checkpoint_every': 2, 'checkpoint_seconds': 600,
            'weight_mutation_probability': 1., 'crossover_probability': 1.,
            'model': {'embedding_dim': 8, 'hidden_size': 12, 'num_layers': 1, 'dropout': .1}}


class ChatTrainingTests(unittest.TestCase):
    def test_three_strategies_have_matched_update_budgets_and_real_genetic_operators(self):
        with tempfile.TemporaryDirectory() as directory:
            result = training.train_chat(config(), dataset(), directory)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['comparison_complete'])
            self.assertTrue(result['matched_counts']['updates'])
            self.assertTrue(result['matched_counts']['processed_training_tokens'])
            self.assertTrue(result['matched_counts']['generated_validation_cases'])
            for strategy in result['strategies'].values():
                self.assertEqual(strategy['updates'], 8)
                self.assertEqual(strategy['processed_training_tokens'], 8 * 2 * 32)
                self.assertEqual(strategy['generated_validation_calls'], 8)
                self.assertEqual(strategy['test']['chat']['split'], 'test')
            births = result['strategies']['evolution']['births']
            self.assertEqual(len(births), 2)
            for birth in births:
                operators = {item['operator'] for item in birth['operators']}
                self.assertTrue({'module_crossover', 'bounded_weight_noise', 'bounded_recipe_mutation'} <= operators)
                self.assertTrue(birth['elite_preserved'])
                self.assertTrue(birth['optimizer_reset_parameters'])
            self.assertTrue((Path(directory) / 'population.json').is_file())
            self.assertTrue((Path(directory) / 'best.pt').is_file())

    def test_mutation_genome_stays_bounded_and_does_not_mutate_parent(self):
        cfg = training._config(config())
        parent = training._initial_gene(cfg, ['identity', 'greeting'])
        saved = copy.deepcopy(parent)
        rng = random.Random(18)
        child = parent
        for _ in range(200):
            child = training.mutate_genome(child, rng)
            for name in ('learning_rate', 'dropout', 'weight_decay', 'gradient_clip'):
                low, high = training.GENE_BOUNDS[name]
                self.assertGreaterEqual(child[name], low)
                self.assertLessEqual(child[name], high)
            self.assertTrue(all(.25 <= value <= 4 for value in child['intent_weights'].values()))
        self.assertEqual(parent, saved)

    def test_partial_cpu_resume_matches_models_optimizer_rng_and_births(self):
        cfg = {**config(), 'strategies': ['evolution']}
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            training.train_chat(cfg, dataset(), whole)
            original = training._step

            def interrupt_after_training_step(*args, **kwargs):
                original(*args, **kwargs)
                args[-2]['max_seconds'] = 1e-12

            with patch.object(training, '_step', side_effect=interrupt_after_training_step):
                interrupted = training.train_chat(cfg, dataset(), split)
            self.assertEqual(interrupted['status'], 'time_limit')
            training.train_chat(cfg, dataset(), split, resume=True)
            a = torch.load(Path(whole) / 'chat-training.pt', weights_only=True)['strategies']['evolution']
            b = torch.load(Path(split) / 'chat-training.pt', weights_only=True)['strategies']['evolution']
            for key in ('history', 'births', 'genetic_rng_state', 'test', 'updates', 'batch_index'):
                self.assertEqual(a[key], b[key], key)
            self.assertTrue(torch.equal(a['rng_state']['cpu'], b['rng_state']['cpu']))
            for left, right in zip(a['candidates'], b['candidates']):
                self.assertEqual(left['genome'], right['genome'])
                self.assertEqual(training._hash_weights(left['model_state']), training._hash_weights(right['model_state']))
                for index, states in left['optimizer_state']['state'].items():
                    for name, value in states.items():
                        other = right['optimizer_state']['state'][index][name]
                        self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)

    def test_warm_start_is_immutable_and_changed_source_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pt'
            model = ByteGRULM(NeuralConfig(**config()['model']))
            save_neural_checkpoint(source, model)
            original = source.read_bytes()
            cfg = {**config(), 'strategies': ['ordinary'], 'init_checkpoint': str(source)}
            report = training.train_chat(cfg, dataset(), Path(directory) / 'run')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report['initialization']['mode'], 'shared_golem_checkpoint')
            with torch.no_grad():
                model.output.bias[0] += 1
            save_neural_checkpoint(source, model)
            with self.assertRaisesRegex(ValueError, 'init_checkpoint_sha256'):
                training.train_chat(cfg, dataset(), Path(directory) / 'run', resume=True)

    def test_training_has_assistant_only_targets_and_no_test_selection(self):
        seen_splits = []
        original = training._chat_evaluate

        def record(model, data, split, cfg, should_stop=None):
            seen_splits.append(split)
            return original(model, data, split, cfg, should_stop=should_stop)

        cfg = {**config(), 'strategies': ['ordinary']}
        docs, buckets, _ = training._prepare(dataset(), training._config(cfg))
        from english_world.neural_model import ASSISTANT, USER
        for doc in docs['train']:
            first_answer = doc.tokens.index(ASSISTANT) + 1
            self.assertEqual(doc.tokens[0], USER)
            self.assertFalse(any(doc.loss_mask[:first_answer]))
            self.assertTrue(all(doc.loss_mask[first_answer:]))
        self.assertEqual(set(buckets), {'greeting', 'identity'})
        with tempfile.TemporaryDirectory() as directory, patch.object(training, '_chat_evaluate', side_effect=record):
            training.train_chat(cfg, dataset(), directory)
        self.assertEqual(seen_splits, ['validation'] * 8 + ['test'])

    def test_gpu_allocator_limit_configured_before_device_use_without_allocating(self):
        cfg = training._config({**config(), 'device': 'cuda:0', 'gpu_memory_limit_gib': 2})
        calls = []
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=8 * 1024**3)), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction', side_effect=lambda fraction, device: calls.append(('limit', fraction))), \
             patch.object(torch.cuda, 'set_device', side_effect=lambda device: calls.append(('device', str(device)))):
            self.assertEqual(str(training._device(cfg)), 'cuda:0')
        self.assertEqual(calls, [('limit', .25), ('device', 'cuda:0')])

    def test_cancelled_validation_is_not_selected_and_is_resumable(self):
        cfg = {**config(), 'strategies': ['ordinary']}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, '_chat_evaluate', side_effect=InterruptedError('bounded cancellation')):
                result = training.train_chat(cfg, dataset(), directory)
            self.assertEqual(result['status'], 'time_limit')
            state = torch.load(Path(directory) / 'chat-training.pt', weights_only=True)['strategies']['ordinary']
            self.assertTrue(state['evaluation_due'])
            self.assertIsNone(state['best'])
            self.assertEqual(state['generated_validation_calls'], 0)
            resumed = training.train_chat(cfg, dataset(), directory, resume=True)
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(resumed['strategies']['ordinary']['generated_validation_calls'], 8)

    def test_cancelled_test_does_not_publish_partial_test_or_repeat_training(self):
        cfg = {**config(), 'strategies': ['ordinary']}
        real = training._chat_evaluate

        def cancel_test(model, data, split, cfg, should_stop=None):
            if split == 'test':
                raise InterruptedError('test cancelled')
            return real(model, data, split, cfg, should_stop=should_stop)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, '_chat_evaluate', side_effect=cancel_test):
                result = training.train_chat(cfg, dataset(), directory)
            self.assertEqual(result['status'], 'time_limit')
            self.assertEqual(result['strategies']['ordinary']['updates'], 8)
            self.assertIsNone(result['strategies']['ordinary']['test'])
            resumed = training.train_chat(cfg, dataset(), directory, resume=True)
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(resumed['strategies']['ordinary']['updates'], 8)
            self.assertEqual(resumed['strategies']['ordinary']['generated_validation_calls'], 8)

    def test_nonfinite_gradients_fail_before_corrupting_weights(self):
        cfg = {**config(), 'strategies': ['ordinary']}
        real = training._runtime

        def poison(strategy, config, device):
            models, optimizers = real(strategy, config, device)
            models[0].output.weight.register_hook(lambda gradient: gradient * float('inf'))
            return models, optimizers

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, '_runtime', side_effect=poison):
                with self.assertRaises(RuntimeError):
                    training.train_chat(cfg, dataset(), directory)
            state = torch.load(Path(directory) / 'chat-training.pt', weights_only=True)
            self.assertEqual(state['status'], 'failed')
            strategy = state['strategies']['ordinary']
            self.assertEqual(strategy['updates'], 0)
            self.assertTrue(all(torch.isfinite(value).all() for value in strategy['candidates'][0]['model_state'].values()))


if __name__ == '__main__':
    unittest.main()
