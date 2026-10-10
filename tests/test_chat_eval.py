import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world.chat_eval import evaluate_chat, prompt_tokens, score_response, validation_selection_key
from english_world.neural_model import USER, ASSISTANT, END, VOCAB_SIZE, decode_bytes, encode_bytes


def case(identity='greeting-1', intent='greeting', question='Hello there', reference='Hello!', required='hello'):
    return {'id': identity, 'intent': intent, 'family_id': identity.split('-')[0],
            'turns': [{'role': 'user', 'content': question}], 'reference': reference,
            'rubric': {'required_any': [required], 'required_all': [], 'forbidden': [], 'max_response_bytes': 120},
            'context_pair_id': None, 'memory_value': None}


def memory_case(name):
    row = case('recall-' + name, 'name_recall', 'What is my name?', 'Your name is ' + name + '.', name)
    row['turns'] = [{'role': 'user', 'content': 'My name is ' + name + '.'},
                    {'role': 'assistant', 'content': 'I will remember that.'}, row['turns'][0]]
    row['context_pair_id'] = 'same-question-different-name'
    row['memory_value'] = name
    row['rubric'] = {'required_any': [], 'required_all': [name],
                     'forbidden': ['Nova' if name == 'Mara' else 'Mara'], 'max_response_bytes': 120}
    return row


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, inputs, hidden=None):
        return torch.zeros((*inputs.shape, VOCAB_SIZE), device=inputs.device) + self.anchor, None


class ChatEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_whole_phrase_boundaries_and_forbidden_statements(self):
        rubric = {'required_any': ['hi', 'hello'], 'required_all': [], 'forbidden': [], 'max_response_bytes': 120}
        self.assertEqual(score_response('this is a word', rubric)['score'], 0.)
        self.assertTrue(score_response('Hi!', rubric)['heuristic_pass'])
        identity = {'required_all': ['Golem'], 'forbidden': ['i am human']}
        self.assertEqual(score_response('I am Golem and I am human.', identity)['score'], 0.)
        self.assertTrue(score_response('I am Golem, an AI assistant.', identity)['heuristic_pass'])

    def test_empty_garbage_repetition_and_partial_requirements(self):
        rubric = {'required_all': ['Golem'], 'max_response_bytes': 120}
        for response in ('', 'Golem\ufffd', 'Golem Golem Golem', 'Golem ' + 'x' * 10, 'Golem\x00'):
            self.assertEqual(score_response(response, rubric)['score'], 0.)
        partial = score_response('Golem is here.', {'required_all': ['Golem', 'help']})
        self.assertEqual(partial['score'], .5)
        self.assertFalse(partial['heuristic_pass'])

    def test_multiturn_prompt_keeps_history_and_separate_role_tokens(self):
        row = memory_case('Mara')
        tokens = prompt_tokens(row['turns'])
        expected = []
        for turn in row['turns']:
            expected += [USER if turn['role'] == 'user' else ASSISTANT] + encode_bytes(turn['content']) + [END]
        self.assertEqual(tokens, expected + [ASSISTANT])
        self.assertIn('My name is Mara.', decode_bytes(tokens))
        self.assertNotIn(row['reference'], decode_bytes(tokens))
        with self.assertRaises(ValueError):
            prompt_tokens(row['turns'], max_prompt_bytes=5)

    def test_actual_generation_saved_no_canned_fallback_or_reference_prompt(self):
        row = case(reference='SECRET_REFERENCE_NOT_IN_PROMPT', required='hello')
        model = UniformModel()
        original_rng = torch.get_rng_state().clone()
        captured = []

        def fake_generate(model, tokens, **kwargs):
            captured.append((tokens, kwargs))
            return ''

        with tempfile.TemporaryDirectory() as tmp, patch('english_world.chat_eval.generate_neural', side_effect=fake_generate):
            path = Path(tmp) / 'report.json'
            report = evaluate_chat(model, {'chat_cases': {'validation': [row]}}, output=path)
            saved = json.loads(path.read_text())
        self.assertEqual(report['rows'][0]['generated'], '')
        self.assertEqual(saved['rows'][0]['generated'], '')
        self.assertEqual(report['score'], 0.)
        self.assertNotIn(row['reference'], decode_bytes(captured[0][0]))
        self.assertLessEqual(captured[0][1]['max_new_tokens'], 120)
        self.assertAlmostEqual(report['reference_loss'], math.log(VOCAB_SIZE), places=5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))

    def test_context_pairs_are_atomic_use_same_rng_and_require_both_answers(self):
        cases = [case(), memory_case('Mara'), memory_case('Nova')]
        seeds = []

        def contextual_generate(model, tokens, **kwargs):
            text = decode_bytes(tokens)
            if 'My name is' in text:
                seeds.append(kwargs['seed'])
                return 'Your name is Mara.' if 'My name is Mara.' in text else 'Your name is Nova.'
            return 'Hello!'

        with patch('english_world.chat_eval.generate_neural', side_effect=contextual_generate):
            report = evaluate_chat(UniformModel(), {'chat_cases': {'validation': cases}}, max_cases=3,
                                   include_reference_nll=False, temperature=.8)
        self.assertEqual(report['score'], 1.)
        self.assertEqual(report['context_pair_pass_rate'], 1.)
        self.assertEqual(seeds[0], seeds[1])
        with patch('english_world.chat_eval.generate_neural', return_value='Your name is Mara.'):
            wrong = evaluate_chat(UniformModel(), {'chat_cases': {'validation': cases}}, max_cases=3,
                                  include_reference_nll=False)
        self.assertEqual(wrong['context_pair_pass_rate'], 0.)
        with patch('english_world.chat_eval.generate_neural', return_value='Hello!'):
            small = evaluate_chat(UniformModel(), {'chat_cases': {'validation': cases}}, max_cases=2,
                                  include_reference_nll=False)
        self.assertEqual(small['cases'], 1)
        self.assertEqual(small['context_pairs'], [])

    def test_selection_is_validation_only_and_generation_bound_enforced(self):
        self.assertGreater(validation_selection_key({'split': 'validation', 'score': .7, 'reference_loss': 3.}),
                           validation_selection_key({'split': 'validation', 'score': .6, 'reference_loss': 1.}))
        self.assertGreater(validation_selection_key({'split': 'validation', 'score': .7, 'reference_loss': 2.}),
                           validation_selection_key({'split': 'validation', 'score': .7, 'reference_loss': 3.}))
        with self.assertRaises(ValueError):
            validation_selection_key({'split': 'test', 'score': 1., 'reference_loss': .1})
        with self.assertRaises(ValueError):
            evaluate_chat(UniformModel(), {'chat_cases': {'validation': [case()]}}, max_new_tokens=121)
        with self.assertRaises(ValueError):
            evaluate_chat(UniformModel(), {'chat_cases': {'train': [case()]}}, split='train')

    def test_cancellation_does_not_publish_partial_scores_and_restores_mode(self):
        model = UniformModel()
        stop = [False]
        calls = []

        def first_response_then_stop(model, tokens, **kwargs):
            calls.append(tokens)
            stop[0] = True
            return 'Hello!'

        data = {'chat_cases': {'validation': [case(), case('greeting-2', question='Hi again')]}}
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            path.write_text('{"previous_complete_report":true}', encoding='utf-8')
            with patch('english_world.chat_eval.generate_neural', side_effect=first_response_then_stop):
                with self.assertRaises(InterruptedError):
                    evaluate_chat(model, data, output=path, should_stop=lambda: stop[0],
                                  include_reference_nll=False)
            self.assertEqual(path.read_text(encoding='utf-8'), '{"previous_complete_report":true}')
        self.assertEqual(len(calls), 1)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_already_cancelled_evaluation_never_generates(self):
        with patch('english_world.chat_eval.generate_neural') as generate:
            with self.assertRaises(InterruptedError):
                evaluate_chat(UniformModel(), {'chat_cases': {'validation': [case()]}}, should_stop=lambda: True)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
