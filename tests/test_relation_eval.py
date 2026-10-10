"""Held-out-only relation scoring and honest distractor construction."""
import unittest
import json
from pathlib import Path
import tempfile

import torch

from english_world.neural_model import ByteGRULM, NeuralConfig
from english_world.relation_eval import evaluate_relations


def fact(identifier, source, target, answer, relation='hypernym'):
    return {'id': identifier, 'source': source, 'target': target, 'relation': relation,
            'question': f'What is a kind of {source}?', 'answer': answer}


class RelationEvaluationTests(unittest.TestCase):
    def test_cli_rejects_checkpoint_replacement_during_load(self):
        import contextlib
        import io
        from unittest.mock import patch
        from english_world.cli import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'live.pt'
            path.write_bytes(b'first checkpoint')
            def replace_checkpoint(_path):
                path.write_bytes(b'new checkpoint')
                return self.model, {}
            errors = io.StringIO()
            with patch('english_world.neural_model.load_neural_checkpoint', side_effect=replace_checkpoint), \
                    contextlib.redirect_stderr(errors):
                result = main(['neural', 'relation-eval', '--checkpoint', str(path)])
            self.assertEqual(result, 2)
            self.assertIn('Checkpoint changed while loading', errors.getvalue())

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.model = ByteGRULM(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=.1))

    def test_only_requested_partition_is_scored_without_model_or_rng_changes(self):
        data = {'facts': {'train': object(), 'validation': object(), 'test': [
            fact('test-a', 'dog-sense', 'animal-sense', 'An animal.'),
            fact('test-b', 'oak-sense', 'tree-sense', 'A tree.') ]}}
        parameters = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        rng = torch.get_rng_state().clone()
        self.model.train()
        report = evaluate_relations(self.model, data, max_cases=2)
        self.assertEqual(report['cases'], 2)
        self.assertEqual(report['contrast_cases'], 2)
        self.assertFalse(report['retrieval_used_by_model'])
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(report['held_out_answer_loss'], 0)
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]))

    def test_known_positive_target_cannot_be_a_distractor(self):
        data = {'facts': {'test': [fact('a', 'dog', 'animal', 'An animal.'),
            fact('b', 'dog', 'mammal', 'A mammal.'),
            fact('c', 'cat', 'mammal', 'Some mammal.'),
            fact('d', 'oak', 'tree', 'A tree.')]}}
        report = evaluate_relations(self.model, data)
        row = next(item for item in report['rows'] if item['id'] == 'a')
        self.assertEqual(row['distractor_id'], 'd')

    def test_contrast_never_crosses_relation_types(self):
        data = {'facts': {'test': [fact('a', 'dog', 'animal', 'Animal.'),
            fact('b', 'canine', 'canine', 'Dog.', relation='synonym')]}}
        report = evaluate_relations(self.model, data)
        self.assertEqual(report['contrast_cases'], 0)
        self.assertIsNone(report['fraction_preferring_correct'])

    def test_train_split_and_empty_facts_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_relations(self.model, {'facts': {}}, split='train')
        with self.assertRaises(ValueError):
            evaluate_relations(self.model, {'facts': {'test': []}})

    def test_full_graph_excludes_positive_missing_from_sampled_facts(self):
        facts = [fact('a', 'dog', 'animal', 'An animal.'),
                 fact('b', 'cat', 'mammal', 'A mammal.'),
                 fact('c', 'oak', 'tree', 'A tree.')]
        facts[0]['answer_visible_in_prompt'] = True
        graph = {'nodes': {'mammal': {'lemmas': ['mammal']}},
                 'edges': [{'source': 'dog', 'target': 'mammal', 'relation': 'hypernym'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            path.write_text(json.dumps(graph), encoding='utf-8')
            report = evaluate_relations(self.model, {'facts': {'test': facts}}, graph_path=path)
        row = next(row for row in report['rows'] if row['id'] == 'a')
        self.assertEqual(row['distractor_id'], 'c')
        self.assertTrue(report['full_graph_used_only_to_exclude_valid_distractors'])
        self.assertFalse(report['retrieval_used_by_model'])
        self.assertEqual(report['prompt_visibility']['visible_in_prompt']['cases'], 1)
        self.assertEqual(report['prompt_visibility']['not_visible_in_prompt']['cases'], 2)

    def test_same_answer_question_contrast_detects_prompt_dependence_and_filters_leaks(self):
        class PromptModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, inputs):
                logits = torch.zeros((*inputs.shape, 260), device=inputs.device)
                # This test-only causal rule identifies the uppercase cue in the
                # question. Answer tokens X/Y/Z cannot supply that cue themselves.
                for row in range(inputs.shape[0]):
                    seen_a = (inputs[row] == ord('A')).cumsum(0) > 0
                    logits[row, :, ord('X')] = seen_a.float() * 6
                    logits[row, :, ord('Y')] = (~seen_a).float() * 6
                return logits, None

        records = [fact('a', 'sense-a', 'answer-x', 'X'), fact('b', 'sense-b', 'answer-y', 'Y'),
                   fact('c', 'sense-c', 'answer-z', 'Z')]
        records[0]['question'] = 'Query A?'
        records[1]['question'] = 'Query B includes X?'
        records[2]['question'] = 'Query C?'
        records[1]['answer_visible_in_prompt'] = True
        data = {'facts': {'test': records}}
        first = evaluate_relations(PromptModel(), data)
        second = evaluate_relations(PromptModel(), data)
        self.assertEqual(first, second)
        row = next(row for row in first['rows'] if row['id'] == 'a')
        self.assertEqual(row['shuffled_question_id'], 'c')
        self.assertEqual(row['shuffled_questions_excluded_answer_visible'], 1)
        self.assertEqual(row['same_answer_bytes'], row['answer_bytes'])
        self.assertGreater(row['shuffled_question_minus_correct_loss'], 5.9)
        self.assertEqual(first['prompt_visibility']['not_visible_in_prompt']['same_answer_question_pairs'], 2)
        self.assertIn('mean_shuffled_question_minus_correct_loss', first)

    def test_all_original_cases_remain_when_no_clean_shuffled_question_exists(self):
        records = [fact('a', 'source-a', 'target-a', 'alpha'), fact('b', 'source-b', 'target-b', 'beta')]
        records[0]['question'] = 'Question containing beta?'
        records[1]['question'] = 'Question containing alpha?'
        report = evaluate_relations(self.model, {'facts': {'test': records}})
        self.assertEqual(report['cases'], 2)
        self.assertEqual(report['same_answer_question_pairs'], 0)
        self.assertEqual(report['shuffled_question_exclusions']['cases_without_eligible_question'], 2)
        self.assertIsNone(report['mean_shuffled_question_minus_correct_loss'])
        self.assertGreater(report['held_out_answer_loss'], 0)


if __name__ == '__main__':
    unittest.main()
