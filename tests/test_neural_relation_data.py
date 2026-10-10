"""Relation labels must always be scored with their full identifying question."""
import random
import unittest

import torch

from english_world.neural_data import prepare_documents, sample_batch
from english_world.neural_model import USER, END, ASSISTANT, encode_bytes, ByteGRULM, NeuralConfig
from english_world.relation_eval import evaluate_relations


class NeuralRelationDataTests(unittest.TestCase):
    def setUp(self):
        self.question = 'Give the hypernym for dog [sense n:001; noun]. ' + 'Source hint. ' * 9
        self.answer = 'A mammal [sense n:002; noun].'
        self.text = 'user: ' + self.question + '\nassistant: ' + self.answer

    def test_relation_repetitions_are_tagged_but_rehearsal_dialogue_is_not(self):
        data = {'texts': {'train': [self.text, self.text, 'user: Hi\nassistant: Hello.']},
                'relation_texts': {'train': [self.text]},
                'facts': {'train': [{'text': self.text, 'id': 'fact1', 'relation': 'hypernym'}]}}
        docs = prepare_documents(data)
        self.assertEqual([doc.kind for doc in docs], ['relation', 'relation', 'dialogue'])
        self.assertEqual(docs[0].metadata['fact_id'], 'fact1')

    def test_facts_alone_are_sufficient_to_identify_relation_text(self):
        data = {'texts': {'train': [self.text]}, 'facts': {'train': [{'text': self.text, 'id': 'fact1'}]}}
        self.assertEqual(prepare_documents(data)[0].kind, 'relation')

    def test_short_windows_cannot_turn_relation_learning_into_answer_only_modeling(self):
        docs = prepare_documents({'texts': {'train': [self.text]}, 'relation_texts': {'train': [self.text]}})
        with self.assertRaisesRegex(ValueError, 'entire relation cue and answer; need at least'):
            sample_batch(docs, 2, 32, random.Random(0))
        batch = sample_batch(docs, 2, 384, random.Random(0))
        document = docs[0]
        answer_start = document.tokens.index(ASSISTANT) + 1
        self.assertEqual(batch['inputs'][0, 0].item(), USER)
        self.assertEqual(batch['targets'][0, answer_start - 1].item(), ord('A'))
        self.assertEqual(batch['inputs'][0, answer_start - 1].item(), ASSISTANT)
        self.assertFalse(batch['loss_mask'][0, :answer_start - 1].any())
        self.assertTrue(batch['loss_mask'][0, answer_start - 1:len(document.tokens) - 1].all())
        self.assertEqual(int(batch['loss_mask'][0].sum()), len(encode_bytes(self.answer)) + 1)
        self.assertEqual(batch['targets'][0, len(document.tokens) - 2].item(), END)

    def test_legacy_nonrelation_document_behavior_is_unchanged(self):
        doc = prepare_documents({'texts': {'train': [self.text]}})[0]
        self.assertEqual(doc.kind, 'dialogue')
        batch = sample_batch([doc], 1, 32, random.Random(0))
        self.assertEqual(tuple(batch['inputs'].shape), (1, 32))

    def test_default_relation_evaluation_scores_all_160_answer_bytes(self):
        torch.set_num_threads(1)
        model = ByteGRULM(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=0.))
        answer = 'a' * 160
        fact = {'id': 'f', 'relation': 'definition', 'source': 'n:1', 'target': 'n:1',
                'question': 'Define the concept.', 'answer': answer}
        report = evaluate_relations(model, {'facts': {'test': [fact]}})
        self.assertEqual(report['answer_bytes'], 160)
        self.assertEqual(report['truncated_answers'], 0)
        limited = evaluate_relations(model, {'facts': {'test': [fact]}}, max_answer_bytes=128)
        self.assertEqual(limited['answer_bytes'], 128)
        self.assertEqual(limited['truncated_answers'], 1)


if __name__ == '__main__':
    unittest.main()
