import random
from pathlib import Path
import tempfile
import unittest

import torch

from english_world.neural_model import (
    ASSISTANT, END, PAD, USER, ByteGRULM, NeuralConfig, clone_ternary,
    decode_bytes, encode_bytes, generate_neural, load_neural_checkpoint,
    save_neural_checkpoint,
)
from english_world.neural_data import (
    dialogue_document, document_from_text, make_recall_documents,
    prepare_documents, sample_batch,
)


class NeuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tiny(self):
        torch.manual_seed(7)
        return ByteGRULM(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=2, dropout=0.))

    def test_default_and_small_parameter_counts(self):
        self.assertEqual(ByteGRULM().parameter_count, 791300)
        self.assertEqual(ByteGRULM(NeuralConfig(embedding_dim=32, hidden_size=64, num_layers=1)).parameter_count, 44036)

    def test_utf8_and_roles_are_distinct(self):
        text = 'Hello, café!\n\t  x = 2'
        self.assertEqual(decode_bytes(encode_bytes(text)), text)
        self.assertEqual(decode_bytes([USER] + encode_bytes(text) + [END, ASSISTANT, PAD]), text)
        self.assertNotEqual(encode_bytes('USER'), [USER])

    def test_recurrent_state_carries_information_beyond_five_bytes(self):
        model = self.tiny().eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [99, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            logits, state = model(inputs)
        self.assertFalse(torch.equal(state[:, 0], state[:, 1]))
        self.assertFalse(torch.equal(logits[0, -1], logits[1, -1]))
        # Architectural capacity only; this is not a learned-memory accuracy claim.

    def test_chunked_state_matches_full_sequence_and_no_future_leakage(self):
        model = self.tiny().eval()
        inputs = torch.tensor([[3, 6, 9, 12, 15, 18, 21]])
        altered = inputs.clone()
        altered[:, 4:] = 77
        with torch.no_grad():
            full, final_state = model(inputs)
            first, state = model(inputs[:, :3])
            second, state = model(inputs[:, 3:], state)
            changed, _ = model(altered)
        self.assertTrue(torch.allclose(torch.cat([first, second], dim=1), full, atol=1e-6))
        self.assertTrue(torch.allclose(state, final_state, atol=1e-6))
        self.assertTrue(torch.equal(full[:, :4], changed[:, :4]))

    def test_checkpoint_roundtrip_safe_and_preserves_rng(self):
        model = self.tiny().eval()
        inputs = torch.tensor([[USER, 65, 66, END, ASSISTANT]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'champion.pt'
            save_neural_checkpoint(path, model, {'generation': 4, 'experiment': 'test'})
            raw = torch.load(path, weights_only=True)
            self.assertIn('model_state', raw)
            self.assertNotIn('model', raw)
            before_rng = torch.get_rng_state().clone()
            restored, metadata = load_neural_checkpoint(path)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(metadata, {'generation': 4, 'experiment': 'test'})
            self.assertEqual(restored.config, model.config)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(inputs)[0], restored(inputs)[0]))
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_generation_is_deterministic_and_stops_at_turn_boundary(self):
        model = self.tiny()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-100.)
            model.output.bias[END] = 100.
        rng_before = torch.get_rng_state().clone()
        self.assertEqual(generate_neural(model, 'hello', max_new_tokens=10), '')
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        with torch.no_grad():
            model.output.bias[END] = -100.
            model.output.bias[ord('A')] = 100.
            model.output.bias[PAD] = 1000.  # Padding is never an output byte.
        self.assertEqual(generate_neural(model, 'hello', max_new_tokens=4), 'AAAA')

    def test_single_layer_dropout_is_tunable_without_changing_shapes(self):
        model = ByteGRULM(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=0.))
        shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        model.set_dropout(.35)
        self.assertEqual(model.input_dropout.p, .35)
        self.assertEqual(model.gru.dropout, 0.)
        self.assertEqual(model.config.dropout, .35)
        self.assertEqual(shapes, {key: tuple(value.shape) for key, value in model.state_dict().items()})

    def test_ternary_clone_leaves_parent_and_embeddings_untouched(self):
        model = self.tiny().eval()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        clone, report = clone_ternary(model)
        self.assertFalse(report['native_bitnet'])
        self.assertFalse(report['packed_storage'])
        self.assertGreater(report['quantized_parameters'], 0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))
            if key == 'embedding.weight' or 'bias' in key:
                self.assertTrue(torch.equal(clone.state_dict()[key], original[key]))
        for matrix in report['matrices']:
            values = clone.state_dict()[matrix['name']]
            if matrix['scale']:
                normalized = values / matrix['scale']
                self.assertTrue(torch.all((normalized == -1) | (normalized == 0) | (normalized == 1)))


class NeuralDataTests(unittest.TestCase):
    def test_plain_text_has_contiguous_targets_and_padding_mask(self):
        document = document_from_text('abc')
        batch = sample_batch([document], 1, 8, random.Random(1))
        self.assertEqual(batch['inputs'][0, :3].tolist(), encode_bytes('abc'))
        self.assertEqual(batch['targets'][0, :3].tolist(), [ord('b'), ord('c'), END])
        self.assertEqual(batch['loss_mask'][0].tolist(), [True, True, True, False, False, False, False, False])
        self.assertTrue(torch.all(batch['targets'][~batch['loss_mask']] == PAD))

    def test_dialogue_loss_is_assistant_only_and_target_aligned(self):
        document = document_from_text('user: abc\nassistant: XY\nuser: Z\nassistant: Q')
        self.assertEqual(document.kind, 'dialogue')
        batch = sample_batch([document], 1, 64, random.Random(2))
        supervised = batch['targets'][batch['loss_mask']].tolist()
        self.assertEqual(supervised, encode_bytes('XY') + [END] + encode_bytes('Q') + [END])
        first_answer = document.tokens.index(ord('X'))
        self.assertEqual(batch['inputs'][0, first_answer - 1].item(), ASSISTANT)
        self.assertEqual(batch['targets'][0, first_answer - 1].item(), ord('X'))
        self.assertFalse(document.loss_mask[document.tokens.index(USER)])

    def test_contiguous_split_source_required(self):
        with self.assertRaises(ValueError):
            prepare_documents({'train': [{'context': 'a', 'target': 'b'}]})
        data = {'texts': {'train': ['abc'], 'validation': ['xyz'], 'test': ['def']}}
        self.assertEqual(prepare_documents(data, 'validation')[0].tokens[:-1], tuple(encode_bytes('xyz')))

    def test_sampling_is_deterministic_and_cannot_cross_documents(self):
        docs = [document_from_text('a' * 50), document_from_text('b' * 50)]
        a = sample_batch(docs, 8, 12, random.Random(123))
        b = sample_batch(docs, 8, 12, random.Random(123))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        for row in a['inputs']:
            letters = set(row.tolist()) - {PAD, END}
            self.assertTrue(letters <= {ord('a')} or letters <= {ord('b')})

    def test_recall_uses_only_answer_and_preserves_distant_cue(self):
        documents = make_recall_documents(10, delay=64, seed=10)
        self.assertEqual({doc.tokens[5] for doc in documents}, set(encode_bytes('0123456789')))
        self.assertEqual(len({doc.tokens[6:doc.metadata['answer_index']] for doc in documents}), 1)
        for document in documents:
            answer = document.metadata['answer_index']
            self.assertEqual(sum(document.loss_mask), 1)
            self.assertTrue(document.loss_mask[answer])
            self.assertEqual(document.tokens[answer], document.tokens[5])
            self.assertEqual(document.tokens[answer - 1], ASSISTANT)
            self.assertGreater(answer - 5, 64)
        batch = sample_batch(documents, 10, 128, random.Random(4))
        self.assertEqual(int(batch['loss_mask'].sum()), 10)
        self.assertTrue(torch.all(batch['inputs'][batch['loss_mask']] == ASSISTANT))
        self.assertTrue(torch.all(batch['targets'][batch['loss_mask']] >= ord('0')))
        self.assertTrue(torch.all(batch['targets'][batch['loss_mask']] <= ord('9')))
        with self.assertRaises(ValueError):
            sample_batch(documents, 1, 32, random.Random(1))


if __name__ == '__main__':
    unittest.main()
