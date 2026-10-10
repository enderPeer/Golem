"""Learned meaning conditioning, shared fact preprocessing and artifact safety."""
from pathlib import Path
import tempfile
import unittest

import torch

from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint, PAD
from english_world.semantic_model import (ACTS, SemanticGRU, encode_history, generate_semantic,
    load_semantic_checkpoint, save_semantic_checkpoint, warm_start_semantic)


class SemanticModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.config = NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=0.)
        self.model = SemanticGRU(self.config)

    def test_default_parameter_count_and_18_meanings(self):
        self.assertEqual(SemanticGRU().parameter_count, 800535)
        self.assertEqual(len(ACTS), 18)

    def test_fact_cue_ignores_assistant_claims_and_sits_next_to_current_user(self):
        turns = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Your name is Noah.'},
                 {'role': 'user', 'content': 'What is my name?'}]
        tokens = encode_history(turns)
        text = bytes(token for token in tokens if token < 256).decode()
        self.assertIn('Your name is Noah.', text)
        self.assertIn('[verified user name: UNKNOWN]', text)
        self.assertLess(text.index('Your name is Noah.'), text.index('[verified user name: UNKNOWN]'))
        self.assertLess(text.index('[verified user name: UNKNOWN]'), text.index('What is my name?'))

    def test_runtime_trim_preserves_full_history_user_facts_but_training_is_strict(self):
        turns = [{'role': 'user', 'content': 'My name is Maya'},
                 {'role': 'assistant', 'content': 'old words ' * 50},
                 {'role': 'user', 'content': 'What is my name?'}]
        with self.assertRaises(ValueError):
            encode_history(turns, max_context_tokens=96)
        tokens = encode_history(turns, max_context_tokens=96, trim_history=True)
        self.assertLessEqual(len(tokens), 96)
        self.assertIn(b'Maya', bytes(token for token in tokens if token < 256))
        with self.assertRaises(ValueError):
            encode_history([{'role': 'user', 'content': 'x' * 1000}], max_context_tokens=96, trim_history=True)

    def test_meaning_head_never_sees_teacher_forced_reply(self):
        self.model.eval()
        prefix = torch.tensor([encode_history([{'role': 'user', 'content': 'hello'}])])
        lengths = torch.tensor([prefix.shape[1]])
        _, first = self.model(prefix, lengths, torch.tensor([[10, 20, 30]]))
        _, second = self.model(prefix, lengths, torch.tensor([[40, 50, 60]]))
        self.assertTrue(torch.equal(first, second))

    def test_predicted_meaning_changes_decoder_and_receives_gradients(self):
        prefix = torch.tensor([encode_history([{'role': 'user', 'content': 'hello'}])])
        lengths = torch.tensor([prefix.shape[1]])
        logits, acts = self.model(prefix, lengths, torch.tensor([[65, 66]]))
        (logits.square().mean() + acts.square().mean()).backward()
        self.assertGreater(float(self.model.meaning_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.meaning_embedding.weight.grad.abs().sum()), 0)
        self.model.eval()
        with torch.no_grad():
            self.model.meaning_head.weight.zero_()
            self.model.meaning_head.bias.fill_(-20)
            self.model.meaning_head.bias[0] = 20
            a = self.model(prefix, lengths, torch.empty((1, 0), dtype=torch.long))[0]
            self.model.meaning_head.bias[0] = -20
            self.model.meaning_head.bias[1] = 20
            b = self.model(prefix, lengths, torch.empty((1, 0), dtype=torch.long))[0]
        self.assertFalse(torch.equal(a, b))

    def test_packed_batch_preserves_different_actual_history_lengths(self):
        self.model.eval()
        rows = [encode_history([{'role': 'user', 'content': text}]) for text in ('hi', 'a much longer question')]
        padded = torch.full((2, max(map(len, rows))), PAD, dtype=torch.long)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = torch.tensor(row)
        batch_hidden, batch_acts = self.model.encode(padded, torch.tensor(list(map(len, rows))))
        for i, row in enumerate(rows):
            hidden, acts = self.model.encode(torch.tensor([row]), torch.tensor([len(row)]))
            self.assertTrue(torch.allclose(hidden[:, 0], batch_hidden[:, i], atol=1e-6))
            self.assertTrue(torch.allclose(acts[0], batch_acts[i], atol=1e-6))

    def test_warm_start_and_generation_are_compatible_without_mutating_old_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            core = ByteGRULM(self.config)
            save_neural_checkpoint(path, core)
            original = path.read_bytes()
            semantic, _ = warm_start_semantic(path, self.config)
            for name, value in core.state_dict().items():
                self.assertTrue(torch.equal(value, semantic.core.state_dict()[name]))
            target = Path(directory) / 'semantic.pt'
            save_semantic_checkpoint(target, semantic, {'source': 'test'})
            rng = torch.get_rng_state().clone()
            restored, metadata = load_semantic_checkpoint(target)
            output = generate_semantic(restored, [{'role': 'user', 'content': 'hello'}], max_new_tokens=8)
            self.assertEqual(metadata['source'], 'test')
            self.assertIn(output['meaning_id'], ACTS)
            self.assertAlmostEqual(sum(output['meaning_probabilities'].values()), 1., places=6)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(path.read_bytes(), original)

    def test_nonfinite_greedy_generation_and_checkpoints_are_rejected(self):
        turns = [{'role': 'user', 'content': 'hello'}]
        with torch.no_grad():
            self.model.meaning_head.bias[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'meaning'):
            generate_semantic(self.model, turns, max_new_tokens=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                save_semantic_checkpoint(Path(directory) / 'bad.pt', self.model)
        with torch.no_grad():
            self.model.meaning_head.bias[0] = 0.
            self.model.core.output.bias[0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'generation logits'):
            generate_semantic(self.model, turns, max_new_tokens=1)

    def test_oversize_architecture_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'ten million'):
            SemanticGRU(NeuralConfig(embedding_dim=4096, hidden_size=4096, num_layers=16))


if __name__ == '__main__':
    unittest.main()
