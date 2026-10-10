"""Forward-value, gradient and selection checks for Golem ternary QAT."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint, load_neural_checkpoint
from english_world import neural_quantization as qat


class NeuralQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = ByteGRULM(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=0.))
        self.inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    def test_exact_ternary_values_with_identity_straight_through_gradient(self):
        value = torch.tensor([[-.82, -.1, .05, .73]], requires_grad=True)
        result = qat.ternary_value(value)
        scale = value.detach().abs().mean()
        self.assertTrue(set(result.detach().reshape(-1).tolist()) <= {-float(scale), 0., float(scale)})
        result.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_actual_forward_matches_materialized_matrices_and_restores_masters(self):
        self.model.eval()
        before = copy.deepcopy(self.model.state_dict())
        original_logits, _ = self.model(self.inputs)
        quantized_logits, _ = qat.forward_ternary(self.model, self.inputs)
        reference, metadata = qat.materialize_ternary(self.model)
        expected, _ = reference(self.inputs)
        self.assertTrue(torch.allclose(quantized_logits, expected, atol=1e-7, rtol=1e-6))
        self.assertFalse(torch.allclose(original_logits, quantized_logits))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        restored_logits, _ = self.model(self.inputs)
        self.assertTrue(torch.equal(restored_logits, original_logits))
        self.assertTrue(metadata['already_quantized'])
        self.assertNotIn('embedding.weight', {row['name'] for row in metadata['matrices']})

    def test_gradients_reach_master_gru_and_output_without_replacing_parameters(self):
        identities = {name: id(parameter) for name, parameter in self.model.named_parameters()}
        logits, _ = qat.forward_ternary(self.model, self.inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 260), torch.full((8,), 32))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertEqual(id(parameter), identities[name])
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if qat.is_ternary_matrix(name, parameter):
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_export_after_training_forward_does_not_copy_cached_autograd_graph(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.model.train()
        logits, _ = qat.forward_ternary(self.model, self.inputs, straight_through=True)
        logits.sum().backward()
        optimizer.step()
        rng_before = torch.get_rng_state().clone()
        reference, metadata = qat.materialize_ternary(self.model)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(metadata['already_quantized'])
        self.assertTrue(all(parameter.is_leaf for parameter in reference.parameters()))
        for name, value in qat.quantized_parameter_dict(self.model, straight_through=False).items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)

    def test_training_never_prepares_test_split_and_exports_selected_reference(self):
        data = {'texts': {'train': ['a simple training sequence. ' * 3],
                         'validation': ['a separate validation sequence.'],
                         'test': object()}}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.pt'
            save_neural_checkpoint(source, self.model)
            source_bytes = source.read_bytes()
            real_prepare = qat.prepare_documents
            with patch.object(qat, 'prepare_documents', wraps=real_prepare) as prepare:
                result = qat.train_ternary(source, data, Path(temporary) / 'run',
                    {'max_updates': 2, 'batch_size': 2, 'sequence_length': 12, 'validation_batches': 1,
                     'evaluate_every': 1, 'threads': 1, 'max_seconds': 20})
            self.assertEqual([call.args[1] for call in prepare.call_args_list], ['train', 'validation'])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertLessEqual(result['best_qat']['loss'], result['post_training_ternary']['loss'])
            state = torch.load(Path(temporary) / 'run' / 'qat-training.pt', weights_only=True)
            self.assertTrue(state['optimizer_state']['state'])
            self.assertEqual(state['updates'], 2)
            self.assertTrue(any(not torch.equal(value, self.model.state_dict()[name])
                                for name, value in state['model_state'].items()))
            selected, metadata = load_neural_checkpoint(Path(temporary) / 'run' / 'qat-reference.pt')
            for matrix in metadata['matrices']:
                values = selected.state_dict()[matrix['name']]
                self.assertTrue(set(values.reshape(-1).tolist()) <= {-matrix['scale'], 0., matrix['scale']})

    def test_initial_ptq_forward_uses_same_scale_as_export(self):
        self.model.eval()
        reference, _ = qat.materialize_ternary(self.model)
        wrapper = qat.TernaryForward(self.model).eval()
        for name, value in qat.quantized_parameter_dict(self.model, straight_through=False).items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
        with torch.no_grad():
            # Parameter substitution can choose a different CPU GRU kernel path;
            # the parameters are bit-exact above, outputs agree to FP32 tolerance.
            self.assertTrue(torch.allclose(wrapper(self.inputs)[0], reference(self.inputs)[0], atol=1e-7, rtol=1e-6))

    def test_time_limit_after_unevaluated_update_still_saves_masters_and_best(self):
        data = {'texts': {'train': ['train only. ' * 3], 'validation': ['held out. ' * 3]}}
        elapsed = [False]
        real_batch = qat.sample_batch

        def training_batch(*args, **kwargs):
            batch = real_batch(*args, **kwargs)
            elapsed[0] = True
            return batch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pt'
            save_neural_checkpoint(source, self.model)
            with patch.object(qat, 'sample_batch', side_effect=training_batch), \
                 patch.object(qat.time, 'monotonic', side_effect=lambda: 181. if elapsed[0] else 0.):
                report = qat.train_ternary(source, data, Path(directory) / 'run',
                    {'max_updates': 200, 'batch_size': 2, 'sequence_length': 12,
                     'validation_batches': 1, 'threads': 1})
            self.assertEqual(report['status'], 'time_limit')
            self.assertEqual(report['updates'], 1)
            self.assertEqual(report['best_update'], 0)
            checkpoint = torch.load(Path(directory) / 'run' / 'qat-training.pt', weights_only=True)
            self.assertEqual(checkpoint['updates'], 1)
            self.assertTrue(checkpoint['optimizer_state']['state'])
            _, metadata = load_neural_checkpoint(Path(directory) / 'run' / 'qat-reference.pt')
            self.assertEqual(metadata['best_update'], 0)

    def test_nonfinite_or_unsafe_budgets_rejected(self):
        for change in ({'learning_rate': float('nan')}, {'max_seconds': 181}, {'threads': 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                qat._configuration(change)


if __name__ == '__main__':
    unittest.main()
