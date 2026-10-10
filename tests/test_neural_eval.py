import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "optional neural runtime not installed")
class NeuralDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(2)

    def test_packed_export_matches_quantized_tensors(self):
        from english_world.neural_eval import ternary_export
        from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint
        model = ByteGRULM(NeuralConfig(embedding_dim=5, hidden_size=7, num_layers=1, dropout=0.))
        data = {"texts": {"validation": ["user: hi\nassistant: hello there"]}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_neural_checkpoint(checkpoint, model)
            report = ternary_export(checkpoint, data, Path(directory) / "packed", batches=1)
            self.assertTrue(report["exact_packed_roundtrip"])
            self.assertFalse(report["packed_execution_implemented"])
            self.assertEqual((Path(directory) / "packed/matrices.bin").stat().st_size,
                             report["packed_matrix_payload_bytes"])
            self.assertGreater(report["residual_parameter_payload_bytes"], 0)

    def test_relevance_scores_same_answers_with_altered_prompts(self):
        from english_world.neural_eval import response_relevance
        from english_world.neural_model import ByteGRULM, NeuralConfig
        model = ByteGRULM(NeuralConfig(embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.))
        data = {"conversations": {"test": [
            {"turns": [{"role": "user", "content": "red"}, {"role": "assistant", "content": "color"}]},
            {"turns": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "number"}]}]}}
        report = response_relevance(model, data)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["answer_bytes"], 11)
        self.assertAlmostEqual(report["shuffled_minus_actual_loss"],
                               report["shuffled_prompt_loss"] - report["actual_prompt_loss"])

    def test_interactive_chat_keeps_special_role_tokens(self):
        from english_world.cli import main
        from english_world.neural_model import ByteGRULM, NeuralConfig, USER, ASSISTANT
        model = ByteGRULM(NeuralConfig(embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.))
        with mock.patch("english_world.neural_model.load_neural_checkpoint", return_value=(model, {})), \
             mock.patch("english_world.neural_model.generate_neural", return_value="OK") as generate, \
             mock.patch("builtins.input", side_effect=["hello", "second", "/quit"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["neural", "chat", "--checkpoint", "unused.pt"]), 0)
        first, second = [call.args[1] for call in generate.call_args_list]
        self.assertIsInstance(first, list)
        self.assertEqual(first.count(USER), 1)
        self.assertEqual(second.count(USER), 2)
        self.assertEqual(second.count(ASSISTANT), 2)


if __name__ == "__main__":
    unittest.main()
