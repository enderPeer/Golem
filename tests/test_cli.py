import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from english_world.cli import main, parser


class CliTests(unittest.TestCase):
    def test_doctor_without_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["doctor"]), 0)
        self.assertIn('"requires_gpu": false', output.getvalue())

    def test_missing_checkpoint_is_actionable(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(main(["chat", "--checkpoint", "nonexistent-champion.json", "--prompt", "hi"]), 2)
        self.assertIn("golem:", output.getvalue())

    def test_command_defaults(self):
        args = parser().parse_args(["evolve"])
        self.assertEqual(args.run_dir, "runs/pilot")
        self.assertFalse(args.resume)

    def test_chat_uses_actual_prompt_suffix(self):
        from english_world.model import Genome
        genome = Genome(max_order=5, smoothing=1, rules={"hello": {"a": 255}, "world": {"b": 255}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.json"
            path.write_text(json.dumps({"genome": genome.to_dict()}), encoding="utf-8")
            outputs = []
            for prompt in ("hello", "world"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["chat", "--checkpoint", str(path), "--prompt", prompt,
                                 "--max-chars", "1", "--temperature", "0.05"])
                self.assertEqual(code, 0)
                outputs.append(output.getvalue().strip())
            self.assertEqual(outputs, ["a", "b"])

    def test_corrupt_champion_hash_is_rejected(self):
        from english_world.model import Genome
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.json"
            path.write_text(json.dumps({"genome": Genome().to_dict(), "genome_hash": "bad"}), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["chat", "--checkpoint", str(path), "--prompt", "hello"]), 2)

    def test_code_prediction_and_self_check_are_text_only(self):
        from english_world.model import Genome
        genome = Genome(domain="code", max_order=8, rules={"def ": {"F": 255}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.json"
            path.write_text(json.dumps({"genome": genome.to_dict(), "genome_hash": genome.digest()}), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["predict", "--checkpoint", str(path), "--text", "def "]), 0)
            self.assertEqual(json.loads(output.getvalue())["next_characters"][0]["character"], "F")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["self-check", "--checkpoint", str(path)]), 0)
            self.assertFalse(json.loads(output.getvalue())["self_prediction"]["used_for_fitness"])


if __name__ == "__main__":
    unittest.main()
