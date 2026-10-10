"""Boundary tests for bounded CLI work and isolated deployment paths."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from english_world.cli import parser


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("worker_guard", ROOT / "ops" / "worker_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class DeploymentSafetyTests(unittest.TestCase):
    def test_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for value in ("..", "../escape", str(root.parent / "outside"), str(root)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    GUARD.contained(root, value)
            self.assertEqual(GUARD.contained(root, "runs/pilot"), root / "runs" / "pilot")

    def test_symlink_cannot_redirect_output_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, external = base / "project", base / "external"
            root.mkdir()
            external.mkdir()
            try:
                (root / "linked").symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlinks unavailable: {exc}")
            with self.assertRaises(ValueError):
                GUARD.contained(root, "linked/run")

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "config.toml").write_text("", encoding="utf-8")
            (root / "data.json").write_text("{}", encoding="utf-8")
            run = root / "run"
            run.mkdir()
            checkpoint = run / "checkpoint.json"
            checkpoint.write_text("preserve me", encoding="utf-8")
            with self.assertRaises(ValueError):
                GUARD.validate(root, "config.toml", "data.json", "run")
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), "preserve me")

    def test_disk_thresholds_stop_before_filling_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = ((50 * GUARD.GIB, 0), (100 * GUARD.GIB, 20 * GUARD.GIB))
            for free, used in cases:
                with self.subTest(free=free, used=used):
                    with patch.object(GUARD.shutil, "disk_usage", return_value=SimpleNamespace(free=free)), \
                         patch.object(GUARD, "project_bytes", return_value=used):
                        with self.assertRaises(RuntimeError):
                            GUARD.budget_check(root)
            with patch.object(GUARD.shutil, "disk_usage", return_value=SimpleNamespace(free=51 * GUARD.GIB)), \
                 patch.object(GUARD, "project_bytes", return_value=19 * GUARD.GIB):
                self.assertEqual(GUARD.budget_check(root), (51 * GUARD.GIB, 19 * GUARD.GIB))


class CliSafetyTests(unittest.TestCase):
    def assert_bad_argument(self, command):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser().parse_args(command)
        self.assertEqual(error.exception.code, 2)

    def test_nonfinite_or_nonpositive_budgets_are_rejected(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assert_bad_argument(["evolve", "--max-seconds", value])
                self.assert_bad_argument(["chat", "--temperature", value])

    def test_work_counts_and_chat_history_are_bounded(self):
        cases = (
            ["benchmark", "--generations", "0"],
            ["evolve", "--generations", "1000001"],
            ["evaluate", "--limit", "-1"],
            ["prepare", "--max-examples", "1000001"],
            ["prepare", "--context-length", "4097"],
            ["chat", "--context-chars", "0"],
            ["chat", "--context-chars", "65537"],
            ["chat", "--max-chars", "2049"],
        )
        for command in cases:
            with self.subTest(command=command):
                self.assert_bad_argument(command)


if __name__ == "__main__":
    unittest.main()
