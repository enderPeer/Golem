"""Resuming a bounded worker never enables an unrelated or missing run."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resume_worker_guard", ROOT / "ops" / "worker_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class ResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "config.toml").write_text("", encoding="utf-8")
        (self.root / "data.json").write_text("{}", encoding="utf-8")
        self.run = self.root / "runs" / "language-hour"
        self.run.mkdir(parents=True)

    def checkpoint(self):
        path = self.run / "checkpoint.json"
        path.write_text('{"generation":622}', encoding="utf-8")
        return path

    def test_resume_accepts_checkpoint_without_modifying_it(self):
        checkpoint = self.checkpoint()
        original = checkpoint.read_bytes()
        result = GUARD.validate(self.root, "config.toml", "data.json", "runs/language-hour", resume=True)
        self.assertEqual(result[2], self.run)
        self.assertEqual(checkpoint.read_bytes(), original)
        with self.assertRaises(ValueError):
            GUARD.validate(self.root, "config.toml", "data.json", "runs/language-hour")

    def test_resume_rejects_missing_or_directory_checkpoint(self):
        for run in ("runs/missing", "runs/language-hour"):
            with self.subTest(run=run), self.assertRaises(ValueError):
                GUARD.validate(self.root, "config.toml", "data.json", run, resume=True)
        (self.run / "checkpoint.json").mkdir()
        with self.assertRaises(ValueError):
            GUARD.validate(self.root, "config.toml", "data.json", "runs/language-hour", resume=True)

    def test_resume_rejects_run_outside_project(self):
        self.checkpoint()
        with self.assertRaises(ValueError):
            GUARD.validate(self.root, "config.toml", "data.json", "../other-run", resume=True)

    def test_resume_rejects_symlink_checkpoint(self):
        target = self.root / "other-checkpoint.json"
        target.write_text("{}", encoding="utf-8")
        try:
            (self.run / "checkpoint.json").symlink_to(target)
        except OSError:
            self.skipTest("File symlinks are unavailable on this host")
        with self.assertRaises(ValueError):
            GUARD.validate(self.root, "config.toml", "data.json", "runs/language-hour", resume=True)

    def test_validate_cli_accepts_resume(self):
        self.checkpoint()
        arguments = ["worker_guard.py", "--validate", str(self.root), "config.toml",
                     "data.json", "runs/language-hour", "--resume"]
        with patch.object(GUARD.sys, "argv", arguments):
            self.assertEqual(GUARD.main(), 0)
        with patch.object(GUARD.sys, "argv", arguments[:-1]), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(GUARD.main(), 1)

    def test_run_cli_propagates_resume(self):
        self.checkpoint()
        arguments = ["worker_guard.py", "--run", str(self.root), "--config", "config.toml",
                     "--data", "data.json", "--run-dir", "runs/language-hour", "--resume"]
        with patch.object(GUARD.sys, "argv", arguments), \
             patch.object(GUARD, "run_worker", return_value=0) as worker:
            self.assertEqual(GUARD.main(), 0)
        self.assertTrue(worker.call_args.kwargs["resume"])

    def test_worker_passes_resume_only_when_requested(self):
        paths = self.root / "config.toml", self.root / "data.json", self.run
        for resume in (False, True):
            child = Mock(returncode=0)
            child.poll.return_value = 0
            with self.subTest(resume=resume), \
                 patch.object(GUARD.os, "name", "posix"), \
                 patch.object(GUARD, "validate", return_value=paths) as validation, \
                 patch.object(GUARD, "preflight"), \
                 patch.object(GUARD.signal, "signal"), \
                 patch.object(GUARD.subprocess, "Popen", return_value=child) as popen:
                self.assertEqual(GUARD.run_worker(self.root, "config.toml", "data.json",
                                                 "runs/language-hour", resume=resume), 0)
            command = popen.call_args.args[0]
            self.assertEqual("--resume" in command, resume)
            self.assertEqual(validation.call_args.kwargs["resume"], resume)


if __name__ == "__main__":
    unittest.main()
