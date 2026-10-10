"""Source snapshots are explicit, immutable and disjoint from held-out files."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from english_world.data import CODE_ALPHABET, SELF_CODE_FILES, load_dataset, normalize_code, prepare_dataset


class CodeDataTests(unittest.TestCase):
    def make_project(self, root):
        tracked = set(SELF_CODE_FILES)
        for index, relative in enumerate(SELF_CODE_FILES):
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            if "__" in source.name:
                text = "# short module stub\r\n"
            else:
                text = (f"def Module{index}(Parameter):\r\n"
                        f"    UniqueModule{index} = 'Marker{index}'\r\n"
                        "\tif Parameter != None:\r\n"
                        f"        return UniqueModule{index} + ' caf\u00e9'\r\n"
                        "    return 'Empty'\r\n")
            source.write_bytes(text.encode("utf-8"))
        # These must not be considered, even when Git claims an unrelated file
        # is tracked or the directory contains attractive source-looking data.
        for relative in ("private.py", "english_world/secrets.py", "data/corpus.py", "tests/test_private.py"):
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("SECRET_OUTSIDE_ALLOWLIST\n" * 12, encoding="utf-8")
            tracked.add(relative)
        return tracked

    def prepare(self, output, project, tracked, **options):
        with patch("english_world.data._git_source_metadata", return_value=(tracked, "a" * 40, True)):
            with patch("english_world.data.fetch_file", side_effect=AssertionError("code data must never download")):
                return prepare_dataset(output, source="self-code", project_root=project, max_examples=120, **options)

    def test_deterministic_file_disjoint_snapshot_and_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            tracked = self.make_project(project)
            first = self.prepare(root / "a", project, tracked, seed=7)
            second = self.prepare(root / "b", project, tracked, seed=7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            data = load_dataset(first)
            self.assertEqual(data["domain"], "code")
            self.assertEqual(data["alphabet"], CODE_ALPHABET)
            self.assertEqual(data["dictionary"], [])
            self.assertEqual(len(data["train"]), 120)
            records = data["manifest"]["source_files"]
            self.assertEqual(len(records), 5)
            self.assertEqual(data["manifest"]["source_commit"], "a" * 40)
            self.assertEqual(data["manifest"]["splits"]["validation"]["source_paths"], ["english_world/cli.py"])
            self.assertEqual(data["manifest"]["splits"]["test"]["source_paths"], ["ops/worker_guard.py"])
            all_text = json.dumps(data)
            self.assertNotIn("SECRET_OUTSIDE_ALLOWLIST", all_text)
            paths = {split: set(data["manifest"]["splits"][split]["source_paths"]) for split in ("train", "validation", "test")}
            self.assertFalse(paths["train"] & paths["validation"])
            self.assertFalse(paths["train"] & paths["test"])
            for record in records:
                raw = (root / "a" / record["snapshot_path"]).read_bytes()
                self.assertEqual(raw, (project / record["path"]).read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            for split in ("train", "validation", "test"):
                for example in data[split]:
                    self.assertTrue(any(example["context"] + example["target"] in doc for doc in data["texts"][split]))
            held_out_markers = ("UniqueModule5", "UniqueModule6")
            for example in data["train"]:
                self.assertFalse(any(marker in example["context"] for marker in held_out_markers))

    def test_preserves_case_indentation_and_escapes_unicode(self):
        text = "def CamelCase(X):\r\n    return X + '\u00e9'\r\n\t# Tab\x00\r"
        expected = "def CamelCase(X):\n    return X + '\\xe9'\n\t# Tab\\x00\n"
        self.assertEqual(normalize_code(text), expected)
        self.assertTrue(set(expected) <= set(CODE_ALPHABET))

    def test_existing_snapshot_is_not_updated_from_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            tracked = self.make_project(project)
            output = root / "out"
            path = self.prepare(output, project, tracked)
            original = path.read_bytes()
            (project / "english_world/model.py").write_text("# runtime changed completely\n", encoding="utf-8")
            with patch("english_world.data._git_source_metadata", side_effect=AssertionError("must reuse snapshot")):
                self.assertEqual(prepare_dataset(output, "self-code", project_root=project, max_examples=120), path)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "different settings"):
                prepare_dataset(output, "self-code", project_root=project, max_examples=121)

    def test_corrupted_snapshot_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            tracked = self.make_project(project)
            path = self.prepare(root / "out", project, tracked)
            snapshot = root / "out/source_snapshot/english_world/model.py"
            snapshot.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_dataset(path)

    def test_untracked_source_is_excluded_and_insufficient_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracked = self.make_project(root / "project")
            tracked.remove("english_world/model.py")
            with self.assertRaisesRegex(ValueError, "at least five"):
                self.prepare(root / "out", root / "project", tracked)

    def test_duplicate_source_is_never_split_across_train_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            tracked = self.make_project(project)
            duplicate = (project / "ops/worker_guard.py").read_bytes()
            (project / "english_world/__init__.py").write_bytes(duplicate)
            data = load_dataset(self.prepare(root / "out", project, tracked))
            records = {record["path"]: record for record in data["manifest"]["source_files"]}
            self.assertEqual(records["ops/worker_guard.py"]["split"], records["english_world/__init__.py"]["split"])
            self.assertEqual(records["ops/worker_guard.py"]["split"], "test")


if __name__ == "__main__":
    unittest.main()
