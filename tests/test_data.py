"""Offline checks for provenance, document splits, and bounded downloads."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from english_world.data import ALPHABET, fetch_file, load_dataset, normalize_text, prepare_dataset


class FakeResponse(io.BytesIO):
    def __init__(self, body, declared=None):
        super().__init__(body)
        self.headers = {} if declared is None else {"Content-Length": str(declared)}


class DataTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_document_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = prepare_dataset(root / "a", seed=23, max_examples=120)
            b = prepare_dataset(root / "b", seed=23, max_examples=120)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            data = load_dataset(a)
            self.assertEqual(len(data["train"]), 120)
            split_docs = {name: set(docs) for name, docs in data["texts"].items()}
            self.assertFalse(split_docs["train"] & split_docs["test"])
            self.assertFalse(split_docs["train"] & split_docs["validation"])
            self.assertFalse(split_docs["test"] & split_docs["validation"])
            for split in ("train", "validation", "test"):
                for example in data[split]:
                    self.assertEqual(len(example["target"]), 1)
                    self.assertTrue(any(example["context"] + example["target"] in doc for doc in data["texts"][split]))
            self.assertFalse(data["manifest"]["dictionary"]["spelling_authority"])

    def test_fixture_vocabulary_does_not_use_held_out_documents(self):
        documents = [{"text": f"This is training example number {i}. unique{chr(97+i)}word appears here."} for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"documents": documents}), encoding="utf-8")
            with patch("english_world.data.FIXTURE", fixture):
                data = load_dataset(prepare_dataset(root / "out", max_examples=20))
            train = " ".join(data["texts"]["train"])
            for word in data["dictionary"]:
                self.assertIn(word, train)
            held_out_unique = [word for doc in data["texts"]["test"] for word in doc.split() if word.startswith("unique")]
            for word in held_out_unique:
                self.assertNotIn(word, data["dictionary"])

    def test_download_cap_rejects_stream_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("english_world.data.urllib.request.urlopen", return_value=FakeResponse(b"a" * 11)):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    fetch_file("https://example.invalid/data", directory, max_bytes=10)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cached_download_is_verified_without_network(self):
        body = b"small known file"
        checksum = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            with patch("english_world.data.urllib.request.urlopen", return_value=FakeResponse(body)) as network:
                path, info = fetch_file("https://example.invalid/data", directory, expected_sha256=checksum)
                again, cached = fetch_file("https://example.invalid/data", directory, expected_sha256=checksum)
                self.assertEqual(network.call_count, 1)
                self.assertEqual(info, cached)
                self.assertEqual(path, again)
                path.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    fetch_file("https://example.invalid/data", directory, expected_sha256=checksum)

    def test_bad_download_hash_is_never_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("english_world.data.urllib.request.urlopen", return_value=FakeResponse(b"wrong")):
                with self.assertRaisesRegex(ValueError, "Checksum"):
                    fetch_file("https://example.invalid/data", directory, expected_sha256="0" * 64)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_normalization_and_schema_validation(self):
        normalized = normalize_text("HELLO\t\u201cFriend\u201d!\nA\u2014B \U0001f600")
        self.assertEqual(normalized, 'hello "friend"!\na-b')
        self.assertTrue(set(normalized) <= set(ALPHABET))
        with tempfile.TemporaryDirectory() as directory:
            path = prepare_dataset(directory, max_examples=10)
            data = load_dataset(path)
            data["train"][0]["target"] = "bad"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
