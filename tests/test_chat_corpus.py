"""Original chat data preserves families, role masks and counterfactual context."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.chat_corpus import (
    SPLITS, _canonical_input, _content_hash, _original_spec, _build_cases,
    _role_tokens, _training_variants, load_chat_corpus, prepare_chat_corpus,
)


class ChatCorpusTests(unittest.TestCase):
    def test_deterministic_original_corpus_and_disjoint_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = prepare_chat_corpus(root / "a")
            b = prepare_chat_corpus(root / "b")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            data = load_chat_corpus(a)
            self.assertGreaterEqual(len(data["chat_cases"]["train"]), 300)
            self.assertEqual(len(data["manifest"]["curriculum"]["intents"]), 12)
            families = {split: set(data["groups"][split]) for split in SPLITS}
            inputs = {split: {_canonical_input(case["turns"]) for case in data["chat_cases"][split]} for split in SPLITS}
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(families[left] & families[right])
                self.assertFalse(inputs[left] & inputs[right])
            self.assertFalse(data["manifest"]["external_downloads"])

    def test_heldout_cases_are_not_augmented_and_do_not_drive_train_variants(self):
        spec = _original_spec()
        before, _ = _build_cases(spec, 61)
        altered = copy.deepcopy(spec)
        altered["intents"]["identity"]["families"]["test"][0][1][0] = "A completely different held-out identity question?"
        after, _ = _build_cases(altered, 61)
        self.assertEqual(before["train"], after["train"])
        self.assertTrue(all(case["variant"] == "original" for split in ("validation", "test") for case in before[split]))
        self.assertTrue(any(case["variant"] == "single_transposition" for case in before["train"]))
        self.assertTrue(any(case["variant"] == "spacing" for case in before["train"]))

    def test_name_pairs_change_only_the_supplied_name(self):
        cases, _ = _build_cases(_original_spec(), 61)
        for split in SPLITS:
            pairs = {}
            for case in cases[split]:
                if case["context_pair_id"]:
                    pairs.setdefault(case["context_pair_id"], []).append(case)
            self.assertTrue(pairs)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                left, right = pair
                self.assertNotEqual(left["memory_value"], right["memory_value"])
                self.assertEqual(left["turns"][1:], right["turns"][1:])
                self.assertEqual(left["turns"][0]["content"].replace(left["memory_value"], "NAME"),
                                 right["turns"][0]["content"].replace(right["memory_value"], "NAME"))
                self.assertIn(left["memory_value"], left["reference"])
                self.assertIn(right["memory_value"], left["rubric"]["forbidden"])

    def test_context_bounds_and_assistant_only_masks(self):
        try:
            from english_world.neural_data import dialogue_document
        except ImportError:
            self.skipTest("Optional neural runtime unavailable")
        with tempfile.TemporaryDirectory() as directory:
            data = load_chat_corpus(prepare_chat_corpus(directory))
            for split in SPLITS:
                for record in data["conversations"][split]:
                    doc = dialogue_document(record["turns"])
                    self.assertEqual(len(doc.tokens), _role_tokens(record["turns"]))
                    self.assertLessEqual(len(doc.tokens), 160 if len(record["turns"]) > 2 else 128)
                    expected = sum(len(turn["content"].encode()) + 1 for turn in record["turns"] if turn["role"] == "assistant")
                    self.assertEqual(sum(doc.loss_mask), expected)
                    offset = 0
                    for turn in record["turns"]:
                        length = len(turn["content"].encode()) + 2
                        self.assertFalse(doc.loss_mask[offset])
                        if turn["role"] == "user":
                            self.assertFalse(any(doc.loss_mask[offset:offset + length]))
                        offset += length

    def test_source_snapshot_prevents_silent_training_example_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prepare_chat_corpus(directory)
            original = path.read_bytes()
            with patch("english_world.chat_corpus._original_spec", side_effect=AssertionError("should reuse saved original source")):
                self.assertEqual(prepare_chat_corpus(directory), path)
            self.assertEqual(path.read_bytes(), original)
            data = load_chat_corpus(path)
            data["chat_cases"]["train"][0]["reference"] = "I am always correct."
            data["manifest"]["content_sha256"] = _content_hash(data)
            path.write_text(json.dumps(data), encoding="utf-8")
            (path.parent / "manifest.json").write_text(json.dumps(data["manifest"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original source families"):
                load_chat_corpus(path)

    def test_typo_augmentation_is_bounded_and_deterministic(self):
        prompt = "who are you?"
        first = _training_variants(prompt, 61, "identity:who_question")
        self.assertEqual(first, _training_variants(prompt, 61, "identity:who_question"))
        self.assertLessEqual(len(first), 5)
        self.assertIn(("original", prompt), first)
        self.assertTrue(any(text == "whoareyou?" for _, text in first))

    def test_greeting_followups_preserve_turns_and_source_families(self):
        cases, groups = _build_cases(_original_spec(), 61)
        for split in SPLITS:
            contextual = [case for case in cases[split] if case["interaction_kind"] == "greeting_followup"]
            self.assertEqual({case["intent"] for case in contextual}, {"identity", "help"})
            for case in contextual:
                self.assertEqual([turn["role"] for turn in case["turns"]], ["user", "assistant", "user"])
                self.assertTrue(set(case["context_families"]) <= set(groups[split]))
                self.assertEqual(case["variant"], "original")

    def test_original_reference_answers_satisfy_independent_rubrics(self):
        try:
            from english_world.chat_eval import score_response
        except ImportError:
            self.skipTest("Optional neural evaluator unavailable")
        cases, _ = _build_cases(_original_spec(), 61)
        for split in SPLITS:
            for case in cases[split]:
                result = score_response(case["reference"], case["rubric"])
                self.assertTrue(result["heuristic_pass"], (case["id"], result["failures"]))


if __name__ == "__main__":
    unittest.main()
