"""Conversation supervision follows user facts, not assistant hallucinations."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from english_world.conversation_sim import (
    ACTS, EXPECTED_ANCHOR_LEMMAS, SPLITS, _build_episodes, _content_hash,
    _flatten, _input_key, _original_spec, _tokens, load_conversation_sim, prepare_conversation_sim,
)
from english_world.dialogue_state import extract_user_facts


class ConversationSimulationTests(unittest.TestCase):
    def graph(self, root):
        root.mkdir(parents=True, exist_ok=True)
        license_bytes = b"Original test fixture license.\n"
        (root / "WORDNET-LICENSE.txt").write_bytes(license_bytes)
        nodes = {identity: {"lemmas": [lemma], "pos": identity[0], "definition": "Original test anchor for " + lemma}
                 for identity, lemma in EXPECTED_ANCHOR_LEMMAS.items()}
        graph = {"schema": "golem-wordnet-graph-v1", "nodes": nodes,
                 "provenance": {"source": "synthetic test anchors", "license": {"path": "WORDNET-LICENSE.txt",
                   "sha256": hashlib.sha256(license_bytes).hexdigest()}}}
        path = root / "graph.json"
        path.write_text(json.dumps(graph), encoding="utf-8")
        return path

    def test_deterministic_verified_families_and_heldout_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = self.graph(root / "graph")
            a = prepare_conversation_sim(graph, root / "a", train_repetitions=1, evaluation_repetitions=1)
            b = prepare_conversation_sim(graph, root / "b", train_repetitions=1, evaluation_repetitions=1)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            data = load_conversation_sim(a)
            self.assertEqual(len(data["meaning_vocab"]), 18)
            self.assertEqual([row["act"] for row in data["meaning_vocab"]], list(ACTS))
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(set(data["groups"][left]) & set(data["groups"][right]))
                self.assertFalse(set(data["manifest"]["splits"][left]["explicit_user_names"]) & set(data["manifest"]["splits"][right]["explicit_user_names"]))
                self.assertFalse({_input_key(case["turns"]) for case in data["samples"][left]} &
                                 {_input_key(case["turns"]) for case in data["samples"][right]})

    def test_every_frame_matches_explicit_user_evidence_and_final_target_only(self):
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        samples, conversations = _flatten(episodes)
        for split in SPLITS:
            for sample, conversation in zip(samples[split], conversations[split]):
                expected = extract_user_facts(sample["turns"])
                self.assertEqual(sample["frame"]["user_name"], expected["user_name"])
                self.assertEqual(sample["frame"]["source_user_turn"], expected["source_user_turn"])
                self.assertEqual(sample["turns"][-1]["role"], "user")
                self.assertEqual(conversation["target_frames"], [sample["frame"]])
                self.assertEqual(conversation["turns"][-1]["content"], sample["reply"])
                self.assertNotIn(sample["frame"]["turn_index"], sample["untrusted_assistant_turns"])
            for episode in episodes[split]:
                for step, frame in zip(episode["steps"], episode["target_frames"]):
                    self.assertEqual(step["reference"], episode["turns"][frame["turn_index"]]["content"])

    def test_false_assistant_name_never_becomes_a_fact(self):
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        for split in SPLITS:
            unknown = [ep for ep in episodes[split] if ep["family_id"].startswith("unsupported_claim:")]
            self.assertTrue(unknown)
            for episode in unknown:
                self.assertTrue(episode["untrusted_assistant_turns"])
                frame = episode["target_frames"][0]
                self.assertIsNone(frame["user_name"])
                self.assertIsNone(frame["source_user_turn"])
                answer = episode["turns"][frame["turn_index"]]["content"]
                self.assertTrue("haven't told" in answer or "don't know" in answer)

    def test_name_correction_replaces_the_last_user_name_and_clear_removes_it(self):
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        for split in SPLITS:
            correction = next(ep for ep in episodes[split] if ep["family_id"].startswith("updating_name:"))
            frames = correction["target_frames"]
            self.assertNotEqual(frames[0]["user_name"], frames[2]["user_name"])
            self.assertEqual(frames[2]["user_name"], frames[3]["user_name"])
            clear = next(ep for ep in episodes[split] if ep["family_id"].startswith("forgetting_name:"))
            self.assertIsNone(clear["target_frames"][2]["user_name"])
            self.assertEqual(clear["target_frames"][2]["act"], "name_recall_unknown")

    def test_latest_question_contrast_has_identical_history_and_distinct_meanings(self):
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        for split in SPLITS:
            grouped = {}
            for episode in episodes[split]:
                if episode["contrast_kind"] == "latest_question":
                    grouped.setdefault(episode["contrast_group_id"], []).append(episode)
            for group in grouped.values():
                self.assertEqual(len(group), 3)
                self.assertEqual(len({json.dumps(ep["turns"][:4], sort_keys=True) for ep in group}), 1)
                self.assertEqual({ep["target_frames"][-1]["act"] for ep in group}, {"identity", "name_recall_known", "loneliness"})

    def test_memory_contrast_same_question_unknown_and_different_names(self):
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        for split in SPLITS:
            grouped = {}
            for episode in episodes[split]:
                if episode["counterfactual_pair_id"]:
                    grouped.setdefault(episode["counterfactual_pair_id"], []).append(episode)
            for group in grouped.values():
                self.assertEqual(len({ep["turns"][-2]["content"] for ep in group}), 1)
                self.assertEqual(len({ep["target_frames"][-1]["user_name"] for ep in group}), 3)
                self.assertIn(None, {ep["target_frames"][-1]["user_name"] for ep in group})

    def test_observed_trace_is_training_regression_not_heldout(self):
        episodes, _, regression = _build_episodes(_original_spec(), 97, 1, 1)
        self.assertEqual(len(regression), 4)
        self.assertTrue(all(ep in episodes["train"] for ep in regression if not ep.get("regression_runtime_only")))
        exact = next(ep for ep in regression if ep.get("regression_runtime_only"))
        self.assertNotIn(exact, episodes["train"])
        self.assertEqual([turn["content"] for turn in exact["turns"] if turn["role"] == "user"],
                         ["greet me", "i ask you to greet me", "what do i make of you", "it is?",
                          "ok. i am lonely", "are you there", "who are you"])
        self.assertEqual(len(exact["turns"]), 14)
        self.assertTrue(all(frame["user_name"] is None for frame in exact["target_frames"]))
        self.assertTrue(exact["allow_history_trimming"])
        self.assertLessEqual(_tokens(exact["turns"]), 1024)
        self.assertTrue(all(not ep["known_regression"] for split in ("validation", "test") for ep in episodes[split]))
        for split in SPLITS:
            for episode in episodes[split]:
                self.assertLessEqual(_tokens(episode["turns"]), 448)
                self.assertTrue(6 <= len(episode["turns"]) <= 8)

    def test_tampering_with_a_gold_name_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prepare_conversation_sim(self.graph(root / "graph"), root / "out", train_repetitions=1, evaluation_repetitions=1)
            data = load_conversation_sim(path)
            data["samples"]["train"][0]["frame"]["user_name"] = "Fabricated"
            data["manifest"]["content_sha256"] = _content_hash(data)
            path.write_text(json.dumps(data), encoding="utf-8")
            (path.parent / "manifest.json").write_text(json.dumps(data["manifest"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified episodes"):
                load_conversation_sim(path)

    def test_all_training_names_are_used_and_neural_context_bounds_hold(self):
        try:
            from english_world.semantic_model import encode_history
            from english_world.conversation_eval import score_conversation_turn
        except ImportError:
            self.skipTest("Optional neural runtime unavailable")
        spec = _original_spec()
        episodes, _, regression = _build_episodes(spec, 97, 12, 2)
        samples, _ = _flatten(episodes)
        actual_names = {row["frame"]["user_name"] for row in samples["train"] if row["frame"]["user_name"]}
        self.assertEqual(actual_names, set(spec["names"]["train"]))
        self.assertEqual(len(actual_names), 128)
        for split in SPLITS:
            for row in samples[split]:
                prefix = encode_history(row["turns"], max_context_tokens=512)
                self.assertLessEqual(len(prefix) + len(row["reply"].encode()) + 1, 512)
                self.assertLessEqual(len(row["reply"].encode()) + 1, 128)
                result = score_conversation_turn(row["reply"], row["frame"])
                self.assertTrue(result["passed"], (row["id"], row["reply"], result))
        for episode in regression:
            for frame in episode["target_frames"]:
                reply = episode["turns"][frame["turn_index"]]["content"]
                result = score_conversation_turn(reply, frame)
                self.assertTrue(result["passed"], (episode["id"], reply, result))


if __name__ == "__main__":
    unittest.main()
