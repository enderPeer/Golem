"""Graph facts remain sense-specific and evaluation concepts stay out of ranks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.relation_corpus import (
    GRAPH_SCHEMA, SPLITS, _assign_concepts, _content_hash, _render,
    load_relation_corpus, prepare_relation_corpus,
)


def toy_graph():
    nodes = {f"n:{index:08d}": {"lemmas": [f"word{index}", f"alias{index}"], "pos": "n",
                                "definition": f"the definition of distinct concept number {index}"}
             for index in range(30)}
    nodes["n:00000000"] = {"lemmas": ["bank", "financial_institution"], "pos": "n",
                            "definition": "an institution that accepts deposits"}
    nodes["n:00000001"] = {"lemmas": ["bank", "riverbank"], "pos": "n",
                            "definition": "sloping land beside a river"}
    edges = [{"source": source, "target": target, "relation": "hypernym"}
             for source in nodes for target in nodes if source != target]
    edges += [{"source": "n:00000002", "target": "n:00000003", "relation": "antonym",
               "source_lemma": "word2", "target_lemma": "alias3"},
              {"source": "n:00000003", "target": "n:00000002", "relation": "part_meronym"}]
    return {"schema": GRAPH_SCHEMA, "nodes": nodes, "edges": edges,
            "word_index": {"bank": ["n:00000000", "n:00000001"]},
            "rank": {identity: 1.0 for identity in nodes}, "provenance": {"source": "original toy graph"}}


class RelationCorpusTests(unittest.TestCase):
    def write_graph(self, root, graph=None):
        root.mkdir(parents=True, exist_ok=True)
        graph = copy.deepcopy(graph or toy_graph())
        license_text = "Original synthetic test graph; used only for tests.\n"
        (root / "WORDNET-LICENSE.txt").write_bytes(license_text.encode("utf-8"))
        graph["provenance"]["license"] = {"path": "WORDNET-LICENSE.txt",
                                          "sha256": hashlib.sha256(license_text.encode()).hexdigest(),
                                          "url": "https://example.invalid/test-license"}
        path = root / "graph.json"
        path.write_text(json.dumps(graph, sort_keys=True), encoding="utf-8")
        return path

    def test_deterministic_disjoint_concepts_and_supported_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            graph_path = self.write_graph(root / "source", graph)
            first = prepare_relation_corpus(graph_path, root / "a", seed=23, max_concepts=30)
            second = prepare_relation_corpus(graph_path, root / "b", seed=23, max_concepts=30)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            data = load_relation_corpus(first)
            assigned = data["manifest"]["concept_assignments"]
            self.assertEqual(len(assigned), 30)
            counts = {split: sum(value == split for value in assigned.values()) for split in SPLITS}
            self.assertEqual(counts, {"train": 24, "validation": 3, "test": 3})
            for split in SPLITS:
                for fact in data["facts"][split]:
                    self.assertEqual(assigned[fact["source"]], split)
                    self.assertEqual(assigned[fact["target"]], split)
                    self.assertIn(fact["source"], fact["question"])
                    self.assertIn("noun", fact["question"])
                    self.assertLessEqual(len(fact["question"].encode("utf-8")), 200)
                    self.assertLessEqual(len(fact["answer"].encode("utf-8")), 160)
                    self.assertEqual(fact["text"], _render([{"role": "user", "content": fact["question"]},
                                                           {"role": "assistant", "content": fact["answer"]}]))
                    if fact["relation"] == "definition":
                        self.assertEqual(fact["answer"], graph["nodes"][fact["source"]]["definition"])
                        self.assertNotIn(fact["answer"], fact["question"])
                    elif fact["evidence"]["kind"] == "edge":
                        self.assertIn(fact["evidence"]["edge"], graph["edges"])
            self.assertGreater(data["manifest"]["cross_partition_edges_excluded"], 0)

    def test_polysemous_words_are_never_merged_into_synonyms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            seed = next(seed for seed in range(100) if _assign_concepts(list(graph["nodes"]), seed)["n:00000000"] !=
                        _assign_concepts(list(graph["nodes"]), seed)["n:00000001"])
            data = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source", graph), root / "out",
                                                               seed=seed, max_concepts=30))
            synonyms = {fact["source"]: fact for split in SPLITS for fact in data["facts"][split] if fact["relation"] == "synonym"}
            self.assertEqual(synonyms["n:00000000"]["answer"], "financial institution")
            self.assertEqual(synonyms["n:00000001"]["answer"], "riverbank")
            self.assertNotIn("riverbank", synonyms["n:00000000"]["answer"])

    def test_training_rank_never_reads_full_graph_rank_or_heldout_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            assignments = _assign_concepts(list(graph["nodes"]), 23)
            first = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source-a", graph), root / "a", max_concepts=30))
            changed = copy.deepcopy(graph)
            changed["rank"] = {identity: index ** 4 for index, identity in enumerate(graph["nodes"])}
            changed["edges"] = [edge for edge in graph["edges"] if assignments[edge["source"]] == "train" and assignments[edge["target"]] == "train"]
            second = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source-b", changed), root / "b", max_concepts=30))
            self.assertEqual(first["manifest"]["curriculum"], second["manifest"]["curriculum"])
            ranks = first["manifest"]["curriculum"]["scores"]
            self.assertEqual(set(ranks), {identity for identity, split in assignments.items() if split == "train"})
            self.assertEqual(first["texts"]["train"], second["texts"]["train"])

    def test_curriculum_is_bounded_and_evaluation_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source"), root / "out", max_concepts=30))
            for split in SPLITS:
                for record in data["conversations"][split]:
                    count = data["texts"][split].count(_render(record["turns"]))
                    self.assertTrue(1 <= count <= 3 if split == "train" else count == 1)
            self.assertGreaterEqual(data["manifest"]["curriculum"]["uniform_base_document_share"], 1 / 3)

    def test_tampered_fact_rejected_even_if_content_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prepare_relation_corpus(self.write_graph(root / "source"), root / "out", max_concepts=30)
            data = load_relation_corpus(path)
            data["facts"]["train"][0]["answer"] = "Invented unsupported answer"
            data["manifest"]["content_sha256"] = _content_hash(data)
            path.write_text(json.dumps(data), encoding="utf-8")
            (path.parent / "manifest.json").write_text(json.dumps(data["manifest"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source graph evidence"):
                load_relation_corpus(path)

    def test_lexical_antonym_is_not_extended_to_every_synonym(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            seed = next(seed for seed in range(100) if _assign_concepts(list(graph["nodes"]), seed)["n:00000002"] ==
                        _assign_concepts(list(graph["nodes"]), seed)["n:00000003"])
            data = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source", graph), root / "out", seed=seed, max_concepts=30))
            antonyms = [fact for split in SPLITS for fact in data["facts"][split] if fact["relation"] == "antonym"]
            self.assertEqual(len(antonyms), 1)
            self.assertEqual(antonyms[0]["source_lemma"], "word2")
            self.assertEqual(antonyms[0]["answer_lemmas"], ["alias3"])

    def test_optional_dialogue_preserves_partitions_share_and_known_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            assignments = _assign_concepts(list(graph["nodes"]), 23)
            heldout = next(identity for identity, split in assignments.items() if split == "test")
            corpus = {"conversations": {split: [] for split in SPLITS}, "manifest": {"source": "original test dialogues"}}
            for split in SPLITS:
                for index in range(20):
                    corpus["conversations"][split].append({"tree_id": f"{split}-{index}", "turns": [
                        {"role": "user", "content": f"Original {split} question {index}?"},
                        {"role": "assistant", "content": f"Original {split} response {index}."}]})
            corpus["conversations"]["train"].append({"tree_id": "leaky", "turns": [
                {"role": "user", "content": "What is this definition?"},
                {"role": "assistant", "content": graph["nodes"][heldout]["definition"]}]})
            corpus_path = root / "dialogue/dataset.json"
            corpus_path.parent.mkdir()
            corpus_path.write_bytes(b"{}")
            for name in ("OASST1-LICENSE.txt", "OASST1-DATASET-CARD.md"):
                (corpus_path.parent / name).write_bytes(b"Original test fixture")
            with patch("english_world.neural_corpus.load_neural_corpus", return_value=corpus):
                data = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source", graph), root / "out",
                                                                   dialogue_path=corpus_path, max_concepts=30))
            self.assertEqual(data["manifest"]["dialogue_mix"]["known_heldout_definition_matches_rejected"], 1)
            for split in SPLITS:
                selected = [row for row in data["conversations"][split] if row["source"] == "oasst1"]
                self.assertTrue(all(row["original_split"] == split for row in selected))
                self.assertTrue(all(row["tree_id"] != "leaky" for row in selected))
                self.assertLessEqual(len(selected) * 4, len(data["texts"][split]))

    def test_long_glosses_are_filtered_and_hints_stay_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = toy_graph()
            graph["nodes"]["n:00000029"]["definition"] = "long source definition " * 30
            graph["nodes"]["n:00000028"]["definition"] = "a source definition with enough detail to need a shortened hint " * 2
            data = load_relation_corpus(prepare_relation_corpus(self.write_graph(root / "source", graph), root / "out", max_concepts=30))
            self.assertNotIn("n:00000029", data["manifest"]["concept_assignments"])
            for split in SPLITS:
                for fact in data["facts"][split]:
                    self.assertLessEqual(len(fact["question"].encode()), 200)
                    self.assertLessEqual(len(fact["answer"].encode()), 160)


if __name__ == "__main__":
    unittest.main()
