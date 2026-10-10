"""Sense-disjoint, source-supported WordNet question/answer corpora.

Concept IDs are partitioned before questions or training ranks are constructed.
The model sees rendered conversations, not a lookup into the graph. Optional
dialogue mixing preserves the existing corpus partitions. Exact source facts,
sampling multiplicities, graph hashes, and limitations accompany every dataset.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil

from .data import _atomic_json, _sha256

SCHEMA = "golem-relation-corpus-v1"
GRAPH_SCHEMA = "golem-wordnet-graph-v1"
SPLITS = ("train", "validation", "test")
COMMON_SEEDS = (
    "person", "child", "family", "friend", "home", "house", "room", "door", "window",
    "water", "food", "bread", "fruit", "apple", "tree", "flower", "garden", "plant",
    "animal", "dog", "cat", "bird", "fish", "horse", "car", "bus", "train", "road",
    "city", "school", "book", "word", "language", "story", "music", "work", "play",
    "time", "day", "night", "year", "sun", "moon", "light", "dark", "hot", "cold",
    "good", "bad", "happy", "sad", "large", "small", "long", "short", "fast", "slow",
    "read", "write", "learn", "think", "know", "ask", "answer", "open", "close",
    "walk", "run", "eat", "drink", "sleep", "make", "give", "take", "help", "love",
)
RELATION_QUESTIONS = {
    "hypernym": "Give one hypernym (broader concept)",
    "instance_hypernym": "Give one class this instance belongs to",
    "hyponym": "Give one hyponym (narrower concept)",
    "instance_hyponym": "Give one instance belonging to this class",
    "part_meronym": "Give one part",
    "member_meronym": "Give one member",
    "substance_meronym": "Give one constituent substance",
    "part_holonym": "Give one whole that has this as a part",
    "member_holonym": "Give one group that has this as a member",
    "substance_holonym": "Give one whole made from this substance",
    "antonym": "Give one explicitly linked antonym",
}
POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective satellite", "r": "adverb"}
MAX_EDGE_FACTS_PER_CONCEPT = 8
MAX_PER_RELATION_PER_CONCEPT = 2
MAX_QUESTION_BYTES = 200
MAX_ANSWER_BYTES = 160


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _content_hash(dataset: dict) -> str:
    return _digest({key: value for key, value in dataset.items() if key != "manifest"})


def _display(lemma: str) -> str:
    return lemma.replace("_", " ")


def _normal_lemma(lemma: str) -> str:
    return " ".join(_display(lemma).casefold().split())


def _render(turns: list[dict]) -> str:
    rows = []
    for turn in turns:
        content = re.sub(r"(?im)^(user|assistant):", r" \1:", turn["content"])
        rows.append(turn["role"] + ": " + content)
    return "\n".join(rows)


def _validate_graph(graph: dict) -> None:
    if not isinstance(graph, dict) or graph.get("schema") != GRAPH_SCHEMA:
        raise ValueError("Expected a golem-wordnet-graph-v1 graph")
    if not isinstance(graph.get("nodes"), dict) or not isinstance(graph.get("edges"), list):
        raise ValueError("Graph requires nodes and edges")
    for identity, node in graph["nodes"].items():
        if not isinstance(identity, str) or not isinstance(node, dict):
            raise ValueError("Invalid concept record")
        if not isinstance(node.get("lemmas"), list) or not node["lemmas"] or any(
                not isinstance(lemma, str) or not lemma for lemma in node["lemmas"]):
            raise ValueError(f"Concept {identity} has invalid lemmas")
        if not isinstance(node.get("definition"), str) or node.get("pos") not in POS_NAMES:
            raise ValueError(f"Concept {identity} has an invalid definition or part of speech")
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or edge.get("source") not in graph["nodes"] or edge.get("target") not in graph["nodes"]:
            raise ValueError("Graph edge has a missing concept endpoint")
        if not isinstance(edge.get("relation"), str):
            raise ValueError("Graph edge has no relation type")


def _select_concepts(graph: dict, maximum: int, seed: int) -> list[str]:
    eligible = {identity for identity, node in graph["nodes"].items()
                if node["definition"].strip() and len(node["definition"].encode("utf-8")) <= MAX_ANSWER_BYTES
                and len(_display(node["lemmas"][0]).encode("utf-8")) <= 72}
    if len(eligible) < 10:
        raise ValueError("At least ten defined concepts are needed for held-out partitions")
    if len(eligible) <= maximum:
        return sorted(eligible)
    index = defaultdict(set)
    for identity in sorted(eligible):
        for lemma in graph["nodes"][identity]["lemmas"]:
            index[_normal_lemma(lemma)].add(identity)
    preferred = []
    seen = set()
    for word in COMMON_SEEDS:
        for identity in sorted(index[word]):
            if identity not in seen:
                preferred.append(identity)
                seen.add(identity)
    chosen = preferred[:maximum]
    # A bounded one-hop neighborhood provides actual relation examples without
    # using global PageRank as a surrogate for frequency or training importance.
    neighbors = set()
    for edge in graph["edges"]:
        if edge["source"] in seen and edge["target"] in eligible:
            neighbors.add(edge["target"])
        if edge["target"] in seen and edge["source"] in eligible:
            neighbors.add(edge["source"])
    rng = random.Random(seed)
    nearby = sorted(neighbors - set(chosen))
    rng.shuffle(nearby)
    chosen.extend(nearby[:max(0, maximum * 3 // 4 - len(chosen))])
    remainder = sorted(eligible - set(chosen))
    rng.shuffle(remainder)
    chosen.extend(remainder[:maximum - len(chosen)])
    return sorted(chosen)


def _assign_concepts(identities: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(set(identities))
    random.Random(seed).shuffle(ordered)
    held_out = max(1, len(ordered) // 10)
    return {identity: ("validation" if index < held_out else "test" if index < held_out * 2 else "train")
            for index, identity in enumerate(ordered)}


def _node_label(identity: str, node: dict, lemma: str | None = None) -> str:
    return f'"{_display(lemma or node["lemmas"][0])}" [sense {identity}; {POS_NAMES[node["pos"]]}]'


def _question_with_hint(prefix: str, definition: str) -> str | None:
    prefix += " Definition hint: "
    available = MAX_QUESTION_BYTES - len(prefix.encode("utf-8"))
    if available < 8:
        return None
    raw = definition.encode("utf-8")
    if len(raw) <= available:
        return prefix + definition
    hint = raw[:available - 3].decode("utf-8", errors="ignore").rsplit(" ", 1)[0].rstrip()
    return prefix + hint + "..."


def _finish_fact(relation, source, target, source_lemma, target_lemma, question, answer,
                 evidence, answer_lemmas=None) -> dict:
    identity = _digest({"relation": relation, "source": source, "target": target,
                        "source_lemma": source_lemma, "target_lemma": target_lemma, "evidence": evidence})[:24]
    lemmas = list(answer_lemmas or [])
    question_folded = question.casefold()
    visible = any(re.search(r"(?<!\w)" + re.escape(_display(lemma).casefold()) + r"(?!\w)", question_folded)
                  for lemma in lemmas)
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES or len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ValueError("Ontology prompt/answer exceeds its bounded context contract")
    text = _render([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    return {"id": identity, "relation": relation, "source": source, "target": target,
            "source_lemma": source_lemma, "target_lemma": target_lemma,
            "question": question, "answer": answer, "text": text, "evidence": evidence,
            "answer_lemmas": lemmas, "answer_concept": target,
            "answer_visible_in_prompt": visible, "conversation_id": "relation-" + identity}


def _node_facts(identity: str, node: dict) -> list[dict]:
    lemma = node["lemmas"][0]
    facts = [_finish_fact("definition", identity, identity, lemma, None,
                         f"What does {_node_label(identity, node)} mean in WordNet?",
                         node["definition"], {"kind": "node_definition", "node": identity}, [])]
    others = list(dict.fromkeys(value for value in node["lemmas"] if _normal_lemma(value) != _normal_lemma(lemma)))
    short_others = [value for value in others if len(_display(value).encode("utf-8")) <= MAX_ANSWER_BYTES]
    if short_others:
        question = _question_with_hint(f"Give one other lemma sharing {_node_label(identity, node)}.", node["definition"])
        if question is not None:
            facts.append(_finish_fact("synonym", identity, identity, lemma, None, question,
                                      _display(short_others[0]), {"kind": "node_lemmas", "node": identity}, others))
    return facts


def _edge_fact(edge: dict, nodes: dict) -> dict | None:
    relation = edge["relation"]
    if relation not in RELATION_QUESTIONS:
        return None
    source, target = edge["source"], edge["target"]
    left, right = nodes[source], nodes[target]
    source_lemma = edge.get("source_lemma") or left["lemmas"][0]
    target_lemma = edge.get("target_lemma") or right["lemmas"][0]
    if source_lemma not in left["lemmas"] or target_lemma not in right["lemmas"]:
        raise ValueError("Lexical relation lemma does not belong to its source synset")
    # Antonym pointers are lexical: never extend a relation between two specific
    # lemmas to every other word sharing either source or target sense.
    if relation == "antonym" and (not edge.get("source_lemma") or not edge.get("target_lemma")):
        return None
    question = _question_with_hint(f"{RELATION_QUESTIONS[relation]} for {_node_label(source, left, source_lemma)}.", left["definition"])
    answer = _node_label(target, right, target_lemma)
    if question is None or len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        return None
    alternatives = [target_lemma] if edge.get("target_lemma") else list(right["lemmas"])
    return _finish_fact(relation, source, target, source_lemma, target_lemma, question, answer,
                        {"kind": "edge", "edge": edge}, alternatives)


def _build_facts(nodes: dict, edges: list[dict], assignments: dict[str, str], seed: int) -> tuple[dict, dict]:
    facts = {split: [] for split in SPLITS}
    triples = {split: [] for split in SPLITS}
    candidates = defaultdict(dict)
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if assignments[source] != assignments[target]:
            continue
        fact = _edge_fact(edge, nodes)
        if fact is not None:
            candidates[source][fact["id"]] = (fact, edge)
    for identity in sorted(nodes):
        split = assignments[identity]
        facts[split].extend(_node_facts(identity, nodes[identity]))
        options = sorted(candidates[identity].values(), key=lambda pair: pair[0]["id"])
        random.Random(f"{seed}:{identity}").shuffle(options)
        per_relation = defaultdict(int)
        accepted = 0
        for fact, edge in options:
            if accepted >= MAX_EDGE_FACTS_PER_CONCEPT:
                break
            if per_relation[fact["relation"]] >= MAX_PER_RELATION_PER_CONCEPT:
                continue
            facts[split].append(fact)
            triples[split].append(edge)
            per_relation[fact["relation"]] += 1
            accepted += 1
    return facts, triples


def _conversation(fact: dict) -> dict:
    return {"id": fact["conversation_id"], "tree_id": fact["conversation_id"], "source": "wordnet",
            "fact_id": fact["id"], "relation": fact["relation"],
            "concept_ids": sorted({fact["source"], fact["target"]}),
            "turns": [{"role": "user", "content": fact["question"]},
                      {"role": "assistant", "content": fact["answer"]}]}


def _train_curriculum(nodes: dict, edges: list[dict], assignments: dict[str, str]) -> dict:
    from .lexical_graph import pagerank
    train_nodes = {identity: node for identity, node in nodes.items() if assignments[identity] == "train"}
    train_edges = [edge for edge in edges if edge["source"] in train_nodes and edge["target"] in train_nodes]
    ranks = pagerank(train_nodes, train_edges, iterations=30, damping=.85)
    if set(ranks) != set(train_nodes) or any(not math.isfinite(value) or value < 0 for value in ranks.values()):
        raise ValueError("Training PageRank returned invalid scores or non-training concepts")
    order = sorted(ranks, key=lambda identity: (-ranks[identity], identity))
    multiplicity = {identity: 1 + int(index < math.ceil(len(order) * .3)) + int(index < math.ceil(len(order) * .1))
                    for index, identity in enumerate(order)}
    return {"method": "uniform base plus bounded rank-weighted rehearsal", "rank_metric": "PageRank",
            "rank_scope": "training induced subgraph only", "rank_is_pair_relevance": False,
            "global_graph_rank_used": False, "scores": ranks, "concept_multiplicity": multiplicity,
            "maximum_multiplicity": 3, "max_edge_facts_per_concept": MAX_EDGE_FACTS_PER_CONCEPT,
            "max_facts_per_relation_per_concept": MAX_PER_RELATION_PER_CONCEPT,
            "note": "Every unique training fact appears once; high-ranked source concepts get one or two extra copies. Uniform document sampling implements this static mixture, not a temporal curriculum."}


def _copy_license(graph_path: Path, output: Path, graph: dict) -> dict:
    metadata = graph.get("provenance", {}).get("license", {})
    if metadata.get("path") != "WORDNET-LICENSE.txt":
        raise ValueError("WordNet graph must provide its WORDNET-LICENSE.txt provenance")
    source = graph_path.parent / metadata["path"]
    if source.is_symlink() or not source.resolve().is_relative_to(graph_path.parent.resolve()):
        raise ValueError("WordNet license path escapes its graph directory")
    if _sha256(source) != metadata.get("sha256"):
        raise ValueError("WordNet license checksum mismatch")
    shutil.copyfile(source, output / "WORDNET-LICENSE.txt")
    return {**metadata, "path": "WORDNET-LICENSE.txt"}


def _mix_dialogue(dataset: dict, dialogue_path: Path, output: Path, seed: int) -> dict:
    from .neural_corpus import load_neural_corpus
    corpus = load_neural_corpus(dialogue_path)
    if dialogue_path.is_dir():
        dialogue_path = dialogue_path / "dataset.json"
    selected = {split: [] for split in SPLITS}
    counts, rejected = {}, 0
    # Exact known graph answers from held-out concepts must not be introduced by
    # an optional dialogue document. This is not a semantic-overlap guarantee.
    heldout_definitions = {" ".join(fact["answer"].casefold().split()) for split in ("validation", "test")
                           for fact in dataset["facts"][split] if fact["relation"] == "definition" and len(fact["answer"]) >= 20}
    heldout_ids = {identity for identity, split in dataset["manifest"]["concept_assignments"].items() if split != "train"}
    for split in SPLITS:
        candidates = list(corpus["conversations"][split])
        random.Random(f"{seed}:dialogue:{split}").shuffle(candidates)
        maximum = len(dataset["texts"][split]) // 3  # added / (base + added) <= 25%.
        for original in candidates:
            if len(selected[split]) >= maximum:
                break
            text = _render(original["turns"])
            folded = " ".join(text.casefold().split())
            if split == "train" and (any(definition in folded for definition in heldout_definitions)
                                      or any(identity in text for identity in heldout_ids)):
                rejected += 1
                continue
            record = {**original, "id": "oasst1-" + original["tree_id"], "source": "oasst1",
                      "ontology_fact": False, "original_split": split}
            dataset["conversations"][split].append(record)
            dataset["texts"][split].append(text)
            selected[split].append(record)
        counts[split] = len(selected[split])
    for filename in ("OASST1-LICENSE.txt", "OASST1-DATASET-CARD.md"):
        shutil.copyfile(dialogue_path.parent / filename, output / filename)
    _atomic_json(output / "source_snapshot" / "selected_dialogue.json", selected)
    return {"source_dataset_sha256": _sha256(dialogue_path), "source_manifest": corpus["manifest"],
            "documents_added": counts, "maximum_document_share": .25,
            "known_heldout_definition_matches_rejected": rejected,
            "snapshot_path": "source_snapshot/selected_dialogue.json",
            "snapshot_sha256": _sha256(output / "source_snapshot" / "selected_dialogue.json"),
            "license_files": {name: _sha256(output / name) for name in ("OASST1-LICENSE.txt", "OASST1-DATASET-CARD.md")},
            "policy": "Original conversation-tree partitions preserved; dialogue is not asserted to be an ontology fact. Exact held-out definitions and explicit held-out sense IDs removed from training; unstructured semantic overlap is not ruled out."}


def prepare_relation_corpus(graph_path: str | Path, output_dir: str | Path,
                            dialogue_path: str | Path | None = None, seed: int = 23,
                            max_concepts: int = 6000) -> Path:
    """Build a pinned graph-derived corpus; no downloads or model training."""
    if type(seed) is not int or type(max_concepts) is not int or not 10 <= max_concepts <= 20000:
        raise ValueError("seed must be an integer and max_concepts must be in 10..20000")
    graph_path, output = Path(graph_path), Path(output_dir)
    if graph_path.is_dir():
        graph_path = graph_path / "graph.json"
    graph_sha = _sha256(graph_path)
    dialogue_sha = None
    if dialogue_path is not None:
        dialogue_path = Path(dialogue_path)
        if dialogue_path.is_dir():
            dialogue_path = dialogue_path / "dataset.json"
        dialogue_sha = _sha256(dialogue_path)
    parameters = {"seed": seed, "max_concepts": max_concepts, "graph_source_sha256": graph_sha,
                  "dialogue_source_sha256": dialogue_sha}
    path = output / "dataset.json"
    if path.exists():
        saved = load_relation_corpus(path)
        if saved["manifest"]["parameters"] != parameters:
            raise ValueError("Existing corpus has different inputs; use a new output directory")
        return path
    if output.exists() and any(output.iterdir()):
        raise ValueError("Relation corpus output directory must be empty")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    _validate_graph(graph)
    identities = _select_concepts(graph, max_concepts, seed)
    assignments = _assign_concepts(identities, seed)  # Before facts, templates, or ranks.
    nodes = {identity: graph["nodes"][identity] for identity in identities}
    edges = [edge for edge in graph["edges"] if edge["source"] in nodes and edge["target"] in nodes]
    facts, triples = _build_facts(nodes, edges, assignments, seed)
    curriculum = _train_curriculum(nodes, edges, assignments)
    conversations = {split: [_conversation(fact) for fact in facts[split]] for split in SPLITS}
    texts = {}
    for split in SPLITS:
        texts[split] = []
        for fact, conversation in zip(facts[split], conversations[split]):
            copies = curriculum["concept_multiplicity"][fact["source"]] if split == "train" else 1
            texts[split].extend([_render(conversation["turns"])] * copies)
    curriculum["unique_training_facts"] = len(facts["train"])
    curriculum["weighted_training_documents"] = len(texts["train"])
    curriculum["uniform_base_document_share"] = len(facts["train"]) / len(texts["train"])
    output.mkdir(parents=True, exist_ok=True)
    license_info = _copy_license(graph_path, output, graph)
    snapshot = {"schema": "golem-wordnet-subgraph-snapshot-v1", "source_graph_sha256": graph_sha,
                "nodes": nodes, "edges": edges, "assignments": assignments, "provenance": graph.get("provenance", {})}
    _atomic_json(output / "source_snapshot" / "graph.json", snapshot)
    manifest = {"parameters": parameters, "source": "Princeton WordNet relation templates",
                "graph_source_sha256": graph_sha, "graph_provenance": graph.get("provenance", {}),
                "license": license_info, "snapshot_path": "source_snapshot/graph.json",
                "snapshot_sha256": _sha256(output / "source_snapshot" / "graph.json"),
                "selection": "Concepts with full definitions <=160 UTF-8 bytes and primary lemmas <=72 bytes; manual common-lemma seed preference, bounded one-hop neighborhood, then seeded uniform fill; not a corpus-frequency ranking.",
                "selection_seed_lemmas": list(COMMON_SEEDS), "available_concepts": len(graph["nodes"]),
                "selected_concepts": len(nodes), "concept_assignments": assignments,
                "split_policy": "Seeded approximately 80/10/10 synset IDs, assigned before question construction; an edge fact requires both endpoints in the same partition.",
                "node_disjoint_proof": {"pairwise_intersections": {"train_validation": [], "train_test": [], "validation_test": []},
                                        "all_facts_have_partition_local_endpoints": True},
                "cross_partition_edges_excluded": sum(assignments[edge["source"]] != assignments[edge["target"]] for edge in edges),
                "curriculum": curriculum, "dialogue_mix": None,
                "text_transforms": "Full definition answers retained; source gloss hints may end with an explicit ellipsis; lemma underscores displayed as spaces; fixed sense-ID/POS question templates; internal role-looking lines indented.",
                "context_contract": {"question_max_utf8_bytes": MAX_QUESTION_BYTES, "answer_max_utf8_bytes": MAX_ANSWER_BYTES,
                                     "max_role_encoded_tokens": MAX_QUESTION_BYTES + MAX_ANSWER_BYTES + 4,
                                     "recommended_sequence_length": 384},
                "limitations": ["Synset-disjoint is not word-disjoint: a polysemous lemma can occur with different meanings in different partitions.",
                                "These concepts are held out from this graph-training stage only; earlier language pretraining may contain related words or facts.",
                                "Relation questions contain the source gloss; some answer lemmas occur in that gloss. Those cases are marked as possible contextual extraction.",
                                "Template continuation loss alone does not demonstrate relation knowledge. Exact graph retrieval and learned neural answers are separate capabilities.",
                                "Training PageRank measures graph centrality, not a pairwise semantic-relevance or truth score."]}
    dataset = {"schema": SCHEMA, "domain": "dialogue_utf8", "texts": texts, "conversations": conversations,
               "relation_texts": {split: [fact["text"] for fact in facts[split]] for split in SPLITS},
               "facts": facts, "raw_triples": triples, "dictionary": [], "manifest": manifest}
    if dialogue_path is not None:
        manifest["dialogue_mix"] = _mix_dialogue(dataset, dialogue_path, output, seed)
    manifest["splits"] = {split: {"concepts": sum(value == split for value in assignments.values()),
                                  "unique_facts": len(facts[split]), "unique_conversations": len(conversations[split]),
                                  "documents_including_rehearsal": len(texts[split]),
                                  "answer_visible_in_prompt": sum(fact["answer_visible_in_prompt"] for fact in facts[split]),
                                  "relation_counts": {relation: sum(fact["relation"] == relation for fact in facts[split])
                                                      for relation in sorted({fact["relation"] for fact in facts[split]})}}
                          for split in SPLITS}
    manifest["content_sha256"] = _content_hash(dataset)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(path, dataset)
    load_relation_corpus(path)
    return path


def load_relation_corpus(path: str | Path) -> dict:
    """Verify saved source evidence, concept separation, texts and licenses."""
    path = Path(path)
    if path.is_dir():
        path = path / "dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("schema") != SCHEMA:
        raise ValueError("Unsupported relation corpus schema")
    manifest = dataset.get("manifest", {})
    if manifest.get("content_sha256") != _content_hash(dataset):
        raise ValueError("Relation corpus content checksum mismatch")
    if json.loads((path.parent / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise ValueError("Relation corpus manifest mismatch")
    if manifest.get("snapshot_path") != "source_snapshot/graph.json":
        raise ValueError("Unexpected relation graph snapshot path")
    snapshot_path = path.parent / manifest["snapshot_path"]
    if _sha256(snapshot_path) != manifest.get("snapshot_sha256"):
        raise ValueError("Relation graph snapshot checksum mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot["source_graph_sha256"] != manifest["graph_source_sha256"]:
        raise ValueError("Relation source graph identity mismatch")
    if manifest.get("license", {}).get("path") != "WORDNET-LICENSE.txt" or _sha256(path.parent / "WORDNET-LICENSE.txt") != manifest["license"]["sha256"]:
        raise ValueError("Relation corpus license checksum mismatch")
    assignments = manifest["concept_assignments"]
    if assignments != snapshot["assignments"] or set(assignments) != set(snapshot["nodes"]):
        raise ValueError("Concept assignment differs from graph snapshot")
    if any(split not in SPLITS for split in assignments.values()):
        raise ValueError("Invalid concept partition")
    expected_facts, expected_triples = _build_facts(snapshot["nodes"], snapshot["edges"], assignments, manifest["parameters"]["seed"])
    if dataset.get("facts") != expected_facts or dataset.get("raw_triples") != expected_triples:
        raise ValueError("Generated facts do not match their source graph evidence")
    if dataset.get("relation_texts") != {split: [fact["text"] for fact in expected_facts[split]] for split in SPLITS}:
        raise ValueError("Unique relation texts do not match source facts")
    curriculum = manifest["curriculum"]
    train_ids = {identity for identity, split in assignments.items() if split == "train"}
    if set(curriculum["scores"]) != train_ids or set(curriculum["concept_multiplicity"]) != train_ids:
        raise ValueError("Training ranks include non-training concepts")
    if any(type(value) is not int or not 1 <= value <= 3 for value in curriculum["concept_multiplicity"].values()):
        raise ValueError("Invalid training sampling multiplicity")
    seen_texts, seen_trees = {}, {}
    for split in SPLITS:
        texts, conversations = dataset["texts"][split], dataset["conversations"][split]
        if not texts or not expected_facts[split]:
            raise ValueError("Empty relation corpus partition")
        expected_wordnet = [_conversation(fact) for fact in expected_facts[split]]
        if [row for row in conversations if row.get("source") == "wordnet"] != expected_wordnet:
            raise ValueError("Fact conversations differ from their source facts")
        expected_texts = Counter()
        for fact in expected_facts[split]:
            expected_texts[fact["text"]] += curriculum["concept_multiplicity"][fact["source"]] if split == "train" else 1
        for row in conversations:
            if row.get("source") == "oasst1":
                expected_texts[_render(row["turns"])] += 1
            elif row.get("source") != "wordnet":
                raise ValueError("Unknown relation corpus conversation source")
        if Counter(texts) != expected_texts:
            raise ValueError("Training texts differ from supported facts and declared rehearsal multiplicity")
        for record in conversations:
            text = _render(record["turns"])
            if text not in texts:
                raise ValueError("Conversation does not match rendered text")
            key = (record.get("source"), record["tree_id"])
            if key in seen_trees and seen_trees[key] != split:
                raise ValueError("Conversation tree crosses partitions")
            seen_trees[key] = split
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("Invalid relation corpus text")
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_texts and seen_texts[digest] != split:
                raise ValueError("Identical document crosses partitions")
            seen_texts[digest] = split
        for fact in expected_facts[split]:
            if assignments[fact["source"]] != split or assignments[fact["target"]] != split:
                raise ValueError("Fact endpoint crosses partitions")
    mix = manifest.get("dialogue_mix")
    if mix is not None:
        if mix.get("snapshot_path") != "source_snapshot/selected_dialogue.json" or _sha256(path.parent / mix["snapshot_path"]) != mix["snapshot_sha256"]:
            raise ValueError("Dialogue mixture snapshot checksum mismatch")
        for name, checksum in mix["license_files"].items():
            if name not in ("OASST1-LICENSE.txt", "OASST1-DATASET-CARD.md") or _sha256(path.parent / name) != checksum:
                raise ValueError("Dialogue mixture license checksum mismatch")
        selected_dialogue = json.loads((path.parent / mix["snapshot_path"]).read_text(encoding="utf-8"))
        for split in SPLITS:
            rows = [row for row in dataset["conversations"][split] if row.get("source") == "oasst1"]
            if selected_dialogue.get(split) != rows:
                raise ValueError("Dialogue mixture differs from its source snapshot")
            if any(row.get("original_split") != split for row in rows) or len(rows) * 4 > len(dataset["texts"][split]):
                raise ValueError("Dialogue mixture violates split or document-share limit")
    elif any(row.get("source") != "wordnet" for split in SPLITS for row in dataset["conversations"][split]):
        raise ValueError("External dialogue records lack mixture provenance")
    return dataset
