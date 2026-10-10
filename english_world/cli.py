"""Small, dependency-free command line; the model never invokes a hosted service."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import tomllib

from . import __version__


def bounded_int(low, high):
    def convert(value):
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number
    return convert


def positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path):
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_champion(path):
    from .model import Genome
    artifact = read_json(path)
    if not isinstance(artifact, dict) or "genome" not in artifact:
        raise ValueError("Checkpoint is not a champion artifact (missing genome).")
    genome = Genome.from_dict(artifact["genome"])
    if artifact.get("genome_hash") is not None and artifact["genome_hash"] != genome.digest():
        raise ValueError("Champion genome hash does not match its content.")
    return genome, artifact


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=True))


def terminal_text(value):
    """Show model output as text without interpreting terminal control bytes."""
    return ''.join(c if c in '\n\t' or (ord(c) >= 32 and not 127 <= ord(c) < 160)
                   else f'\\x{ord(c):02x}' for c in value)


def doctor(_args):
    total, used, free = shutil.disk_usage(Path.cwd())
    print_json({
        "project": "Golem", "version": __version__,
        "python": platform.python_version(), "platform": platform.system(),
        "cpu_count": os.cpu_count(), "working_directory": str(Path.cwd()),
        "disk_free_gib": round(free / 2**30, 2),
        "backend": "stdlib CPU", "requires_gpu": False,
        "pretrained_model": None,
        "capability": "experimental character continuation; general chat quality is unproven",
    })
    return 0


def ternary_check(_args):
    from .ternary import self_test
    print_json(self_test())
    return 0


def graph_command(args):
    from .lexical_graph import build_graph, load_graph, normalize_word, rank_related, relation_path
    from .lexical_export import export_graphml, export_neighborhood, export_rank_table, ranked_words
    if args.graph_command == "build":
        path = build_graph(args.output)
        graph = load_graph(path)
        export_rank_table(graph, Path(args.output) / "ranked-words.tsv")
        print_json({"graph": str(path), "counts": graph["counts"], "rank_note": "Connectivity, not frequency or universal importance"})
        return 0
    if args.graph_command == "corpus":
        from .relation_corpus import prepare_relation_corpus
        path = prepare_relation_corpus(args.graph, args.output, dialogue_path=args.dialogue,
                                       seed=args.seed, max_concepts=args.max_concepts)
        print(f"Relation training corpus ready: {path}")
        return 0
    graph = load_graph(args.graph)
    if args.graph_command == "rank":
        print_json({"method": "WordNet connectivity PageRank aggregated over meanings", "words": ranked_words(graph, args.top)})
    elif args.graph_command == "related":
        senses = graph["word_index"].get(normalize_word(args.word), [])
        if not senses:
            raise ValueError("No exact WordNet lemma entry; try its singular/base form")
        print_json({"word": args.word, "senses": [{"id": s, "definition": graph["nodes"][s]["definition"]} for s in senses],
                    "score_note": "Heuristic typed relatedness, not a probability or synonym score",
                    "related": rank_related(graph, args.word, sense=args.sense, limit=args.limit, max_depth=args.depth)})
    elif args.graph_command == "pair":
        result = relation_path(graph, args.first, args.second, source_sense=args.source_sense,
                               target_sense=args.target_sense, max_hops=args.max_hops)
        result["readable_path"] = [{"from": graph["nodes"][edge["source"]]["lemmas"],
                                    "relation": edge["relation"],
                                    "to": graph["nodes"][edge["target"]]["lemmas"]}
                                   for edge in result.get("path", [])]
        print_json(result)
    elif args.graph_command == "export":
        base = Path(args.graph)
        if base.is_file():
            base = base.parent
        license_text = (base / "WORDNET-LICENSE.txt").read_text(encoding="utf-8")
        if args.format == "graphml":
            result = export_graphml(graph, args.output, license_text=license_text)
        else:
            result = export_neighborhood(graph, args.word, args.output, sense=args.sense,
                                         limit=args.limit, license_text=license_text)
        (Path(args.output).parent / "WORDNET-LICENSE.txt").write_text(license_text, encoding="utf-8")
        print(f"Graph export: {result}")
    return 0


def load_neural_input(path):
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "dataset.json"
    schema = read_json(candidate).get("schema")
    if schema == "golem-neural-corpus-v1":
        from .neural_corpus import load_neural_corpus
        return load_neural_corpus(candidate)
    if schema == "golem-relation-corpus-v1":
        from .relation_corpus import load_relation_corpus
        return load_relation_corpus(candidate)
    from .data import load_dataset
    return load_dataset(candidate)


def neural(args):
    if args.neural_command == "prepare":
        from .neural_corpus import prepare_neural_corpus
        print(f"Neural corpus ready: {prepare_neural_corpus(args.output, seed=args.seed, max_conversations=args.max_conversations)}")
        return 0
    if args.neural_command == "status":
        root = Path(args.run_dir)
        for name in ("report.json", "status.json"):
            if (root / name).exists():
                print_json(read_json(root / name))
                return 0
        raise FileNotFoundError(f"No neural report or status in {root}")
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Neural commands need the isolated PyTorch runtime; see requirements-neural.txt and docs/neural.md.") from exc
    if args.neural_command == "relation-eval":
        from .neural_model import load_neural_checkpoint
        from .relation_eval import evaluate_relations
        torch.set_num_threads(args.threads)
        checkpoint_sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        model, _metadata = load_neural_checkpoint(args.checkpoint)
        if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != checkpoint_sha:
            raise ValueError("Checkpoint changed while loading; retry evaluation or use a frozen copy")
        data = load_neural_input(args.data)
        result = evaluate_relations(model, data, split=args.split,
                                    max_cases=args.cases, graph_path=args.graph)
        result["provenance"] = {
            "checkpoint_name": Path(args.checkpoint).name,
            "checkpoint_sha256": checkpoint_sha,
            "corpus_content_sha256": data.get("manifest", {}).get("content_sha256"),
            "graph_sha256": hashlib.sha256(Path(args.graph).read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
        }
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print_json(result)
        return 0
    if args.neural_command in ("quantize", "relevance"):
        from .neural_eval import response_relevance, ternary_export
        from .neural_model import load_neural_checkpoint
        torch.set_num_threads(args.threads)
        data = load_neural_input(args.data)
        if args.neural_command == "quantize":
            result = ternary_export(args.checkpoint, data, args.output, split=args.split, already_ternary=args.already_ternary)
        else:
            model, _metadata = load_neural_checkpoint(args.checkpoint)
            result = response_relevance(model, data, split=args.split, max_cases=args.cases)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print_json(result)
        return 0
    if args.neural_command == "qat":
        from .neural_quantization import train_ternary
        config = load_config(args.config)
        config["threads"] = args.threads
        print_json(train_ternary(args.checkpoint, load_neural_input(args.data), args.run_dir, config))
        return 0
    if args.neural_command == "chat":
        from .neural_model import ASSISTANT, END, USER, encode_bytes, generate_neural, load_neural_checkpoint
        torch.set_num_threads(args.threads)
        model, metadata = load_neural_checkpoint(args.checkpoint, map_location="cpu")
        model.eval()
        if args.prompt is not None:
            print(terminal_text(generate_neural(model, args.prompt, max_new_tokens=args.max_bytes,
                                               temperature=args.temperature, seed=args.seed, dialogue=True)))
            return 0
        print("Project Golem — experimental recurrent neural model")
        print("New randomly initialized model trained on project data; general chat quality must be measured.")
        print("Commands: /info /reset /reload /quit. History is not saved.")
        history = []
        turn = 0
        while True:
            try:
                message = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if message.strip() in ("/quit", "/exit"):
                return 0
            if message.strip() == "/reset":
                history = []
                turn = 0
                print("History cleared.")
                continue
            if message.strip() == "/info":
                print_json(metadata)
                continue
            if message.strip() == "/reload":
                model, metadata = load_neural_checkpoint(args.checkpoint, map_location="cpu")
                model.eval()
                history = []
                turn = 0
                print("Latest saved neural checkpoint loaded; history cleared.")
                continue
            if message.strip():
                user_turn = [USER] + encode_bytes(message)[-args.context_bytes:] + [END, ASSISTANT]
                prompt = (history + user_turn)[-args.context_bytes:]
                answer = generate_neural(model, prompt, max_new_tokens=args.max_bytes,
                                         temperature=args.temperature, seed=args.seed + turn, dialogue=True)
                print("golem> " + terminal_text(answer))
                history = (prompt + encode_bytes(answer) + [END])[-args.context_bytes:]
                turn += 1
        return 0
    from .neural_training import benchmark as neural_benchmark, compare, train_memory
    config = load_config(args.config)
    if args.threads is not None:
        config["threads"] = args.threads
    if args.neural_command == "memory":
        result = train_memory(config, args.run_dir)
    else:
        data = load_neural_input(args.data)
        if args.neural_command == "benchmark":
            result = neural_benchmark(config, data, args.run_dir)
        else:
            result = compare(config, data, args.run_dir, resume=args.resume)
    print_json(result)
    return 0


def prepare(args):
    from .data import prepare_dataset
    result = prepare_dataset(args.output, source=args.source, seed=args.seed,
                             max_examples=args.max_examples, context_length=args.context_length)
    print(f"Dataset ready: {result}")
    return 0


def evolve(args):
    from .data import load_dataset
    from .evolution import train
    config = load_config(args.config)
    for key in ("generations", "max_seconds", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    result = train(config, load_dataset(args.data), args.run_dir, resume=args.resume)
    print_json(result)
    return 0


def status(args):
    root = Path(args.run_dir)
    for name in ("status.json", "champion.json"):
        candidate = root / name
        if candidate.exists():
            result = read_json(candidate)
            if name == "champion.json":
                result = {key: value for key, value in result.items() if key != "genome"}
            result["artifact"] = str(candidate)
            result["age_seconds"] = round(time.time() - candidate.stat().st_mtime, 1)
            print_json(result)
            return 0
    raise FileNotFoundError(f"No run status yet in {root}. Start evolve first.")


def evaluate(args):
    from .data import load_dataset
    from .model import evaluate as score
    genome, artifact = load_champion(args.checkpoint)
    data = load_dataset(args.data)
    if data.get("domain", "english") != getattr(genome, "domain", "english"):
        raise ValueError("Dataset and checkpoint domains differ; choose a matching code or English dataset.")
    examples = data[args.split]
    if args.limit:
        examples = examples[:args.limit]
    result = score(genome, examples, data["dictionary"])
    report = {"checkpoint": str(args.checkpoint), "genome_hash": genome.digest(),
              "split": args.split, "metrics": result}
    if args.baselines:
        from .model import baseline_metrics
        report["baselines"] = baseline_metrics(data["train"], examples, domain=getattr(genome, "domain", "english"))
        report["baseline_note"] = "Empirical-count controls are fitted on training examples only, never used as evolutionary founders."
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print_json(report)
    return 0


def self_check(args):
    from .model import self_prediction_metrics
    genome, artifact = load_champion(args.checkpoint)
    print_json({"checkpoint": args.checkpoint, "generation": artifact.get("generation"),
                "genome_hash": genome.digest(),
                "self_prediction": self_prediction_metrics(genome, max_examples=args.max_examples),
                "interpretation": "Diagnostic prediction of this genome's serialization; not a reward or proof of self-understanding."})
    return 0


def predict(args):
    from .model import Predictor, normalize_text
    genome, _artifact = load_champion(args.checkpoint)
    context = normalize_text(args.text, domain=getattr(genome, "domain", "english"))
    distribution = Predictor(genome).distribution(context)
    choices = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:args.top]
    print_json({"genome_hash": genome.digest(), "domain": getattr(genome, "domain", "english"),
                "effective_context": context[-genome.max_order:],
                "next_characters": [{"character": char, "probability": probability} for char, probability in choices]})
    return 0


def benchmark(args):
    from .data import load_dataset
    from .evolution import train
    config = load_config(args.config)
    target_generations = config.get("generations", 1000)
    config["generations"] = args.generations
    config["max_seconds"] = args.max_seconds
    start = time.monotonic()
    result = train(config, load_dataset(args.data), args.run_dir)
    elapsed = time.monotonic() - start
    print_json({"wall_seconds": round(elapsed, 3),
                "requested_generations": args.generations,
                "rough_target_minutes": round(elapsed / args.generations * target_generations / 60, 2),
                "estimate_note": "Startup included. Genome growth can slow later generations; not a quality ETA.",
                "result": result})
    return 0


def chat(args):
    from .model import generate
    genome, artifact = load_champion(args.checkpoint)
    history = ""
    message_no = 0

    def reply(message):
        nonlocal history, message_no
        # A short suffix predictor must see the actual end of user input.
        # A fixed "assistant: " suffix would hide every user's actual message.
        # Continue the user's text directly; do not pretend this is a role model.
        context = (history + message.rstrip("\r\n"))[-args.context_chars:]
        output = generate(genome, context, max_chars=args.max_chars,
                          seed=args.seed + message_no, temperature=args.temperature)
        message_no += 1
        history = (context + output + "\n")[-args.context_chars:]
        return output

    if args.prompt is not None:
        print(reply(args.prompt))
        return 0
    print("Project Golem — experimental evolved character model")
    domain = getattr(genome, "domain", "english")
    print(f"{domain.capitalize()} continuation mode: up to {genome.max_order} characters of effective context.")
    if domain == "code":
        print("Generated code is text only and is never executed; correctness and understanding are unproven.")
    else:
        print("This checkpoint may produce fragments or nonsense; general chat is unproven.")
    print(f"Checkpoint: {args.checkpoint} | genome: {genome.digest()[:12]}")
    print("Commands: /info /reset /reload /quit. Ctrl-C or Ctrl-D exits. Chat is not saved.")
    while True:
        try:
            message = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.strip() in ("/quit", "/exit"):
            return 0
        if message.strip() == "/reset":
            history = ""
            print("History cleared.")
            continue
        if message.strip() == "/info":
            print_json({key: value for key, value in artifact.items() if key != "genome"})
            continue
        if message.strip() == "/reload":
            genome, artifact = load_champion(args.checkpoint)
            history = ""
            print(f"Loaded {genome.digest()[:12]}; history cleared.")
            continue
        if message.strip():
            try:
                print("world> " + reply(message))
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")


def parser():
    ap = argparse.ArgumentParser(prog="golem", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    commands = ap.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Inspect the local runtime without changing it")
    p.set_defaults(func=doctor)
    p = commands.add_parser("ternary-check", help="Verify packed ternary arithmetic against exact integer results")
    p.set_defaults(func=ternary_check)
    p = commands.add_parser("graph", help="Dictionary relations, sense-aware ranks and training corpus")
    gcommands = p.add_subparsers(dest="graph_command", required=True)
    gp = gcommands.add_parser("build")
    gp.add_argument("--output", default="data/lexicon")
    gp.set_defaults(func=graph_command)
    for name in ("rank", "related", "pair", "export", "corpus"):
        gp = gcommands.add_parser(name)
        gp.add_argument("--graph", default="data/lexicon/graph.json")
        gp.set_defaults(func=graph_command)
        if name == "rank":
            gp.add_argument("--top", type=bounded_int(1, 1000), default=25)
        elif name == "related":
            gp.add_argument("word")
            gp.add_argument("--sense")
            gp.add_argument("--limit", type=bounded_int(1, 200), default=20)
            gp.add_argument("--depth", type=bounded_int(1, 3), default=2)
        elif name == "pair":
            gp.add_argument("first")
            gp.add_argument("second")
            gp.add_argument("--source-sense")
            gp.add_argument("--target-sense")
            gp.add_argument("--max-hops", type=bounded_int(1, 6), default=4)
        elif name == "export":
            gp.add_argument("--word", default="cat")
            gp.add_argument("--sense")
            gp.add_argument("--limit", type=bounded_int(1, 20), default=10)
            gp.add_argument("--format", choices=("neighborhood", "graphml"), default="neighborhood")
            gp.add_argument("--output", required=True)
        else:
            gp.add_argument("--output", default="data/relations")
            gp.add_argument("--dialogue")
            gp.add_argument("--seed", type=int, default=23)
            gp.add_argument("--max-concepts", type=bounded_int(10, 20000), default=6000)
    p = commands.add_parser("neural", help="Optional PyTorch neural model and controlled evolution experiments")
    ncommands = p.add_subparsers(dest="neural_command", required=True)
    np = ncommands.add_parser("prepare", help="Fetch a pinned English dialogue corpus with conversation-disjoint splits")
    np.add_argument("--output", default="data/neural-dialogue")
    np.add_argument("--seed", type=int, default=23)
    np.add_argument("--max-conversations", type=bounded_int(20, 10000), default=512)
    np.set_defaults(func=neural)
    np = ncommands.add_parser("relation-eval", help="Measure model likelihood on held-out dictionary facts without retrieval")
    np.add_argument("--checkpoint", required=True)
    np.add_argument("--data", default="data/relations/dataset.json")
    np.add_argument("--graph", default="data/lexicon/graph.json")
    np.add_argument("--split", choices=("validation", "test"), default="validation")
    np.add_argument("--cases", type=bounded_int(2, 512), default=64)
    np.add_argument("--threads", type=bounded_int(1, 2), default=1)
    np.add_argument("--output")
    np.set_defaults(func=neural)
    for name in ("benchmark", "compare", "memory"):
        np = ncommands.add_parser(name)
        np.add_argument("--config", default="configs/neural-memory.toml" if name == "memory" else "configs/neural-pilot.toml")
        np.add_argument("--data", default="data/neural-dialogue/dataset.json")
        np.add_argument("--run-dir", default=f"runs/neural-{name}")
        np.add_argument("--threads", type=bounded_int(1, 2))
        if name == "compare":
            np.add_argument("--resume", action="store_true")
        np.set_defaults(func=neural)
    np = ncommands.add_parser("status")
    np.add_argument("--run-dir", default="runs/neural-compare")
    np.set_defaults(func=neural)
    for name in ("quantize", "relevance"):
        np = ncommands.add_parser(name)
        np.add_argument("--checkpoint", required=True)
        np.add_argument("--data", default="data/neural-dialogue/dataset.json")
        np.add_argument("--split", choices=("validation", "test"), default="validation")
        np.add_argument("--output", required=name == "quantize")
        np.add_argument("--threads", type=bounded_int(1, 2), default=2)
        if name == "relevance":
            np.add_argument("--cases", type=bounded_int(2, 256), default=32)
        else:
            np.add_argument("--already-ternary", action="store_true", help="Pack verified ternary weights without quantizing them again")
        np.set_defaults(func=neural)
    np = ncommands.add_parser("qat", help="Bounded ternary-forward fine-tuning of an existing neural checkpoint")
    np.add_argument("--checkpoint", required=True)
    np.add_argument("--data", default="data/neural-dialogue/dataset.json")
    np.add_argument("--run-dir", default="runs/neural-qat")
    np.add_argument("--config", default="configs/neural-qat.toml")
    np.add_argument("--threads", type=bounded_int(1, 2), default=2)
    np.set_defaults(func=neural)
    np = ncommands.add_parser("chat")
    np.add_argument("--checkpoint", required=True)
    np.add_argument("--prompt")
    np.add_argument("--max-bytes", type=bounded_int(1, 2048), default=160)
    np.add_argument("--context-bytes", type=bounded_int(1, 8192), default=2048)
    np.add_argument("--temperature", type=positive_float, default=0.8)
    np.add_argument("--seed", type=int, default=1)
    np.add_argument("--threads", type=bounded_int(1, 2), default=1)
    np.set_defaults(func=neural)
    p = commands.add_parser("prepare", help="Prepare original fixtures or public English data")
    p.add_argument("--source", choices=("fixture", "english", "self-code"), default="fixture")
    p.add_argument("--output", default="data/fixture")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-examples", type=bounded_int(1, 1000000), default=8192)
    p.add_argument("--context-length", type=bounded_int(1, 4096), default=128)
    p.set_defaults(func=prepare)
    for name, fn in (("evolve", evolve), ("benchmark", benchmark)):
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/pilot.toml")
        p.add_argument("--data", default="data/english/dataset.json")
        p.add_argument("--run-dir", default="runs/pilot" if name == "evolve" else "runs/benchmark")
        p.add_argument("--generations", type=bounded_int(1, 1000000), default=None if name == "evolve" else 3)
        p.add_argument("--max-seconds", type=positive_float, default=None if name == "evolve" else 120)
        if name == "evolve":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--seed", type=int)
        p.set_defaults(func=fn)
    p = commands.add_parser("status")
    p.add_argument("--run-dir", default="runs/pilot")
    p.set_defaults(func=status)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", default="runs/pilot/champion.json")
    p.add_argument("--data", default="data/english/dataset.json")
    p.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    p.add_argument("--limit", type=bounded_int(1, 1000000), default=1024)
    p.add_argument("--baselines", action="store_true", help="Include count-fitted unigram and n-gram controls")
    p.add_argument("--output", help="Save this evaluation report as JSON")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("self-check", help="Measure prediction of the model's own serialized genome")
    p.add_argument("--checkpoint", default="runs/self-code/champion.json")
    p.add_argument("--max-examples", type=bounded_int(1, 4096), default=512)
    p.set_defaults(func=self_check)
    p = commands.add_parser("predict", help="Inspect next-character probabilities for a prefix")
    p.add_argument("--checkpoint", default="runs/self-code/champion.json")
    p.add_argument("--text", required=True)
    p.add_argument("--top", type=bounded_int(1, 97), default=8)
    p.set_defaults(func=predict)
    p = commands.add_parser("chat", help="Generate with a saved evolved predictor; no external model")
    p.add_argument("--checkpoint", default="runs/pilot/champion.json")
    p.add_argument("--prompt", help="One-shot generation instead of interactive chat")
    p.add_argument("--max-chars", type=bounded_int(1, 2048), default=160)
    p.add_argument("--context-chars", type=bounded_int(1, 65536), default=2048)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=positive_float, default=0.8)
    p.set_defaults(func=chat)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"golem: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
