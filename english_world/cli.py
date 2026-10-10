"""Small, dependency-free command line; the model never invokes a hosted service."""

import argparse
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
    return Genome.from_dict(artifact["genome"]), artifact


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=True))


def doctor(_args):
    total, used, free = shutil.disk_usage(Path.cwd())
    print_json({
        "project": "alife-english", "version": __version__,
        "python": platform.python_version(), "platform": platform.system(),
        "cpu_count": os.cpu_count(), "working_directory": str(Path.cwd()),
        "disk_free_gib": round(free / 2**30, 2),
        "backend": "stdlib CPU", "requires_gpu": False,
        "pretrained_model": None,
        "capability": "experimental character continuation; general chat quality is unproven",
    })
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
    examples = data[args.split]
    if args.limit:
        examples = examples[:args.limit]
    result = score(genome, examples, data["dictionary"])
    report = {"checkpoint": str(args.checkpoint), "genome_hash": genome.digest(),
              "split": args.split, "metrics": result}
    if args.baselines:
        from .model import baseline_metrics
        report["baselines"] = baseline_metrics(data["train"], examples)
        report["baseline_note"] = "Empirical-count controls are fitted on training examples only, never used as evolutionary founders."
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print_json(report)
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
        # The first predictor has at most five characters of effective context.
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
    print("aLIFE English — experimental evolved character model")
    print("Continuation mode: up to 5 characters of effective context; general chat is unproven.")
    print("This checkpoint may produce fragments or nonsense; spelling is not general understanding.")
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
    ap = argparse.ArgumentParser(prog="english-world", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    commands = ap.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Inspect the local runtime without changing it")
    p.set_defaults(func=doctor)
    p = commands.add_parser("prepare", help="Prepare original fixtures or public English data")
    p.add_argument("--source", choices=("fixture", "english"), default="fixture")
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
        print(f"english-world: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
