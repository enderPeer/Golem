# Project Golem

A separate evolutionary English world, with reproducible experiments and a terminal interface to a saved character predictor.

**Status: experimental.** The legacy worlds evolve small character prediction programs and integer context tables without gradients. A separate neural backend uses hybrid learning. Neither backend uses pretrained or hosted language models, and a working chat command does not establish conversational understanding.

The optional **neural backend** adds a 791,300-parameter GRU trained from scratch, a delayed-memory test, and a four-candidate hybrid evolutionary comparison. It intentionally uses gradient learning inside candidates. The legacy table worlds remain mutation-only. See [neural Golem](docs/neural.md) and [research boundaries](docs/research-boundaries.md).

[First measured results](docs/first-neural-results.md): synthetic recall passed, ordinary training beat the evolutionary pilot, and ternary-aware adaptation recovered much of the quantization loss. General conversational ability and low-power hardware performance remain unestablished.

The [dictionary relation graph](docs/relations.md) adds WordNet meanings, typed relations, connectivity rankings, graph exports, and a separate supervised relation-training corpus. Querying the dictionary and testing what the neural model learned are separate commands.

The [chat-evolution experiment](docs/chat-evolution.md) trains original short conversations with ordinary, independent and genetic controls. It adds held-out paraphrase families, context-use checks and bounded cluster workers. The initial language checkpoint scored 0.0 on the 56-case chat validation diagnostic; final results for this new stage are pending. Live replies come from a saved neural checkpoint, not an answer lookup.

The newer [conversation-world experiment](docs/conversation-world.md) targets false-name claims, irrelevant replies and multi-turn error recovery. It combines a learned 18-class meaning head with explicit USER-only fact preprocessing and evaluates actual generated histories. The final allocation is **32 islands /160 model slots**, with the extended run planned through **12 October 2026, 10:00 CEST**; no final quality result is claimed yet.

The optional [DNA-3 program bridge](english_world/program_bridge.py) extracts one explicitly bounded increment-modulo-four compute slice from a frozen observed genome. It verifies 204 answers against the original reference VM and emits deterministic Python. This is not a general genotype compiler. The [paired curriculum pilot](ops/program_bridge_pilot.py) tests whether verified program examples improve conversational validation and held-out program phrasing before any long-run adoption; it never modifies active training or evaluates the final test partition automatically.

## Run without installing anything

Python 3.11 or newer is sufficient; the CPU implementation has no third-party runtime dependencies.

```bash
git clone https://github.com/enderPeer/Golem.git
cd Golem
python3 -m english_world doctor
python3 -m english_world prepare --source fixture --output data/fixture
python3 -m english_world evolve --config configs/smoke.toml --data data/fixture/dataset.json --run-dir runs/smoke
python3 -m english_world chat --checkpoint runs/smoke/champion.json
```

The smoke run checks the pipeline on original tiny fixtures. It is not evidence of learned English.

For public English data and a bounded pilot:

```bash
python3 -m english_world prepare --source english --output data/english
python3 -m english_world benchmark --data data/english/dataset.json --run-dir runs/benchmark
python3 -m english_world evolve --config configs/pilot.toml --data data/english/dataset.json --run-dir runs/pilot
python3 -m english_world status --run-dir runs/pilot
python3 -m english_world chat --checkpoint runs/pilot/champion.json
```

Use `--resume` with `evolve` to continue a checkpoint. Increase `--generations` to raise the total target. Run `./golem --help` for commands. The `golem` shell launcher is an equivalent shortcut on Linux; `english-world` remains an alias. The internal Python package and checkpoint schemas retain their versioned `english_world` names.

Chat commands: `/info`, `/reset`, `/reload`, `/quit`. `/reload` picks up a newly saved champion. This version continues the actual end of your input; its effective context is at most five characters. It is not an instruction-following assistant. Chat histories exist only in the terminal process and are not used for training or saved automatically.

## What evolves

Each composite genome contains a replication/control program, a bounded context-selection rule, and sparse integer next-character weights. The first release keeps the reproduction instruction sequence fixed while evolving predictor rules and bounded predictor settings. The character distribution uses matching suffix contexts with smoothing/backoff. All predictor changes are explicit genomic mutation events; parent selection favors useful prediction within resource constraints.

Some mutation proposals are **corpus-guided**: a training example proposes a context/next-character association. Others change or remove rules without using the answer. This is supervised, data-informed evolutionary search, not spontaneous language emergence from random letters. There is no hidden full-corpus n-gram fitting step in the evolutionary model. Statistical baselines, when reported, are labeled separately.

Correct spelling receives a capped auxiliary reward. The `spelling_score` is the number of distinct accepted complete tokens divided by all complete tokens; it is a diversity-weighted dictionary acceptance score, not an ordinary spelling-accuracy percentage. Only `a` and `i` qualify as single-letter words. Invalid mixed tokens remain in the denominator. Prediction of unseen continuation is measured separately. Repeated dictionary words must not acquire unlimited reward. The dictionary is available to the scorer, not as an output lookup instruction.

The first model has short effective context. Although the terminal accepts dialogue history, it cannot infer long-range reasoning abilities from that interface. Generated output is displayed without replacing it with dictionary words or a pretrained response.

## Predict its own code

Version 0.2 adds a distinct code-domain experiment using a pinned snapshot of Golem's own tracked Python files. It preserves case, punctuation and indentation, evolves inherited predictor rules, and selects champions on held-out files. Its own-genome prediction score is a diagnostic, not a fitness shortcut. Existing English checkpoints retain their original v1 representation.

```bash
./golem prepare --source self-code --output data/self-code
bash ops/run_pilot.sh configs/self-code.toml data/self-code/dataset.json runs/self-code
./golem self-check --checkpoint runs/self-code/champion.json
./golem chat --checkpoint runs/self-code/champion.json
```

This improves the model genome through selection. It does not rewrite or execute the host Python implementation. See [the self-code experiment](docs/self-code.md) for the exact boundary and evaluation rules.

## Relationship to DNA-2

The original reference and its MIT license are frozen under `upstream/dna2/`, with source provenance. The original repository is not modified.

English-world v1 adds a **composite heritable record**. The entire reproduction program and predictor payload are copied and hashed together. Its chunked copy interpreter carries over three DNA-2 ideas: COPY waits for a claim, TERM follows the complete terminator, and DIVIDE rearms the parent. This is a new explicitly specified world rule; it does not claim bit-for-bit equivalence to the original slot simulator. See [the specification](docs/world.md).

## Measurement

- Training examples drive mutation proposals and reproductive rewards.
- Validation examples select saved champions.
- The final test split is reserved for reporting after selection.
- Predictive loss, bits per character, generated-word validity, repetition, ancestry, and resource costs are distinct measurements.
- Compare against random search and simple statistical predictors before claiming that evolution improved the loss/cost tradeoff.
- `configs/random-parent-control.toml` supplies a random-parent ablation without elites; energy eligibility still applies. Count-fitted baselines in `evaluate --baselines` cap integer weights at 255 and are labeled controls.
- A bounded pilot may find no improvement. Its actual results must remain visible.

## Data and publication

Dataset preparation records sources, revisions, digests, normalization, splits, and license notices. Original small fixtures are included for offline testing. Downloaded corpora, run checkpoints, caches, and interactive prompts are not committed by default. Their licenses remain distinct from the source-code license. See [data provenance](docs/data.md).

## Shared-machine execution

`ops/preflight.sh` and `ops/run_pilot.sh` support a bounded CPU run in an isolated user service. The initial policy is four CPU threads, low priority, 8 GiB RAM, a 20 GiB project budget, and a maximum 30-minute run. No driver changes or system-wide Python installation is required.

The initial implementation uses CPU execution. A Vulkan implementation is a future optimization gated on a measured bottleneck and reference-equivalence tests; no GPU training support is claimed in this version.

A continued English experiment can use `configs/language-hour.toml` and `ops/run_language_hour.sh`. It resumes a checkpoint already placed in `runs/language-1h`, trains for one additional hour, and reports session elapsed/remaining time separately from cumulative training time. Its model settings match the initial English pilot. The supervisor allows a final minute for checkpoint shutdown and retains the CPU, memory and disk limits.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Source code is MIT licensed. Retain the upstream attribution in redistributed copies.
