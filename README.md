# aLIFE English

A separate evolutionary English world, with reproducible experiments and a terminal interface to a saved character predictor.

**Status: experimental.** This project evolves small character prediction programs and integer context tables. It is not a transformer-scale LLM, and a working chat command does not establish conversational understanding. No pretrained language model, hosted model, or gradient training supplies its responses.

## Run without installing anything

Python 3.11 or newer is sufficient; the CPU implementation has no third-party runtime dependencies.

```bash
git clone https://github.com/enderPeer/alife-english.git
cd alife-english
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

Use `--resume` with `evolve` to continue a checkpoint. Increase `--generations` to raise the total target. Run `python3 -m english_world --help` for commands. The `english-world` shell launcher is an equivalent shortcut on Linux.

Chat commands: `/info`, `/reset`, `/reload`, `/quit`. `/reload` picks up a newly saved champion. This version continues the actual end of your input; its effective context is at most five characters. It is not an instruction-following assistant. Chat histories exist only in the terminal process and are not used for training or saved automatically.

## What evolves

Each composite genome contains a replication/control program, a bounded context-selection rule, and sparse integer next-character weights. The first release keeps the reproduction instruction sequence fixed while evolving predictor rules and bounded predictor settings. The character distribution uses matching suffix contexts with smoothing/backoff. All predictor changes are explicit genomic mutation events; parent selection favors useful prediction within resource constraints.

Some mutation proposals are **corpus-guided**: a training example proposes a context/next-character association. Others change or remove rules without using the answer. This is supervised, data-informed evolutionary search, not spontaneous language emergence from random letters. There is no hidden full-corpus n-gram fitting step in the evolutionary model. Statistical baselines, when reported, are labeled separately.

Correct spelling receives a capped auxiliary reward. The `spelling_score` is the number of distinct accepted complete tokens divided by all complete tokens; it is a diversity-weighted dictionary acceptance score, not an ordinary spelling-accuracy percentage. Only `a` and `i` qualify as single-letter words. Invalid mixed tokens remain in the denominator. Prediction of unseen continuation is measured separately. Repeated dictionary words must not acquire unlimited reward. The dictionary is available to the scorer, not as an output lookup instruction.

The first model has short effective context. Although the terminal accepts dialogue history, it cannot infer long-range reasoning abilities from that interface. Generated output is displayed without replacing it with dictionary words or a pretrained response.

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

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Source code is MIT licensed. Retain the upstream attribution in redistributed copies.
