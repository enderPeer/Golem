# Evolving a small neural chat model

This experiment trains Golem's byte-level GRU to answer a narrow set of everyday
messages, including greetings, identity questions and short follow-ups. Replies
come from the neural checkpoint. The terminal does not look up a canned answer,
call a hosted model or replace generated output with a training reference.

The experiment is supervised hybrid learning: gradient updates train each
candidate, while genetic operators change selected weights and its training
recipe. This is a separate backend from the original DNA-2 interpreter and the
mutation-only context-table model.

## Original chat curriculum

`chat_corpus.py` authors 12 intents: greeting, identity, wellbeing, capabilities,
thanks, goodbye, help, uncertainty, clarification, preferences, encouragement
and name recall. Responses identify Golem as an experimental program and teach
limits for unsupported questions. Wellbeing and preference examples avoid
inventing human feelings or personal experiences.

| Partition | Cases | Paraphrase families | Longest complete byte/role sequence |
|---|---:|---:|---:|
| Training | 643 | 72 | 97 |
| Validation | 56 | 24 | 104 |
| Test | 56 | 24 | 125 |

Families are assigned before any augmentation. Case, punctuation, collapsed
spacing and bounded single-transposition variants are created only from training
prompts. Case/spacing/punctuation-equivalent complete inputs cannot cross
families or partitions. Validation and test contain independently authored
paraphrase families, not held-out spellings of the same training prompt.

The corpus includes real multi-turn sequences: a greeting followed by an identity
or help question, and a user introducing a name before asking for it later.
Counterfactual name pairs share the same final question but give different names
earlier in the conversation. Names are balanced and shared across partitions;
this tests use of context rather than the spelling of an unseen name.

Original conversations, case rubrics, family assignments and a source snapshot
are saved with hashes and the project's MIT notice. There are no external
downloads in this preparation step. The current corpus content hash is
`a37f2d7a29ea1acd4e25be469d9afc2edcf06ec8a7659b4f77be97924717c763`.

Sequence length 128 retains every complete example in the current corpus.
Assistant targets and END tokens receive loss; user text is context, not an
answer target. The trainer rejects a configured window that would crop a
required conversation. Terminal history uses explicit user/assistant role tokens.

## What is compared inside a worker

Each worker, called an island in the launch manifest, runs three strategies:

| Strategy | Learners | Updates and selection |
|---|---:|---|
| Ordinary training | 1 | A single learner receives the strategy's update budget; validation selects its checkpoint. |
| Independent learners | 4 | The budget is divided among four learners, with no inheritance or crossover; validation selects the best. |
| Evolutionary population | 4 | Learners train, are evaluated, and reproduce through bounded genetic operators. |

The controls within one worker share the frozen initializer, architecture,
configured optimizer-update count, batch shapes and evaluation-request schedule.
They do not receive equal total CPU time, GPU work or generated-token counts.
An evolved curriculum can also change the number of supervised assistant
targets. Actual counts, timings, initialization provenance and completion status
are recorded. Prior training of the frozen initializer is outside this stage's
update budget and is disclosed.

Workers differ in seeds, initial Golem checkpoint, batch sizes, learning rates
and mutation settings. CPU workers use 2,400 updates per strategy and batch size
8; the GPU configurations use 800 updates and batch size 32. These are not
equal-compute comparisons across hardware or across islands. The workers do not
exchange candidates with one another. Any island-wide comparison must retain
those differences instead of attributing all improvement to evolution.

## Genetic operators

Evolution ranks candidates by generated validation rubric score, using lower
assistant prediction loss to break ties. The leading candidates are protected;
lower-ranked slots receive descendants. A descendant can:

- Inherit the parent's full model and optimizer state.
- Mutate bounded learning rate, dropout, weight decay or gradient clipping.
- Change a training-intent sampling weight within its declared limits.
- Receive a complete embedding, recurrent or output module from another
  protected candidate.
- Receive bounded, RMS-scaled noise on a selected weight matrix.

Optimizer state is reset for parameters changed by module crossover or weight
noise. Parent and child IDs, operator records, tensor digests, gene digests and
the preserved elite are recorded. Tests cover bounds, parent preservation and
CPU checkpoint replay. These operators modify model tensors and training
settings; they do not execute generated code or rewrite Golem's Python sources.

## What the evaluation means

The evaluator generates replies before consulting its reference-answer loss.
Rubrics check transparent intent keywords, forbidden phrases, length, nonempty
text and repetition. Phrase boundaries prevent matches such as `hi` inside
`this`. Name cases also reject the other configured names. Generated replies
and individual checks remain visible in the report.

These are **narrow heuristics, not semantic quality or human accuracy**. A reply
can satisfy keywords while remaining awkward or unhelpful. Reference-answer
likelihood is another diagnostic, not a substitute for inspecting actual
generation. Counterfactual pairs provide evidence about context use only when
the model changes its response appropriately for both inputs.

Sweep configurations evaluate 26 validation cases per selection point, covering
all 12 intents while retaining complete name pairs. Validation selects candidates
and islands. Test cases are for final reporting after training and selection;
they must not be used to choose a live checkpoint.

Before this chat stage, the frozen **language initializer** scored **0.0 on all
56 validation cases**, with heuristic pass rate 0.0. This is recorded in
`runs/chat-initial-validation.json`; it is not a result for every initializer or
an assertion that the model has no language ability. The current run has no
final chat-quality result yet.

## Recorded cluster deployment, 2026-10-10

Launches began between **23:09 and 23:11 CEST**. The reviewed allocation is:

| Host | CPU workers × threads | GPU workers | Total requested host CPU threads |
|---|---:|---|---:|
| specht32 | 4 × 2 | Two per admitted RX 9060 / RX 9070 device: 4 workers, one host thread each | 12 |
| falke64 | 4 × 2 | Two on the admitted free R9700, one host thread each | 10 |
| adler40 | 10 × 2 | None | 20 |
| knecht24 | 9 × 2 | None | 18 |
| Total | 27 CPU workers | 6 GPU workers on 3 devices | 60 |

All 33 units were subsequently observed running, with no failed sweep workers
at the check. All three admitted GPUs passed a full eight-update-per-strategy
pipeline test including genetic births. Falke's borrowed runtime failed a
dropout kernel compilation because a header was missing; a separate Golem
`.venv-rocm` with the pinned ROCm/PyTorch stack passed that test. Its existing
runtime was left intact. This confirms compatibility, not completed training.
Falke's GPU jobs evaluate the evolutionary strategy first to provide early
genetic feedback; the other two controls retain their complete declared budgets.

Excluded devices remain excluded: Falke's other occupied R9700, Adler's GPUs
already occupied by DNA/loaded chat work, and all Knecht GPUs running DNA. CPU
workers on Falke, Adler and Knecht borrow an existing Python runtime read-only,
with bytecode writing disabled and Golem's own cache directory. Its package files
and the other project's model files are not modified. GPU runtime setup belongs
to a separate Golem environment.

Each worker runs at nice 19, with an explicit CPU quota and memory limit. CPU
limits are 2 GiB per worker, reduced to 1.5 GiB on Falke; GPU workers have 3 GiB
host-memory limits. The trainer's GPU allocator limit is separate and does not
measure or cap every driver allocation. The launcher checks aggregate planned
memory against available host memory and retains a 2 GiB reserve.

The training deadline is 3,300 seconds, or 55 minutes, per worker session. The
systemd unit ceiling is 3,630 seconds, or 60 minutes 30 seconds, allowing bounded
startup/shutdown overhead. These are ceilings, not a promised amount of useful
training or guaranteed completion of all strategies. Low-priority scheduling
does not make total execution time equal across hosts.

Persistent deployments use `/home/ender/golem` and the 50 GiB free-disk reserve.
Knecht uses the explicitly temporary `/tmp/golem-chat-20261010` checkout because
its persistent disk did not meet that reserve. Its tmpfs uses host RAM; its free
capacity is not an additional independent memory pool. The temporary deployment
uses a 2 GiB free-filesystem reserve. All deployments retain the 20 GiB project
limit and periodically check host memory. These periodic checks are reactive;
the per-unit limits and reviewed allocation remain necessary.

**Preserve Knecht's reports, manifests and selected checkpoints to persistent
storage before cleaning the temporary checkout or rebooting.** No cleanup or
preservation success is implied by merely launching a worker.

## Prepare, launch and inspect

Prepare the original corpus locally:

```bash
python -m english_world neural chat-prepare --output data/chat --seed 61
```

Materialize an immutable plan in a new directory. This command launches nothing:

```bash
python ops/make_chat_sweep.py --output runs/chat-cluster-new
```

The script records the inspected 2026-10-10 allocation. Recheck current workloads,
available memory and device identities before reusing it. Deploy the reviewed
project snapshot, identical corpus, frozen initializers, host configuration files
under `runs/chat-configs/`, and the matching manifest as `runs/chat-manifest.json`.
Then run on the selected host:

```bash
python3 ops/launch_chat_sweep.py --manifest runs/chat-manifest.json --kind cpu
python3 ops/launch_chat_sweep.py --manifest runs/chat-manifest.json --kind gpu
```

The GPU command is appropriate only for the admitted, verified devices. The
launcher checks host/project identity, CPU count and aggregate memory. It skips
jobs with an existing launch receipt; a receipt alone does not prove the worker
succeeded. The per-job systemd unit is recorded under `runs/launches/`, and each
worker saves status, reports, checkpoints, population genes and genetic events in
its own `runs/chat-sweep/<job-id>/` directory. Resuming a failed or interrupted
experiment requires deliberate review of its recorded status and inputs.

For an individual controlled experiment, use a new run directory:

```bash
./golem neural chat-train --config configs/neural-chat.toml --data data/chat/dataset.json --run-dir runs/chat-single
./golem neural status --run-dir runs/chat-single
./golem neural chat-eval --checkpoint runs/chat-single/best.pt --data data/chat/dataset.json --split validation --cases 56 --output runs/chat-single/validation-full.json
```

Direct CLI training uses its configured bounds; the shared-machine cgroup and
storage protections are supplied by `ops/run_chat.sh` and `ops/chat_guard.py`.
Use the guarded launcher for shared deployments. A time-limited report is not a
completed matched comparison, even if it contains a useful saved candidate.

## Live terminal and provenance

The promoted live artifact is `runs/chat-live/champion.pt` on specht32. Select it
using validation results and retain the source job, checkpoint digest, corpus
digest and evaluation record. Its promotion is separate from training and from
final test reporting.

`ops/collect_chat_sweep.py` collects verified snapshots from the recorded hosts.
Global ranking uses the same 26 greedy validation cases and recomputes missing
reference losses on those cases, preserving original checkpoint bytes and hashes.
Before publication, `ops/chat_release_gate.py` requires six known-request checks
and a three-turn greeting/identity/wellbeing exchange to pass. These known-task
checks establish minimum usability, not held-out semantic accuracy. A candidate
that scores better on approximate validation keywords can still fail this gate.

```powershell
python ops/collect_chat_sweep.py --manifest runs/chat-cluster-20261010/manifest.json --output runs/chat-collector --publish-host specht32
```

Reports distinguish the raw validation leader from `publication_selected`.
Publication checks the source hash, uses a remote lock and atomic replacement,
and keeps the incumbent when no eligible improvement is found. The first live
checkpoint came from `adler40-cpu-5`, ordinary strategy at update 825, and passed
all seven direct checks. Its SHA-256 is
`12c09b5fd3bda06b2548b8477de3ef5fca243be287686aecc59947b81506df7c`.
This selection does not establish an evolutionary advantage; those comparisons
continue separately with their declared controls.

From Windows, open the remote terminal with:

```powershell
& .\ops\Open-GolemChat.ps1
```

The wrapper creates a new console with its own keyboard/display handles and
connects to the saved neural checkpoint with greedy decoding. `/reload`
loads a newly promoted artifact and clears history; `/reset` clears history
without changing weights. The live endpoint is specht32; other hosts do not
automatically receive a promoted checkpoint.

Previous repositories and experiment reports are reference material, inspected
read-only. Borrowing a runtime or checking an old function-search result is not
equivalent to importing another project's intelligence. A library of small VM
functions is not a neural chat model or a drop-in implementation of its GRU.
Source snapshots, artifact digests and explicit evaluation records must support
each claim of improvement.
