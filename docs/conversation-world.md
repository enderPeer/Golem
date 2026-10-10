# Conversation worlds: meaning, grounded facts and error recovery

This stage addresses the observed failures where Golem answered a greeting with
an unrelated help prompt, repeated one response across topics, or assigned the
user the name Noah without any user introduction. The new training environment
makes those failures explicit and measurable. It does not claim that training
has already eliminated them.

## Model and meaning labels

The default model has **800,535 parameters**: the existing 791,300-parameter byte
GRU plus a learned meaning classifier, learned meaning embeddings and a learned
conditioning gate. The classifier predicts 18 communicative meanings:

`greeting`, `presence`, `identity`, `request_help`, `clarify`, `rephrase`,
`correction`, `loneliness`, `empathy`, `thanks`, `goodbye`, `preferences`,
`name_set`, `name_recall_unknown`, `name_recall_known`, `name_correction`,
`topic_chat`, and `uncertainty`.

Each label has an explicitly authored alignment to verified WordNet 3.0 senses.
For example, greeting uses `n:06630017`, identity uses `n:05763412`, loneliness
uses `n:07534108`, empathy uses `n:07554856`, and names use `n:06333653`.
The original definitions and source hashes are saved in the meaning-graph
snapshot. The dialogue-act edges, including `loneliness -> calls_for -> empathy`,
are **human-authored mappings**, not automatically discovered WordNet facts.

The meaning label describes the latest **user** message. The desired response
act can differ: expressing loneliness calls for an empathetic response. An
assistant identity question and a user-name question have different subjects.
These 18 labels cover a small conversational domain, not unrestricted semantic
understanding.

The decoder receives a soft mixture of its own predicted meaning embeddings.
It uses this predicted distribution during both training and generation. There
is no inference interface that supplies the correct meaning label or retrieves
the reference reply. WordNet is used to document the authored labels, not to
replace a generated answer.

## Explicit user facts are separate from neural memory

`dialogue_state.py` records a name only from a supported explicit **USER**
assertion, such as `My name is Mira`, `Call me Mira`, `I am called Mira`, or
`I go by Mira`. An explicit correction replaces the previous name, and a
supported request to forget it clears the slot. Names are bounded single-word
labels; this is not a general personal-information extractor.

Assistant messages never establish a user fact. An assistant saying `Your name
is Noah` therefore leaves the name unknown when the user has provided none.
Questions such as `it is?` do not introduce a name either. Every saved training
frame is checked against the same USER-only extractor used during inference.

The resulting fact marker is placed near the current request in the model's
input, for example `[verified user name: UNKNOWN]`. This is visible symbolic
preprocessing. It must not be reported as learned neural memory. The network
still has to produce a correct reply, and it can still ignore this evidence or
make an incorrect claim; the free-rollout evaluator checks that separately.

The terminal retains the supplied role history in the current process. If model
context needs trimming, it removes complete old turns and derives verified facts
from the complete supplied history first. `/reset` and `/reload` clear the
conversation and name state. Interactive messages are not automatically training
data and are not saved by the chat command.

## Original simulated conversations

The frozen corpus contains normal short conversations, topic changes, short
follow-ups, rephrasing, explicit name corrections, unknown-name questions and
recovery from deliberately incorrect assistant history. Ordinary episodes have
six or eight role messages. Explicitly injected bad assistant turns are marked
as untrusted context and never treated as authoritative facts.

| Partition | Episodes | Current-reply training/evaluation targets | Scenario families | Distinct user-provided names |
|---|---:|---:|---:|---:|
| Training | 867 | 3,082 | 59 | 128 |
| Validation | 72 | 256 | 28 | 4 |
| Test | 72 | 256 | 28 | 8 |

Entire scenario/paraphrase families are partitioned before names, topics and
reference-response variants are introduced. All branches of a contrast remain
in the same partition. Explicitly introduced validation and test names do not
appear as training-name labels. Unsupported assistant claims may repeat familiar
names on purpose, so the learner must distinguish role and evidence rather than
accepting any name it sees.

The 128 training labels include deterministic, pronounceable name-like strings.
All 128 are actually used. This reduces the usefulness of memorizing a tiny name
list, although held-out performance must still demonstrate successful copying
and grounding.

Two kinds of hard contrast are included:

- The same prior conversation followed by different latest requests: assistant
  identity, user-name recall, or an expression of loneliness.
- The same final name question with no user-provided name, one explicitly given
  name, or a different explicitly given name.

Known and unknown states remain independent between episodes. A wrong name
emitted by one simulated assistant cannot become a fact for another world.

Normal reference conversations are capped at 448 role tokens, leaving room for
fact preprocessing within a 512-token model input. Measured maximum encoded
prefix lengths, including verified facts, are **323 / 331 / 367 tokens** for
training/validation/test. The longest current reply plus END is **63 tokens**.
No normal training history needs silent cropping.

The prepared dataset is `data/conversation-world/dataset.json`, 43,688,244 bytes.
Its canonical content SHA-256 is
`e4215a3b8e4428c867c436dda27c12609fbb4619f930c3f3304ecbb834eb47d9`;
the file-byte SHA-256 is
`478b61e93933bf67313893fb5ae6478fdf218a9949734c9ae125a5a8606c8c1a`.
Original examples retain the project MIT notice, and WordNet anchor data retains
its separate license. No external model provider supplies these examples or
answers.

## Why the training objective changed

Only the current final assistant reply is a language-model target. Previous
assistant messages, including injected incorrect claims, are conditioning
context only. The learner is never trained to reproduce a bad historical name
claim as a language target.

The loss combines current-reply prediction with supervised classification of
the latest user meaning. The initial bytes of the **current reply** receive
extra weight. This gives additional pressure to begin with an appropriate
response, rather than starting an unrelated answer and continuing it fluently.
The term “first-response weight” does not mean loss on the input/history prefix.

Population recipes can change learning rate, dropout, meaning-loss weight,
first-response weight and the training-act sampling mixture within explicit
bounds. Exploit-and-mutate events within a stage inherit model and optimizer
state. A new supervised stage warm-starts model weights from the island's best
validation-selected checkpoint and **resets the optimizer**; it is not an exact
continuation of the preceding optimizer trajectory. Flat parent metadata records
the source checkpoint hash without nesting the complete history in every file.
The ordinary learner is a control using the same configured update budget within
that stage. Token lengths, rollout lengths, population management and device costs
can differ, so total CPU/GPU work is measured rather than claimed equal.

Validation selection gives priority to zero **observed** false-memory claims,
then generated conversation scores and whole-episode success, with current-reply
loss as a later comparison. A correct meaning classification alone is not enough:
the decoder may still emit an incorrect name or an unhelpful sentence.

## Free rollouts and the exact regression

During free evaluation, each next user message is combined with the assistant's
**actual generated previous replies**. Reference assistant responses are not
replayed as history. The only prefilled assistant messages are explicitly marked
adversarial prefixes, injected once to test recovery from a wrong claim.

The evaluator records actual histories, predicted meanings, request relevance,
repetition, explicit name claims and their USER evidence. Complete counterfactual
groups stay together. These are transparent scenario checks, not a complete
semantic judge or a detector of every possible hallucination. Passing them does
not guarantee safe or correct behavior for arbitrary conversations.

Three short repair probes are included in training and separately identified as
known regressions. A fourth regression preserves the user's **exact seven
requests** as one complete free-running conversation:

1. `greet me`
2. `i ask you to greet me`
3. `what do i make of you`
4. `it is?`
5. `ok. i am lonely`
6. `are you there`
7. `who are you`

No user name is supplied anywhere in that sequence. Its reference transcript has
14 role messages and 443 tokens. The complete trace is regression-only, not a
held-out test and not a cropped training example. If actual generation makes its
history longer, the model may trim complete old turns to its input bound while
the evaluation log retains the entire interaction and USER fact history. A good
score on three shorter probes must never be substituted for passing this trace.

## Extended run plan and deadline

The requested cutoff is **Monday, 12 October 2026 at 10:00 Europe/Berlin
(10:00 CEST), equivalent to 08:00 UTC**. Work must stop at that absolute deadline
or an earlier explicit stop. A resource or device failure is a recorded failure,
not a reason to extend the deadline or use an occupied device.

The final reviewed allocation has **32 islands and 160 candidate slots per
stage**: 26 CPU workers and six GPU workers, each with one ordinary learner plus
four evolutionary learners. This does not mean that all five candidates perform
gradient updates simultaneously. The aggregate host CPU quotas allow **61 logical
threads**; this is a scheduling allowance, not a measured utilization figure.

| Host | CPU workers × threads | GPU workers × host threads | Total thread quota |
|---|---:|---:|---:|
| specht32 | 4 × 2 | 4 × 1: two workers per admitted RX 9060/RX 9070 device | 12 |
| falke64 | 3 × 3 | 2 × 1 on the admitted free R9700 | 11 |
| adler40 | 10 × 2 | 0 | 20 |
| knecht24 | 9 × 2 | 0 | 18 |

GPU workers have a **4,096 MiB host `MemoryMax`**, following measured probe RSS
of 2.35–2.93 GiB. CPU workers have 2,048 MiB limits, except Falke's three workers
at 1,536 MiB each; its CPU probe reached 1.265 GiB RSS. These are host-memory
limits, separate from GPU device-memory limits. Falke's reduced CPU-worker count
preserves memory headroom alongside the two GPU workers.

Borrowed CPU runtimes on Falke, Adler and Knecht remain read-only, with Python
bytecode writing disabled and caches directed into the dedicated Golem project.
Existing other workloads are left untouched. Occupied GPUs remain excluded, and
workers run at Nice 19 under their explicit CPU quotas. Availability and actual
utilization must be checked from the run records; assigning a worker to a device
does not prove continuous 100% GPU utilization. The extended run repeatedly
schedules bounded stages on the admitted devices while time and resource limits
permit.

A finite `world-train` invocation or `run_world.sh` unit is only one bounded
stage. The extended supervisor is responsible for subsequent stages, absolute
deadline checks, result retention and stopping. Stage completion must not be
confused with completion of the entire requested run.

The training loop has two separate review paths:

- Assistant/teacher review inspects validation outputs, source-grounding failures,
  resource state and logs. It can recommend a documented correction or stop; it
  must not tune against held-out test results.
- TRAIN-only self-repair probes the current model on training scenarios. A
  generated prior reply that differs from the reference can become new context,
  paired with the existing verified TRAIN target reply and meaning. This includes
  valid alternative phrasing as well as errors; the collection rule is not a
  claim that every collected history failed a quality test. Generated errors
  never become trusted answer targets. Validation and test examples remain
  unchanged, and known-regression episodes are excluded from repair sampling.

Repair data belongs to a new stage with a new data identity; it cannot be silently
inserted into an exact-resume checkpoint. The repair overlay is intended to avoid
copying the whole 43 MB corpus for every stage. `world_replay.py` saves a small
overlay referencing the frozen base and source checkpoint hashes. It checks exact
USER turns, original TRAIN targets, current meaning labels and USER-only facts
before adding samples in memory. It does not stack preceding overlays. The
default stage probe covers at most 32 training episodes and adds at most 256
samples; cancellation publishes no partial overlay.

Full held-out model testing is reserved for the final selected experiment at the
end of the run. The supervisor forces `evaluate_test_on_complete = false` in each
stage configuration, so the trainer does not prepare or evaluate test batches.
The supervisor does not run final testing automatically; the final assessment is
a separate operation after the deadline and validation-based checkpoint
selection. Regression probes are known failures, not a substitute for the final
held-out assessment.
The final report must say whether the deadline, a budget, an interruption or a
completed plan ended the run.

`ops/world_supervisor.py` runs one island as a sequence of stages of at most
1,800 seconds. It fixes validation rollout seed 40097, teacher-forced seed 30097
and batch size 8 unless explicitly configured otherwise. A recorded protocol hash
covers the validation data and scoring settings; a stage with a changed protocol
cannot replace that island's existing champion. Cross-island comparison still
requires matching protocol hashes and documented evaluation settings.

The initial stage budget is eight rounds with **1,600 updates per strategy on
CPU** or **800 per strategy on GPU**, inside the 1,800-second stage limit.
The initial deployment executed strategies sequentially. Its order alternated across islands so that
the whole allocation does not always give the ordinary learner first access to
the time budget. This reduces allocation bias but does not make an unfinished
stage a matched comparison.

Review the first stage's actual updates, validation calls and recorded cost for
each strategy alongside the unit's elapsed time. Full conversation rollouts can
consume substantial time; a short startup canary does not measure that cost.
If the earlier strategy repeatedly consumes most of the stage, fresh warm-starts
can repeatedly leave the second strategy unfinished. Such stages are useful
exploration, but must not support a claim of an advantage at matched update
budgets.

A reviewed future configuration could use **four rounds and CPU 800 / GPU 400
updates per strategy**. This retains the current 50 / 25 updates per candidate
block while halving both training work and the number of validation calls.
Halving updates alone would leave validation overhead unchanged. Make any such
adjustment through a reviewed fresh supervisor run; do not edit an active run's
pinned base configuration. Those original stage records remain intact. The first
hourly review found the second strategy at zero updates on slower hosts. New
stages now default to `strategy_schedule = "round_robin"`: each strategy gets one
complete training and validation block, then yields to the next, preserving its
weights, optimizer and random state. Legacy checkpoint resumes retain their
original sequential schedule unless a new experiment is explicitly created.
This prevents repeated starvation; it does not establish identical hardware work.

### First hourly teaching experiment

The 11 October 01:00 Berlin review found good meaning classification but poor
generated replies in a frozen Falke candidate. On 64 strictly non-regression
TRAIN episodes, meaning accuracy was 93.89%, while no complete conversation
passed the authored reply checks. A separate known-regression trace also failed.
No held-out test was evaluated and this candidate was not released to live chat.

On two failed TRAIN examples, the learned meaning contribution was only
0.20–0.28% of the encoder state norm. Removing it left both wrong greedy replies
unchanged. Teacher-forced and incremental logits agreed to about 1e-5, so this
probe found weak conditioning rather than a target-shift error.

The isolated `falke64-world-teacher-1` experiment uses an explicit
`meaning_condition_gain = 32.0`, saved in its checkpoints and reused at inference.
Existing models retain gain 1.0, including legacy checkpoints without the field.
The gain is bounded to 1..64 and is an experimental intervention, not a measured
improvement. Its recipe weights the first eight reply bytes by 6.0 and keeps
meaning-loss weight at 0.5.

Its additional immutable teacher overlay is
`data/conversation-world/teacher-repairs-20261011T0105.json`, SHA-256
`34d2209474192f66ac364ba0b7c8eac28bc8e6d3a897e97d2efd53f9adfac534`.
It contains 89 corrected contexts from 53 original TRAIN episodes. All targets
are original verified TRAIN references; generated replies are untrusted history.
`ops/build_teacher_repairs.py` selects cases with a correct predicted meaning and
a failed reply, rejects changed USER facts and held-out/regression episodes,
and validates the result with the same replay loader. At most two overlays can
be combined against the pristine base; their separate hashes and source models
are recorded, and duplicate sample IDs are included once. The original corpus,
validation and test partitions remain fixed.

The supervisor starts helpers and training in owned process groups and never
signals unrelated jobs. It has a 15-second shutdown reserve before the absolute
UTC deadline. The actual `run_world_long.sh` launcher also sets an independent
systemd runtime limit that begins shutdown 30 seconds before the deadline, with
at most 20 seconds for stopping the owned control group. CPU-only inspection,
warm-start preparation and repair helpers run before the trainer, not concurrently
with it. Memory, free storage and both per-island and project storage limits can
stop a worker early. Three consecutive failed stages stop that island instead of
retrying indefinitely.

The run root contains `supervisor-state.json`, a compact `status.json`, and the
validation-selected `global_best.pt` with its hash and provenance in
`global_best.json`. Each numbered stage retains its configuration and summary.
Older reports, status/population records and trainer logs are archived losslessly
with gzip and verified against their uncompressed SHA-256. Cleanup removes only
recognized tensor artifacts in owned stage directories, preserves the latest two
actual resumable states, and leaves the island's global champion intact. A final
stage candidate that cannot be inspected before the shutdown reserve is recorded
as pending and retained for post-deadline collection.

## Prepare and inspect

The commands below use the current model/corpus interfaces. A direct `world-train`
run is finite and, unless its configuration disables it, evaluates test data when
complete. Use the supervisor for the extended run and deferred-test policy.

```bash
./golem neural world-prepare --graph data/lexicon/graph.json --output data/conversation-world --seed 97
./golem neural world-train --config configs/neural-world.toml --data data/conversation-world/dataset.json --run-dir runs/conversation-world
./golem neural status --run-dir runs/conversation-world
./golem neural world-eval --checkpoint runs/conversation-world/best.pt --data data/conversation-world/dataset.json --split validation --episodes 24 --output runs/conversation-world/validation-rollout.json
./golem neural world-eval --checkpoint runs/conversation-world/best.pt --data data/conversation-world/dataset.json --split regression --episodes 4 --output runs/conversation-world/regression-rollout.json
./golem neural world-chat --checkpoint runs/conversation-world/best.pt
```

The allocation generator writes configs and a manifest without launching jobs:

```bash
python -B ops/make_world_sweep.py --output runs/world-long-allocation-example --deadline 2026-10-12T08:00:00Z
```

Use a fresh output directory. For each reviewed host, stage that host's generated
TOML files at the manifest's `config` paths under `runs/world-long-configs/`, and
place its reviewed manifest in the dedicated project. Then, on that host:

```bash
python -B ops/launch_chat_sweep.py --manifest runs/world-long-manifest.json --kind all
```

The launch command selects only the local hostname's jobs. It checks the project
root, aggregate thread count and memory headroom, and uses receipt files to avoid
launching the same job twice. Its name is shared with the earlier chat experiment;
the world manifest's `supervised_until_deadline` mode selects `run_world_long.sh`.
That shell launcher supplies `GOLEM_PYTHON`, device visibility, thread quotas,
memory limits and the absolute deadline to the supervisor. Persistent roots use
a 50 GiB free-storage reserve; Knecht's dedicated tmpfs uses 2 GiB.

For inspection or controlled development, the equivalent supervisor interface for
one dedicated Linux CPU island is below. Direct invocation does not install the
launcher's independent systemd CPU, memory or runtime limits:

```bash
python -B ops/world_supervisor.py \
  --deadline 2026-10-12T08:00:00Z \
  --config-base configs/neural-world.toml \
  --data data/conversation-world/dataset.json \
  --run-dir runs/world-extended-example \
  --stage-seconds 1800 --memory-limit-mib 2048 \
  --min-free-gib 50 --run-budget-gib 1 --project-budget-gib 20 \
  --self-replay --replay-episodes 32 --replay-samples 256
```

The launcher must supply the reviewed per-host device, CPU, memory and storage
limits; the example's 50 GiB storage reserve does not fit Knecht's smaller tmpfs.
`--resume` requires the same base configuration hash, frozen dataset hash and UTC
deadline, and refuses a surviving previous child. This resumes orchestration;
each subsequent stage still begins from a flat model warm-start. Inspect the
supervisor's root `status.json` directly and its stage reports for progress.
Use `global_best.pt`, rather than the finite-run path above, when inspecting or
chatting with a supervisor-selected champion.

`world-chat` supports `/meaning`, `/memory`, `/info`, `/reset`, `/reload` and
`/quit`. `/meaning` shows the model's predicted class distribution; `/memory`
shows the explicit USER-only fact state. Replies remain generated by the neural
decoder, with no fixed answer replacement.

Promote a checkpoint only from documented validation evidence and preserve its
source stage and hashes. Retain reports and selected checkpoints before pruning
old stages. Knecht's temporary checkout must be copied to persistent storage
before cleanup or reboot; tmpfs capacity is shared host RAM, not an independent
storage reserve.
