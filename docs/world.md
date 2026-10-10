# English-world v1 specification

This is a new experimental world built from DNA-2 replication ideas. The frozen DNA-2 interpreter is a regression reference, not a claim that all English-world ticks match DNA-2.

## Heritable state and identity

An organism carries its replication program, maximum suffix order, and sparse next-character integer rules. A canonical serialized representation and full-record digest identify the entire heritable genome. Parent identity and organism identity are distinct from genome identity.

Material after the old DNA-2 STOP would not be copied by the old machine. This implementation therefore uses a new composite record with explicit termination. Predictor parameters are inside that record. Reproduction must prove exact child equality at zero mutation before any optional post-copy mutation event.

## Replication

The bounded copy interpreter uses START, ALLOC, COPY, TERM and DIVIDE. Allocation grants are delayed; COPY does not advance pointers without a granted destination. COPY transfers bounded chunks of the serialized payload. Copy cost accounts for all bytes, regardless of chunk size. TERM allows DIVIDE only after the full record terminator is copied. DIVIDE rearms the template so repeated copies can occur.

This chunked full-record process is deliberately different from DNA-2's one-base COPY. Its purpose is to preserve and verify the required inheritance properties for the larger predictor payload. It is not a molecular simulation.

## Predictor and mutations

The initial predictor is a bounded suffix-context model with mutable integer next-character weights. It emits a distribution over a fixed English character alphabet. Predictions use only the provided past context, never the next-character label or the dictionary.

Mutation proposals can add, strengthen, weaken or remove context rules and alter bounded program choices. Corpus-guided proposals draw only from the training split. They are an explicitly documented source of supervision. Parameter updates occur through mutations and selection, with no gradient optimization or pretrained weights.

## Environment and scoring

Common training examples give candidates comparable predictive tasks. Predictive log loss is the main fitness signal. Generated spelling contributes a bounded auxiliary term; repetition cannot collect unlimited spelling credit. Execution and full-record copying have explicit costs. Initial resources allow random founders to be evaluated.

Training feedback and validation selection are separate. Test examples are excluded from mutation and parent selection. Dictionary spelling is not a proxy for contextual relevance. Generation is bounded so neither execution nor repeated output can run indefinitely.

## Persistence

Checkpoints must retain the full population, lineage state, pseudo-random state, generation/progress position and configuration/data identities. Writes are atomic. Resume must preserve future results for matching conditions; a new target generation limit may extend a run, but changing model or dataset semantics requires a new run.

## Terminal inference

Terminal chat loads an immutable champion snapshot. It uses the actual end of the user's text and samples a bounded continuation. It must not append a fixed assistant-role marker: with a five-character suffix window that would hide the user input entirely. The short suffix window limits which parts of history influence a response. The interface is not evidence of general conversational ability.

## Release criteria

Tests cover complete zero-mutation inheritance, the three replication fixes, valid probabilities, deterministic generation and restart, reward bounds, split discipline and the frozen upstream regression. Learning claims require held-out results and appropriate baselines at matched evaluation budgets.
