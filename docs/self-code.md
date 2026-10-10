# Predicting Golem's own code

Version 0.2 adds a separate code-prediction experiment. It keeps the English pilot and its model artifacts intact.

## What the experiment learns

The training corpus is a pinned snapshot of explicitly allowlisted tracked Golem Python source files. Whole files belong to only one of train, validation, or test. The manifest records the source commit, file digests, snapshot paths, normalization and split membership.

Code preserves capitalization, indentation, newlines and punctuation. Non-ASCII characters are represented as ASCII backslash escapes. The code genome uses a new v2 format and supports suffix contexts up to 16 characters. This remains a small local-context model, not a source-code understanding model.

Training targets are actual next characters in the training files. Corpus-guided mutation can propose changes from those examples; selection retains inherited predictors that perform better. Validation chooses champions on separate files. The final test files are excluded from both training and champion selection.

The source snapshot stays fixed. The trainer does not replace its own runtime source, import generated programs, execute model completions, or train on private directories.

## What improves itself

An organism changes its inherited prediction rules, integer weights, context order and smoothing through mutation and selection. The entire changed genome is copied to its offspring. The reproduction scaffold remains fixed in this version.

Each champion is also tested on its own canonical genome serialization. This `self_prediction` score is diagnostic only: a model must not gain fitness by simplifying or corrupting the target it is asked to predict. Scores on different genome strings are not directly comparable evidence of improvement, because the targets differ. Improvement is judged on fixed held-out source examples.

This is evolutionary self-improvement of the predictor, not autonomous rewriting of the Python engine. Lower prediction loss does not demonstrate code correctness, self-understanding, reasoning, or conversational ability.

## Run it

```bash
./golem prepare --source self-code --output data/self-code
bash ops/run_pilot.sh configs/self-code.toml data/self-code/dataset.json runs/self-code
./golem status --run-dir runs/self-code
./golem self-check --checkpoint runs/self-code/champion.json
./golem predict --checkpoint runs/self-code/champion.json --text 'def '
./golem chat --checkpoint runs/self-code/champion.json
```

The bounded launcher retains the same CPU, memory, storage and runtime policy. Reusing a prepared snapshot does not silently refresh it from changed runtime files; create a new dataset directory for a new source version.

Generated code is displayed as text. It is not evaluated or installed.
