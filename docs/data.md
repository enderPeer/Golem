# Data provenance

Original offline fixtures live in `fixtures/`. They are small implementation/test examples, not a language benchmark.

The public-data preparation command writes its exact source URLs, revision identifiers, SHA-256 digests, license notices, and split counts into the generated dataset directory. The published code does not embed downloaded training corpora.

Candidate English sources are:

- [ESDB/SCOWL](https://github.com/en-wl/wordlist): English wordlists with their required [copyright notices](https://github.com/en-wl/wordlist/blob/v2/Copyright).
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): synthetic simple English text, listed under CDLA-Sharing-1.0. A subset of the upstream validation file may be used to create this project's own document-separated train/validation/test partitions. Such a run is **not** the official TinyStories benchmark.
- [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1): a possible later dialogue source under its repository's Apache-2.0 terms; it is not required by the first pilot.

Read the generated manifest to see what was actually used. If preparation falls back to a vocabulary derived from training text, that dictionary is labeled as training vocabulary rather than an independent spelling authority. Download failures must not be silently presented as successful external-dataset preparation.

The source-code MIT license does not replace dataset terms. No personal workspace documents or interactive user prompts enter the training dataset automatically.
