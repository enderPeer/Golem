# First neural and low-bit results — 2026-10-10

These are measured prototype results, not evidence of a general-purpose LLM, open-ended algorithm invention, or a fabricated accelerator.

## Models and data

The language model is a 791,300-parameter byte GRU: embedding 128, two hidden layers of 256, and 260 byte/control outputs. The separate memory-test model has 44,036 parameters. Both were trained from random initialization; no pretrained language checkpoint supplied their behavior.

The pinned OpenAssistant pilot contains 512 selected English conversations: 410 training, 51 validation and 51 test. Whole conversation groups were split before extracting branches. The code repository includes preparation scripts and attribution; downloaded conversations and checkpoints remain outside Git.

## Does evolution help in the first pilot?

Each arm received 480 optimizer updates, 245,760 processed training tokens, 227,878 supervised targets and 64 validation forward batches. Architecture, batch shapes and the global training-batch schedule were identical. Four population rounds produced three checkpoint-inheritance events.

| Method | Validation bits/supervised token | Test bits/supervised token | Measured CPU seconds | Wall seconds |
|---|---:|---:|---:|---:|
| Ordinary gradient training | 2.8249 | **2.9850** | 62.03 | 32.78 |
| Four-candidate evolutionary training | 3.3853 | 3.6099 | 63.06 | 33.60 |

Lower prediction loss is better. **Ordinary training won this pilot.** Evolution has not demonstrated an advantage. This matches neural training/evaluation work, not exact total CPU time: population management added about 1.7% CPU in this run. Actual timings include initialization, copying and checkpoint costs. One seed and no independent best-of-four control limit any broader conclusion.

The saved checkpoints are `runs/neural-compare/ordinary_champion.pt` and `evolution_champion.pt` on the deployment. A longer follow-up is specified separately in `configs/neural-language.toml`; its outcomes are not included in this table.

## Does a neural organism retain earlier information?

The separate 44,036-parameter memory model reached the criterion after 175 updates, in 3.39 seconds of training/evaluation wall time on specht32's CPU.

| Gap between source digit and query | Held-out examples | Correct digit | Correct after removing source cue |
|---|---:|---:|---:|
| 32 distractor bytes | 500 | **100%** | 10% |
| 64 distractor bytes | 500 | **100%** | 10% |

The balanced chance rate is 10%. This demonstrates learned synthetic recall in that model and test. It does not demonstrate that the language checkpoint understands user questions.

## Does the language model use the question meaningfully?

On 32 held-out prompt/answer pairs, the ordinary checkpoint's mean answer loss was 2.15106 with the actual prompt and 2.14990 with a shuffled prompt. The evolutionary checkpoint's corresponding losses were 2.56832 and 2.56837. These near-zero differences do not establish useful response relevance. Generated prose remains unreliable.

## What happens with ternary weights?

The GRU/output matrices can be represented as 754,688 ternary values:

| Payload | Bytes |
|---|---:|
| Original FP32 parameters | 3,165,200 |
| Packed two-bit GRU/output matrices | 188,672 |
| Remaining FP32 embeddings and biases | 146,448 |
| Five FP32 matrix scales | 20 |
| Combined raw packed parameter payload | **335,140** |

That is about a 9.44x reduction of raw parameter payload. File headers, manifests, activations, recurrent state, workspaces and training state are additional. The export was round-trip verified, but a packed execution kernel is not implemented; the compatibility checkpoint still runs FP32 operations.

On one fixed validation set with eight evaluation batches:

| Representation/training | Bits/supervised token |
|---|---:|
| Original FP32 reference | **3.0558** |
| Direct post-training ternary conversion | 4.0894 |
| After 200 extra ternary-forward fine-tuning updates | 3.2578 |

Quantization-aware fine-tuning recovered much, but not all, of the prediction loss. It used 93,104 additional supervised targets and 29.82 CPU seconds. It is not an equal-training-compute comparison against the original model, and it is not native BitNet pretraining. Embeddings, biases, master parameters and execution remain floating point. No energy advantage is measured.

## AMD training compatibility

A separate user-space ROCm 7.14.1 / PyTorch 2.12.0 environment successfully ran five GRU training updates on the RX 9070 XT. Maximum CPU/GPU forward-logit difference was 2.09e-7, and peak allocated VRAM was about 178 MiB. No driver installation or reboot was performed. An optional rocSHMEM NUMA warning did not prevent this single-GPU test.

This verifies that specific forward/backward path. The controlled learning comparison above used PyTorch 2.14.1 on CPU. No FPGA, PIM, power or end-to-end low-bit latency result is claimed.

## Reproduce and inspect

See [neural setup and commands](neural.md). Reports and artifacts on the deployment are under `runs/neural-memory`, `runs/neural-compare`, `runs/neural-ternary`, `runs/neural-qat`, `runs/neural-qat-packed`, and `runs/runtime`. Runtime package versions were saved with the results. The original one-hour table-world language run was left running independently.
