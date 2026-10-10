# Research boundaries and a path to efficient hardware

The project has an executable reference for packed ternary weights and exact
signed-int8 arithmetic. It does not yet have a measured FPGA or ASIC inference
engine. Replication, useful prediction, program synthesis, and efficient
hardware are separate results with separate acceptance tests.

## What the earlier results establish

The original DNA-2 sweep counted 768 intact five-codon copiers under its standard
code and 6 under its uniform code. At six codons the counts were 120,624 and
1,320. These are factors of 128 and approximately 91.38 in **intact-copier
density in the enumerated candidate spaces**. They do not measure a 91–128-fold
improvement in resistance to mutation. The sweeps used fixed START codons,
STOP-padded slots, eight-slot worlds, zero mutation, and a specified execution
budget. Operation frequencies, synonymous encodings and STOP frequencies differ
between the two code tables. Mutation robustness needs a separate controlled
experiment measuring reproductive success after mutations.

An intact child and a persistent lineage are also different measurements. The
length-six detailed records include 48 intact copiers without grandchildren
even after 256 ticks. Descendant counts must be joined to fidelity records before
making claims about faithful multigeneration propagation. The frozen reference
vendored here has its own limited [regression record](../upstream/dna2/PROVENANCE.json).
The full original sweep is not reproduced by that small regression.

The function-search machine and DNA-2 have different instruction sets, state,
input layouts and output conventions. A discovered function table is not a
callable DNA-2 instruction. Integrating one requires an adapter or a new VM
specification, exact truth-table tests, state-preservation tests and an explicit
execution cost.

The original bounded MAC search reports no correct modulo-four
multiply-accumulate for machine 50098 at program lengths 1–16, for every step
budget 1–256 on 256-cell rings and 1–512 on 512-cell rings, under its prescribed
inputs and output rules. This is a strong negative result **within those search
spaces**. It is not a proof that longer programs, different layouts, different
machines or composed primitives cannot perform a MAC. The new Python reference
implements the arithmetic directly; it does not claim that the old VM evolved
that operation.

## What “the model fits in two bits” means

The original Ghost report inspected Microsoft's packed BitNet checkpoint at
revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`. Its accounting is:

| Stored component | Parameter count | Tensor payload bytes |
|---|---:|---:|
| 210 ternary matrices, four two-bit weights per byte | 2,084,044,800 | 521,011,200 |
| Remaining BF16 embedding, normalization and scale parameters | 328,775,890 | 657,551,780 |
| Combined tensor payload | 2,412,820,690 | 1,178,562,980 |

The 328,775,890 figure counts **parameters**, not bytes. The first row is a matrix
subset, not the complete model or its live memory footprint. Tensor headers,
activations, KV caches, temporary buffers, framework objects and training state
add further storage. A three-valued alphabet has an information-theoretic size
of log2(3), approximately 1.585 bits per symbol; this simple format physically
uses two bits per weight.

Microsoft documents ternary weights, eight-bit activations and a model trained
with that quantization scheme. Its model card also states that the ordinary
Transformers path does not provide the specialized kernels needed for its
reported speed and energy benefits. Packed weights alone therefore do not
establish those benefits for Golem. See the [pinned Microsoft model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/04c3b9ad9361b824064a1f25ea60a8be9599b127/README.md)
and its [official inference implementation](https://github.com/microsoft/BitNet).

The Ghost report's accumulator-width wording also needs care: signed values
with magnitude up to 2,560 × 127 require 20 signed bits; magnitude up to
6,912 × 127 requires 21 signed bits. Allowing the full signed-int8 range adds
the -128 endpoint, giving bounds of 327,680 and 884,736 respectively, still
requiring 20 and 21 signed bits. The reference below uses checked int32
accumulation, including the case -1 × -128 = +128.

## The implemented primitive

[`english_world/ternary.py`](../english_world/ternary.py) defines:

- `pack_ternary` / `unpack_ternary`: -1 → `00`, 0 → `01`, +1 → `10`; `11` is
  rejected. Consecutive row-major weights occupy low-to-high bit pairs. Unused
  high pairs are zero padding, and the caller supplies the exact shape.
- `mac_int8` / `dot_int8`: signed-int8 inputs and signed-int32 accumulation,
  checked at every operation. Overflow raises; it never silently wraps or
  saturates.
- `matvec_int8`: a packed row-major matrix and an explicitly sized input vector.
  Rows can cross byte boundaries. No implicit scales, biases, quantization or
  normalization are applied.
- `storage_accounting`: analytical payload bytes with exclusions listed. Python
  process memory is not estimated from the packed array size.
- `self_test`: 1,093 exhaustive packing round trips, all 768 single-weight/int8
  products, 3,618 dot-product checks including long accumulator cases, 64 seeded
  matrix cases, and rejection checks. Expectations use exact Python integers.

Run the standalone reference checks without installing an ML framework:

```bash
python -m unittest discover -s tests -p test_ternary.py -v
python -c 'import json; from english_world.ternary import self_test; print(json.dumps(self_test(), indent=2))'
```

The project's contiguous row-major format differs from the row-block packing
described for the original checkpoint. Sharing the same two-bit value codes does
not make the byte layouts interchangeable. A checkpoint importer needs a tested
layout conversion before its matrix results can be compared.

## Hardware milestones

1. **Freeze the arithmetic contract.** Specify every scale, rounding rule,
   activation range, accumulator width and tensor layout for the selected
   network. The current primitive establishes only the integer dot product.
2. **Verify complete exported layers.** Compare an integer reference with the
   training model on saved inputs, including quantization boundaries, zeros,
   negative values and extreme magnitudes. Report the error tolerance and any
   prediction-quality change.
3. **Measure a software kernel.** An optimized implementation must match the
   reference before benchmarking. Report shape, batch size, context length,
   hardware, compiler, warm-up, transfer costs, latency distribution and memory
   use. A Python reference can be slower despite its smaller stored payload.
4. **Simulate a small hardware block.** A future RTL block can accept one signed
   activation and one ternary code per cycle, expose clear/valid/last signals,
   maintain a 32-bit accumulator and flag invalid codes or overflow. Its
   reset, stall and stream-boundary behavior need a testbench using the reference
   vectors. No RTL, synthesis or timing-closure result is claimed here.
5. **Build and measure an FPGA prototype.** Include packing, memory transfers,
   scales, normalization, nonlinear functions and attention or recurrence as
   required by the chosen network. Report LUTs, DSPs, block RAM, clock, latency,
   throughput, peak memory and measured power. State the measurement boundary.
6. **Consider ASIC design after measured success.** Area, SRAM capacity,
   off-chip bandwidth, process, timing closure and power are engineering results
   to obtain, not consequences of ternary storage.

Power is measured in watts; energy per inference or token is measured in joules.
One model has no universal wattage, and a data center's gigawatt-scale demand
cannot be assigned to a single model instance. A reduction in packed weight
bytes is not automatically the same reduction in wall-clock time or energy.
Training may still retain floating-point master weights and optimizer state.

## Current AMD compatibility evidence

Checked on 2026-10-10: AMD's ROCm 7.14.1 matrix includes Radeon RX 9070 / 9070 XT
as `gfx1201` and RX 9060 / 9060 XT as `gfx1200`. Its Radeon section lists Ubuntu
26.04 with GA kernel 7.0; the framework matrix includes Python 3.14. Exact
driver, GPU and framework combinations still need to match the selected
configuration. See the [official AMD compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html).

The reported project environment initially lacked Torch and a verified ROCm
stack. That means GPU execution was **untested in that environment**, not that
Ubuntu 26.04 or Python 3.14 is inherently unsupported. The arithmetic reference
needs neither package. No GPU or FPGA latency/energy result is established by
its tests, and this module does not install drivers or frameworks.

AMD also documents a user-space wheel path. For a separate project virtual
environment on an otherwise compatible `gfx1201` Linux system, the minimal
runtime-plus-Torch selection relevant to this project is:

```bash
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[libraries,device-gfx1201]==7.14.1' \
  'torch[device-gfx1201]==2.12.0+rocm7.14.1'
```

These selections come from AMD's [ROCm wheel instructions](https://rocm.docs.amd.com/en/docs-7.14.1/install/rocm.html)
and [PyTorch instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
The [official wheel index](https://repo.amd.com/rocm/whl-multi-arch/torch/) lists
the corresponding `cp314-cp314-linux_x86_64` wheel. This is a documented candidate
installation, not a claim that it was installed or tested by the ternary checks.
An existing compatible kernel driver and device permissions remain prerequisites.
Keep a working CPU environment separate; verify import, actual GPU execution and
numerical agreement before assigning this environment training work. No driver
replacement, reboot or elevated install is part of this recipe.

## What would demonstrate continuing improvement

Selection can improve a measured objective under a particular mutation process,
dataset and resource budget. It does not guarantee a global optimum, indefinite
improvement or open-ended evolution. Finite genome and resource limits restrict
the reachable behaviors. Guided mutations use supervision and must be reported
as such; gradient training and evolutionary selection must also be identified
separately when both are used.

Prediction of held-out code is useful evidence of statistical generalization.
It does not establish that generated programs execute correctly or improve the
running implementation. A self-code snapshot freezes its sources; a lower loss
on that task must not be described as the program rewriting or understanding
itself. Novelty requires an explicit behavioral measure, while useful novelty
requires independently checked task performance.

Use repeated seeds, fixed held-out evaluations and matched compute baselines.
Require an improvement to survive transfer to unseen tasks, preserve earlier
capabilities and meet its actual execution/memory budget. Publish failures and
plateaus as well as successful runs. Spelling accuracy, persistence, language
quality, code correctness and hardware efficiency remain separate metrics.

## Original experiment provenance

The numeric audit above read these supplied local reports. They are not bundled
into this repository; readers need the original experiment archive to reproduce
its large sweeps. Hashes identify the exact report bytes inspected, rather than
claiming those external experiments were rerun here.

| Original archive path | SHA-256 |
|---|---|
| `Ghost/bitnet_u2/REPORT.md` | `ef08ec0dba2455bc3f07c175cb2934aa8cbae5c57d2b32f95b86982857f0b96f` |
| `universe 2/docs/15_dna2.md` | `9cd5ee745cdb62ec1599d0462088c6dd1551065e8ad68d347274c8f92e789b26` |
| `universe 2/docs/14_mac_any_step.md` | `b2556ad96a67274b9548764e45d0df907205fe75e6155e982dc22eae817234a6` |
