"""Post-training diagnostics: prompt sensitivity and ternary export correctness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from .neural_model import ASSISTANT, END, USER, clone_ternary, encode_bytes, load_neural_checkpoint, save_neural_checkpoint
from .neural_data import prepare_documents
from .neural_training import _evaluate
from .ternary import pack_ternary, unpack_ternary


def response_relevance(model, dataset, split="test", max_cases=32, prompt_bytes=256, answer_bytes=128):
    """Compare held-out answers after their actual and a shuffled user prompt.

    The same answer bytes are scored in each pair. Positive shuffled-minus-true
    loss is evidence of prompt sensitivity, not a human relevance/accuracy score.
    """
    pairs = []
    for conversation in dataset.get("conversations", {}).get(split, []):
        user = None
        for turn in conversation["turns"]:
            if turn["role"] == "user":
                user = turn["content"]
            elif turn["role"] == "assistant" and user is not None:
                if user and turn["content"]:
                    pairs.append((user, turn["content"]))
                break
        if len(pairs) >= max_cases:
            break
    if len(pairs) < 2:
        raise ValueError("At least two held-out prompt/answer pairs are required")
    model.eval()
    device = next(model.parameters()).device

    def score(user, answer):
        prefix = [USER] + encode_bytes(user)[-prompt_bytes:] + [END, ASSISTANT]
        target = encode_bytes(answer)[:answer_bytes]
        tokens = prefix + target
        with torch.inference_mode():
            logits, _ = model(torch.tensor([tokens[:-1]], dtype=torch.long, device=device))
            positions = logits[0, len(prefix) - 1:len(prefix) - 1 + len(target)]
            loss = torch.nn.functional.cross_entropy(positions, torch.tensor(target, device=device), reduction="sum")
        return float(loss), len(target)

    actual = shuffled = tokens = 0
    better = 0
    differences = []
    for index, (user, answer) in enumerate(pairs):
        good, count = score(user, answer)
        bad, _ = score(pairs[(index + 1) % len(pairs)][0], answer)
        actual += good
        shuffled += bad
        tokens += count
        better += int(good < bad)
        differences.append((bad - good) / count)
    return {"split": split, "cases": len(pairs), "answer_bytes": tokens,
            "actual_prompt_loss": actual / tokens, "shuffled_prompt_loss": shuffled / tokens,
            "shuffled_minus_actual_loss": (shuffled - actual) / tokens,
            "fraction_preferring_actual_prompt": better / len(pairs),
            "paired_loss_differences": differences,
            "interpretation": "Descriptive prompt-sensitivity diagnostic; not proof that generated answers are relevant or correct."}


def ternary_export(checkpoint, dataset, output_dir, split="validation", batches=8):
    """Export packed matrix payloads and measure post-training ternary accuracy.

    Runtime still uses the separately saved float reference, never the packed file.
    No throughput, power, or native BitNet training claim follows from this export.
    """
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Ternary output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    model, metadata = load_neural_checkpoint(checkpoint)
    quantized, report = clone_ternary(model)
    docs = prepare_documents(dataset, split)
    cfg = {"batch_size": 4, "sequence_length": 128}
    report["evaluation"] = {"split": split,
                            "original": _evaluate(model, docs, cfg, batches, 4101),
                            "ternary_reference": _evaluate(quantized, docs, cfg, batches, 4101)}
    state = quantized.state_dict()
    payload = bytearray()
    for matrix in report["matrices"]:
        values = state[matrix["name"]].detach().cpu()
        scale = matrix["scale"]
        codes = (values / scale).round().to(torch.int8).reshape(-1).tolist() if scale else [0] * values.numel()
        packed = pack_ternary(codes)
        restored = torch.tensor(unpack_ternary(packed, len(codes)), dtype=values.dtype).reshape(values.shape) * scale
        if not torch.equal(restored, values):
            raise RuntimeError("Packed ternary round-trip changed a quantized matrix")
        matrix.update(offset=len(payload), packed_bytes=len(packed), packing="golem-ternary-row-major-v1")
        payload.extend(packed)
    (directory / "matrices.bin").write_bytes(payload)
    quantized_names = {m["name"] for m in report["matrices"]}
    residual = {name: tensor for name, tensor in state.items() if name not in quantized_names}
    torch.save(residual, directory / "residual-fp32.pt")
    save_neural_checkpoint(directory / "ternary-reference.pt", quantized,
                           metadata={**metadata, "quantization": "post_training_fake_ternary"})
    report.update(source_checkpoint=str(checkpoint), exact_packed_roundtrip=True,
                  packed_matrix_payload_bytes=len(payload),
                  residual_parameter_payload_bytes=sum(t.numel() * t.element_size() for t in residual.values()),
                  original_parameter_payload_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  packed_sha256=hashlib.sha256(payload).hexdigest(),
                  manifest_includes_scales=True, packed_execution_implemented=False,
                  packed_payload_exists=True)
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
