"""Small explicit single-GPU GRU training compatibility probe; no power claim."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from english_world.neural_model import ByteGRULM, NeuralConfig


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the intended AMD GPU with HIP_VISIBLE_DEVICES")
    name = torch.cuda.get_device_name(0)
    if "9070 XT" not in name:
        raise RuntimeError(f"Refusing to allocate on unexpected GPU: {name}")
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.25, 4 * 2**30 / props.total_memory), 0)
    torch.manual_seed(7)
    cpu = ByteGRULM(NeuralConfig()).eval()
    gpu = ByteGRULM(NeuralConfig()).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    inputs = torch.randint(0, 256, (4, 64))
    targets = torch.randint(0, 256, (4, 64), device="cuda")
    with torch.no_grad():
        a, _ = cpu(inputs)
        b, _ = gpu(inputs.cuda())
        error = float((a - b.cpu()).abs().max())
        if not torch.allclose(a, b.cpu(), atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"CPU/GPU forward disagreement: {error}")
    gpu.train()
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=.001)
    losses = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = gpu(inputs.cuda())
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 260), targets.reshape(-1))
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite GPU training result")
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    report = {"schema": "golem-gpu-compatibility-v1", "device": name, "torch": torch.__version__,
              "hip": torch.version.hip, "parameters": gpu.parameter_count,
              "cpu_gpu_max_absolute_logit_error": error, "training_steps": 5,
              "losses": losses, "training_wall_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "interpretation": "Compatibility smoke test on random data; not language quality, packed inference, or energy efficiency."}
    out = Path(__file__).resolve().parents[1] / "runs/runtime/gpu-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
