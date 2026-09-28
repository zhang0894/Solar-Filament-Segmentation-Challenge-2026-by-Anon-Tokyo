"""Short forward/backward benchmark; this is not a full training run."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parents[1] / "weights/huggingface")
)
os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parents[1] / "weights/torch")
)

import torch

from filament.models import make_model, make_optimizer, semantic_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["semantic", "instance"], default="semantic")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--sparse-points", action="store_true")
    p.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    a = p.parse_args()
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    model = make_model(a.kind, sparse_points=a.sparse_points).cuda().train()
    if a.channels_last and a.kind == "semantic":
        model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model, a.kind)
    x = torch.randn(a.batch, 3, a.size, a.size, device="cuda")
    if a.channels_last and a.kind == "semantic":
        x = x.contiguous(memory_format=torch.channels_last)
    target = torch.zeros(a.batch, 2, a.size, a.size, device="cuda")
    target[:, :, 100:150, 100:500] = 1
    timings = []
    for i in range(a.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.kind == "semantic":
                loss = semantic_loss(model(x), target)
            else:
                out = model(
                    pixel_values=x,
                    mask_labels=[t[:1] for t in target],
                    class_labels=[
                        torch.zeros(1, dtype=torch.long, device="cuda") for _ in target
                    ],
                )
                loss = out.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        print(
            json.dumps(
                {
                    "step": i,
                    "seconds": timings[-1],
                    "loss": float(loss.detach()),
                    "max_GB": torch.cuda.max_memory_allocated() / 1e9,
                }
            ),
            flush=True,
        )
    result = {
        "kind": a.kind,
        "size": a.size,
        "batch": a.batch,
        "seconds_per_step": sum(timings[3:]) / len(timings[3:]),
        "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/profile_{a.kind}_{a.size}_b{a.batch}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
