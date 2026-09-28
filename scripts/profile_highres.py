"""Stress-test the optional detail head using many training instances."""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "weights/huggingface"))

import torch

from filament.models import make_model, make_optimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--instances", type=int, default=26)
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    model = (
        make_model("instance", pretrained=False, dense_masks=True, highres_masks=True)
        .cuda()
        .train()
    )
    opt = make_optimizer(model, "instance", 1e-4)
    image = torch.randn(a.batch, 3, a.size, a.size, device="cuda")
    masks = torch.zeros(a.instances, a.size, a.size, device="cuda")
    for i in range(a.instances):
        x = 32 + i * (a.size - 64) // a.instances
        masks[i, 32 : a.size - 32, x : x + 3] = 1
    timings = []
    for i in range(a.steps):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                pixel_values=image,
                mask_labels=[masks] * a.batch,
                class_labels=[torch.zeros(a.instances, dtype=torch.long, device="cuda")]
                * a.batch,
            )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        opt.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic() - start)
        print(
            json.dumps(
                {
                    "step": i,
                    "loss": float(out.loss),
                    "seconds": timings[-1],
                    "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                }
            ),
            flush=True,
        )
    result = vars(a) | {
        "seconds_per_step": sum(timings[2:]) / len(timings[2:]),
        "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
    }
    (ROOT / "runs" / f"profile_highres_b{a.batch}.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
