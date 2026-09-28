"""Train a native-resolution refiner using only the outer training fold."""

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "weights/huggingface"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from filament.data import worker_init
from filament.performance import (
    CUDAPrefetcher,
    EMAUpdater,
    check_finite,
    cpu_state_dict,
)
from filament.refinement import RefinementDataset, make_refiner, refinement_loss
from filament.runtime import register_training_process, remaining_session_seconds
from scripts.train import save_atomic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quality-weight", type=float, default=0.1)
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--detail-residual", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.quality_weight < 0:
        p.error("quality-weight must be nonnegative")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    random.seed(20260928)
    np.random.seed(20260928)
    torch.manual_seed(20260928)
    run = ROOT / "runs" / a.name
    run.mkdir(exist_ok=True)
    register_training_process(run)
    if (run / "last.pt").exists() and not a.resume:
        raise ValueError("Choose a new run name")
    config = vars(a) | {
        "kind": "refinement",
        "manifest_sha256": hashlib.sha256(
            (ROOT / "data/manifest.json").read_bytes()
        ).hexdigest(),
    }
    if not a.resume:
        (run / "config.json").write_text(json.dumps(config, indent=2))
    dataset = RefinementDataset(a.predictions, a.fold, a.size, a.samples)
    loader = DataLoader(
        dataset,
        batch_size=a.batch,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        worker_init_fn=worker_init,
    )
    model = (
        make_refiner(not a.resume, a.encoder, a.detail_residual)
        .cuda()
        .to(memory_format=torch.channels_last)
    )
    ema = deepcopy(model).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    updater = EMAUpdater(ema, model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=True)
    total = len(loader) * a.epochs
    stop = False

    def halt(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    step = 0
    start_epoch = 0
    if a.resume:
        previous = torch.load(
            run / "training_state.pt", map_location="cpu", weights_only=False
        )
        if previous["config"].get("quality_weight", 0.1) != a.quality_weight:
            raise ValueError("Resume config mismatch: quality_weight")
        if previous["config"].get("encoder", "resnet18") != a.encoder:
            raise ValueError("Resume config mismatch: encoder")
        if previous["config"].get("detail_residual", False) != a.detail_residual:
            raise ValueError("Resume config mismatch: detail_residual")
        for key in [
            "fold",
            "size",
            "batch",
            "epochs",
            "samples",
            "lr",
            "predictions",
            "manifest_sha256",
        ]:
            if previous["config"][key] != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        model.load_state_dict(previous["model"])
        ema.load_state_dict(previous["ema"])
        opt.load_state_dict(previous["optimizer"])
        for group in opt.param_groups:
            for parameter in group["params"]:
                state = opt.state[parameter]
                if parameter.ndim == 4:
                    for key in ["exp_avg", "exp_avg_sq"]:
                        state[key] = state[key].contiguous(
                            memory_format=torch.channels_last
                        )
                state["step"] = state["step"].cuda()
        start_epoch, step = previous["epoch"], previous["step"]
        random.setstate(previous["rng_python"])
        np.random.set_state(previous["rng_numpy"])
        torch.set_rng_state(previous["rng_torch"])
        torch.cuda.set_rng_state_all(previous["rng_cuda"])
        del previous
    start = time.monotonic()
    for epoch in range(start_epoch, a.epochs):
        if remaining_session_seconds() < 120:
            break
        model.train()
        losses = []
        epoch_start = time.monotonic()
        for batch in CUDAPrefetcher(loader, channels_last=True):
            step += 1
            progress = step / total
            lr = a.lr * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
            if step < len(loader):
                lr *= max(0.1, step / len(loader))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, quality = model(batch["image"])
                loss = refinement_loss(
                    logits, quality, batch, quality_weight=a.quality_weight
                )
            check_finite(loss, "Nonfinite refinement loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            opt.step()
            updater.update(min(0.99, (1 + step) / (10 + step)))
            losses.append(loss.detach())
        state = {"model": cpu_state_dict(ema), "config": config, "epoch": epoch + 1}
        save_atomic(run / "last.pt", state)
        save_atomic(
            run / "training_state.pt",
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "step": step,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
            },
        )
        if (epoch + 1) % 3 == 0:
            save_atomic(run / f"epoch_{epoch + 1:02d}.pt", state)
        record = {
            "epoch": epoch + 1,
            "loss": float(torch.stack(losses).mean()),
            "seconds": time.monotonic() - epoch_start,
            "elapsed": time.monotonic() - start,
            "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
        }
        print(json.dumps(record), flush=True)
        with (run / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if stop or remaining_session_seconds() < record["seconds"] + 120:
            break


if __name__ == "__main__":
    main()
