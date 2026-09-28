"""Train a reproducible grouped-fold segmentation experiment."""

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "weights/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from filament.data import FilamentDataset, instance_collate, worker_init
from filament.models import make_model, make_optimizer, semantic_loss
from filament.performance import (
    CUDAPrefetcher,
    EMAUpdater,
    accumulation_fraction,
    check_finite,
    cpu_state_dict,
    validate_initialization_config,
    validate_resume_config,
)
from filament.runtime import register_training_process, remaining_session_seconds
from filament.validation import collect_predictions, score_predictions


def save_atomic(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=["semantic", "instance", "rcnn"], default="semantic"
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Final fixed-epoch deployment training without held-out validation",
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sparse-points", action="store_true")
    parser.add_argument("--dense-masks", action="store_true")
    parser.add_argument("--highres-masks", action="store_true")
    parser.add_argument("--component-balanced", action="store_true")
    parser.add_argument("--annotation-weighted", action="store_true")
    parser.add_argument("--contrast-input", action="store_true")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=1)
    parser.add_argument("--encoder", default="tu-convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=2)
    args = parser.parse_args()
    if args.all_data:
        if args.limit:
            parser.error("all-data training must include the complete training set")
        args.fold = -1
    if args.component_balanced and args.kind != "semantic":
        parser.error("component-balanced supervision is for semantic models")
    if args.train_repeat < 1:
        parser.error("train-repeat must be positive")
    if args.tile_size and (
        args.kind != "semantic" or not 0 < args.tile_size <= 2048 or args.contrast_input
    ):
        parser.error(
            "Tile training requires semantic grayscale input and tiles <= 2048"
        )
    if args.init and args.resume:
        parser.error("init and resume are mutually exclusive")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    # ROI mask-head batch sizes vary with the sampled positive proposals.
    # Searching convolution algorithms for every new count stalls detection
    # training; fixed-shape semantic/query models still benefit from autotuning.
    torch.backends.cudnn.benchmark = args.kind != "rcnn"
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    register_training_process(run)
    config = vars(args).copy()
    config["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    config["manifest_sha256"] = hashlib.sha256(
        (ROOT / "data/manifest.json").read_bytes()
    ).hexdigest()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        validate_resume_config(checkpoint["config"], config)
        (run / f"config_resume_{time.time_ns()}.json").write_text(
            json.dumps(config, indent=2)
        )
    else:
        if (run / "last.pt").exists():
            raise ValueError(
                "Existing checkpoint: use resume or a different experiment name"
            )
        (run / "config.json").write_text(json.dumps(config, indent=2))
    train = FilamentDataset(
        args.fold,
        True,
        args.size,
        args.kind,
        limit=args.limit,
        component_balanced=args.component_balanced,
        contrast_input=args.contrast_input,
        tile_size=args.tile_size,
        repeat=args.train_repeat,
    )
    valid = FilamentDataset(
        args.fold,
        False,
        2048 if args.tile_size else args.size,
        "image",
        limit=args.limit,
        contrast_input=args.contrast_input,
    )
    collate = instance_collate if args.kind in {"instance", "rcnn"} else None
    loader_args = {
        "num_workers": args.workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
        "persistent_workers": args.workers > 0,
        "collate_fn": collate,
    }
    if args.workers:
        loader_args["prefetch_factor"] = args.prefetch_factor
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (
        WeightedRandomSampler(
            [len(row["annotators"]) for row in train.rows],
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
        if args.annotation_weighted
        else None
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **loader_args,
    )
    val_options = {
        "num_workers": args.val_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init,
        "persistent_workers": args.val_workers > 0,
    }
    if args.val_workers:
        val_options["prefetch_factor"] = args.prefetch_factor
    valid_loader = DataLoader(
        valid, batch_size=args.batch, shuffle=False, **val_options
    )
    model = make_model(
        args.kind,
        pretrained=not bool(args.init or args.resume),
        sparse_points=args.sparse_points,
        dense_masks=args.dense_masks,
        highres_masks=args.highres_masks,
        encoder=args.encoder,
    ).cuda()
    channels_last = args.channels_last and args.kind == "semantic"
    if channels_last:
        model.to(memory_format=torch.channels_last)
    if args.init:
        previous = torch.load(args.init, map_location="cpu", weights_only=False)
        previous_config = previous["config"]
        validate_initialization_config(previous_config, config)
        adding_detail = args.highres_masks and not previous_config.get(
            "highres_masks", False
        )
        loaded = model.load_state_dict(previous["model"], strict=not adding_detail)
        if adding_detail and (
            loaded.unexpected_keys
            or any(
                "pixel_level_module.detail_head." not in key
                for key in loaded.missing_keys
            )
        ):
            raise ValueError(
                "Unexpected initialization mismatch beyond the new detail head"
            )
        del previous
    optimizer = make_optimizer(model, args.kind, args.lr)
    updates_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_updates = updates_per_epoch * args.epochs

    def schedule(step):
        warmup = updates_per_epoch
        if step < warmup:
            return max(0.05, (step + 1) / warmup)
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = deepcopy(model).eval() if args.ema else None
    if ema:
        for p in ema.parameters():
            p.requires_grad_(False)
    start_epoch, best_score, update = 0, -1.0, 0
    resume_validation = False
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["fused"], group["foreach"] = True, None
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if channels_last and parameter.ndim == 4:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key] = state[key].contiguous(
                                memory_format=torch.channels_last
                            )
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].cuda()
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema:
            ema.load_state_dict(checkpoint["ema"])
        start_epoch, best_score, update = (
            checkpoint["epoch"],
            checkpoint["best_score"],
            checkpoint["update"],
        )
        resume_validation = checkpoint.get("validation_due", False)
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"])
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        generator.set_state(checkpoint["rng_loader"])
        if update != start_epoch * updates_per_epoch:
            raise ValueError("Resume changed optimizer updates per epoch")
        del checkpoint
        if (run / "best.pt").exists():
            prior = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            prior_score = prior.get("validation", {}).get("official", {}).get("pq")
            if prior_score is not None:
                best_score = max(best_score, prior_score)
            del prior
    updater = EMAUpdater(ema, model) if ema is not None else None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(
            json.dumps({"event": "stop_requested", "action": "save_at_epoch_boundary"}),
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    scorer = ThreadPoolExecutor(max_workers=1)
    pending = None

    def finish_validation(wait=False):
        nonlocal pending, best_score
        if pending is None or (not wait and not pending[0].done()):
            return
        future, epoch_number, snapshot, directory = pending
        best, results = future.result()
        if best["official"]["pq"] > best_score:
            best_score = best["official"]["pq"]
            save_atomic(
                run / "best.pt",
                {
                    "model": snapshot,
                    "config": config,
                    "epoch": epoch_number,
                    "validation": best,
                },
            )
            for name in ["validation.json", "predictions.json", "entries.json"]:
                (run / ("best_" + name)).write_bytes((directory / name).read_bytes())
        record = {
            "event": "validation",
            "epoch": epoch_number,
            "validation": best,
            "thresholds": results,
            "best_pq": best_score,
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
        with (run / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        pending = None

    def launch_validation(epoch_number):
        nonlocal pending
        finish_validation(wait=True)
        inference_model = ema if ema is not None else model
        raw = collect_predictions(
            inference_model,
            valid_loader,
            args.kind,
            channels_last=channels_last,
            tile_size=args.tile_size,
            model_size=args.size,
        )
        snapshot = cpu_state_dict(inference_model)
        directory = run / f"validation_epoch_{epoch_number:03d}"
        pending = (
            scorer.submit(
                score_predictions, raw, valid.rows, args.kind, save_dir=directory
            ),
            epoch_number,
            snapshot,
            directory,
        )

    print(
        json.dumps(
            {
                "event": "start",
                "config": config,
                "train_images": len(train),
                "valid_images": len(valid),
                "fast_cache": train.fast_metadata is not None,
                "steps_per_epoch": len(train_loader),
                "resume_epoch": start_epoch,
            }
        ),
        flush=True,
    )
    start = time.monotonic()
    completed_epoch = start_epoch
    if resume_validation and remaining_session_seconds() >= 150:
        launch_validation(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        if remaining_session_seconds() < 150:
            stop_requested = True
            print(json.dumps({"event": "budget_stop", "epoch": epoch}), flush=True)
            break
        finish_validation()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        epoch_start = time.monotonic()
        for step, batch in enumerate(
            CUDAPrefetcher(train_loader, channels_last=channels_last)
        ):
            image = batch["image"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.kind == "semantic":
                    target = batch["target"]
                    loss = semantic_loss(model(image), target)
                else:
                    output = model(
                        pixel_values=image,
                        mask_labels=batch["masks"],
                        class_labels=batch["labels"],
                    )
                    loss = output.loss
            check_finite(loss, "Nonfinite training loss")
            # Correctly normalize the final partial accumulation group.
            group_size = min(
                args.accum, len(train_loader) - (step // args.accum) * args.accum
            )
            fraction = (
                accumulation_fraction(step, args.batch, args.accum, len(train))
                if args.kind == "semantic"
                else 1 / group_size
            )
            (loss * fraction).backward()
            losses.append(loss.detach())
            if (step + 1) % args.accum == 0 or step + 1 == len(train_loader):
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, foreach=True
                )
                check_finite(norm, "Nonfinite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
                if ema:
                    decay = min(args.ema, (1 + update) / (10 + update))
                    updater.update(decay)
            if (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "loss": float(torch.stack(losses[-100:]).mean()),
                            "epoch_seconds": round(time.monotonic() - epoch_start, 1),
                        }
                    ),
                    flush=True,
                )
        mean_loss = float(torch.stack(losses).mean())
        train_seconds = time.monotonic() - epoch_start
        if remaining_session_seconds() < train_seconds + 150:
            stop_requested = True
        finish_validation()
        record = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "train_seconds": round(train_seconds, 2),
            "images_per_second": round(len(train) / train_seconds, 2),
            "lr": optimizer.param_groups[-1]["lr"],
            "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
        }
        validation_due = (
            not args.all_data
            and not stop_requested
            and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs)
        )
        record["elapsed_seconds"] = round(time.monotonic() - start, 1)
        record["best_pq"] = best_score
        with (run / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        # Persist a completed epoch before validation touches CUDA again. A
        # failed validation can then be replayed from this exact EMA on resume.
        save_atomic(
            run / "last.pt",
            {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_score": best_score,
                "update": update,
                "config": config,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
                "rng_loader": generator.get_state(),
                "validation_due": validation_due,
            },
        )
        completed_epoch = epoch + 1
        if validation_due:
            launch_validation(epoch + 1)
        if stop_requested or epoch + 1 == args.epochs:
            finish_validation(wait=True)
        if stop_requested:
            break
    finish_validation(wait=True)
    scorer.shutdown(wait=True)
    if args.all_data and completed_epoch == args.epochs:
        save_atomic(
            run / "best.pt",
            {
                "model": cpu_state_dict(ema if ema is not None else model),
                "config": config,
                "epoch": completed_epoch,
                "selection": "fixed_epochs_all_training_images_no_validation",
            },
        )
    print(
        json.dumps(
            {
                "event": "paused" if stop_requested else "complete",
                "best_pq": best_score if not args.all_data else None,
                "minutes": round((time.monotonic() - start) / 60, 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
