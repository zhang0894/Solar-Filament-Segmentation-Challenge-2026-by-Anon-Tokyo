"""Deterministic validation/test inference and native-resolution RLE export."""

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "weights/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from filament.data import image_tensor, read_manifest, worker_init
from filament.metrics import Evaluator, decode_mask
from filament.models import make_model
from filament.prediction import (
    filter_instances,
    query_candidates_cpu,
    rcnn_candidates_cpu,
    semantic_instances,
)
from filament.tta import semantic_tta


class InferenceDataset(Dataset):
    def __init__(self, split, fold, size, contrast_input=False):
        self.size = size
        self.contrast_input = contrast_input
        self.rows = []
        if split == "test":
            self.paths = sorted(
                (ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg")
            )
        else:
            self.rows = [
                r
                for r in read_manifest()["images"]
                if (r["fold"] != fold if split == "train" else r["fold"] == fold)
            ]
            self.paths = [
                ROOT / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images" / r["filename"]
                for r in self.rows
            ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image = (
            cv2.resize(
                image, (self.size, self.size), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255
        )
        return {
            "image": image_tensor(image, self.contrast_input),
            "image_id": path.stem,
        }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=-1)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--min-score", type=float, default=0)
    parser.add_argument("--group-radius", type=int, default=0)
    parser.add_argument("--tta", choices=["none", "flip", "d4"], default="none")
    parser.add_argument("--cache-probs", action="store_true")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    with Path(args.checkpoint).open("rb") as checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        checkpoint_file.seek(0)
        checkpoint_digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    config = checkpoint["config"]
    if config["fold"] < 0 and args.split == "valid":
        raise ValueError(
            "An all-data deployment model has no held-out validation split"
        )
    expected_hash = hashlib.sha256(
        (ROOT / "data/manifest.json").read_bytes()
    ).hexdigest()
    if config["manifest_sha256"] != expected_hash:
        raise ValueError("Checkpoint/manifest mismatch")
    kind = config["kind"]
    torch.backends.cudnn.benchmark = kind != "rcnn"
    if args.deterministic:
        from filament.runtime import configure_deterministic_inference

        configure_deterministic_inference()
    model = (
        make_model(
            kind,
            pretrained=False,
            sparse_points=config.get("sparse_points", False),
            dense_masks=config.get("dense_masks", False),
            highres_masks=config.get("highres_masks", False),
            encoder=config.get("encoder", "tu-convnext_tiny.fb_in22k_ft_in1k"),
        )
        .to(args.device)
        .eval()
    )
    model.load_state_dict(checkpoint["model"])
    if kind == "semantic":
        model.to(memory_format=torch.channels_last)
    if args.tta != "none" and kind != "semantic":
        raise ValueError(
            "Instance TTA requires explicit instance matching and is not enabled"
        )
    size = args.size or config["size"]
    tile_size = config.get("tile_size", 0)
    if tile_size and (kind != "semantic" or config.get("contrast_input", False)):
        raise ValueError("Unsupported native tile checkpoint")
    threshold = (
        args.threshold
        if args.threshold >= 0
        else checkpoint.get("validation", {}).get("threshold", 0.5)
    )
    dataset = InferenceDataset(
        args.split,
        config["fold"],
        2048 if tile_size else size,
        config.get("contrast_input", False),
    )
    batch_size = args.batch or config["batch"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=args.device == "cuda",
        worker_init_fn=worker_init,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.cache_probs:
        (output / "probabilities").mkdir(exist_ok=True)
    predictions = {}

    def finish_image(name, raw):
        if kind == "semantic":
            candidates = semantic_instances(
                raw,
                threshold=threshold,
                min_area=args.min_area,
                min_score=args.min_score,
                group_radius=args.group_radius,
            )
            if args.cache_probs:
                np.savez_compressed(
                    output / "probabilities" / f"{name}.npz",
                    probability=raw.astype(np.float16),
                )
        else:
            raw = query_candidates_cpu(raw) if kind == "instance" else raw
            candidates = filter_instances(
                raw, score_threshold=threshold, min_area=args.min_area
            )
            if args.cache_probs:
                (output / "probabilities" / f"{name}.json").write_text(json.dumps(raw))
        return candidates

    from filament.performance import CUDAPrefetcher

    batches = (
        CUDAPrefetcher(loader, channels_last=kind == "semantic")
        if args.device == "cuda"
        else loader
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        for i, batch in enumerate(batches):
            image = batch["image"]
            if kind == "semantic":
                image = image.contiguous(memory_format=torch.channels_last)
            with torch.autocast(args.device, dtype=torch.bfloat16):
                if kind == "semantic":
                    if tile_size:
                        from filament.tiling import tiled_logits

                        logits = tiled_logits(model, image, tile_size, size, args.tta)
                    else:
                        logits = semantic_tta(model, image, args.tta)
                    raw_batch = logits.float().sigmoid().cpu().numpy()
                elif kind == "rcnn":
                    raw_batch = [
                        rcnn_candidates_cpu(o) for o in model(pixel_values=image)
                    ]
                else:
                    out = model(pixel_values=image)
                    masks = out.masks_queries_logits.to(torch.float16).cpu().numpy()
                    classes = out.class_queries_logits.float().cpu().numpy()
                    raw_batch = [
                        {"mask_logits": masks[j], "class_logits": classes[j]}
                        for j in range(len(batch["image_id"]))
                    ]
            for name, raw in zip(batch["image_id"], raw_batch):
                pending.append((name, pool.submit(finish_image, name, raw)))
            if (i + 1) % 10 == 0:
                print(
                    json.dumps({"processed": len(pending), "total": len(dataset)}),
                    flush=True,
                )
        predictions = {name: future.result() for name, future in pending}
    (output / "predictions.json").write_text(json.dumps(predictions))
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint_digest,
        "model_config": config,
        "inference": vars(args),
        "resolved_threshold": threshold,
        "resolved_size": size,
        "images": len(predictions),
        "instances": sum(len(v) for v in predictions.values()),
        "empty_images": [k for k, v in predictions.items() if not v],
    }
    if args.split == "valid":
        ev = Evaluator()
        for row in dataset.rows:
            rles = [p["rle"] for p in predictions[row["image_id"]]]
            for ann in row["annotators"]:
                ev.update(row["image_id"], ann["id"], ann["rles"], rles)
        metadata["evaluation"] = ev.result()
        (output / "entries.json").write_text(json.dumps(ev.entries))
    elif args.split == "test":
        csv_path = output / "submission.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filament_id", "segmentation_rle"])
            for name, candidates in predictions.items():
                for j, pred in enumerate(candidates, 1):
                    rle = pred["rle"]
                    if rle["size"] != [2048, 2048] or not decode_mask(rle).any():
                        raise ValueError("Invalid submission mask")
                    writer.writerow([f"{name}_{j}", rle["counts"]])
        # Parse CSV back to ensure counts survive serialization exactly.
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert (
            len({r["filament_id"] for r in rows}) == len(rows) == metadata["instances"]
        )
        for r in rows:
            assert decode_mask(
                {"size": [2048, 2048], "counts": r["segmentation_rle"]}
            ).any()
        metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
