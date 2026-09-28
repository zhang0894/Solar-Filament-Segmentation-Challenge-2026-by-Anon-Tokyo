"""Refine independent instance proposals with overlapping native image crops."""

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from torch.utils.data import DataLoader, Dataset

from filament.data import image_tensor, read_manifest, worker_init
from filament.metrics import Evaluator, encode_crop_mask
from filament.prediction import filter_instances
from filament.refinement import crop_square, make_refiner


@lru_cache(maxsize=16)
def read_image(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def tile_origins(start, end, size, stride):
    if end - start <= size:
        return [round((start + end - size) / 2)]
    positions = list(range(start, end - size + 1, stride))
    if positions[-1] != end - size:
        positions.append(end - size)
    return positions


class RefinementInference(Dataset):
    def __init__(self, predictions, split, size):
        self.predictions, self.size = predictions, size
        self.image_dir = (
            ROOT
            / "data/MAGFiLO_1.0_Kaggle_2026"
            / ("test/test_images" if split == "test" else "train/train_images")
        )
        self.paths = {p.stem: p for p in self.image_dir.glob("*.jpeg")}
        self.proposals, self.tiles = [], []
        for name, items in predictions.items():
            for i, item in enumerate(items):
                x, y, w, h = mu.toBbox(item["rle"]).astype(int)
                x0, y0, x1, y1 = (
                    max(0, x - 48),
                    max(0, y - 48),
                    min(2048, x + w + 48),
                    min(2048, y + h + 48),
                )
                proposal = {
                    "name": name,
                    "index": i,
                    "box": [x0, y0, x1, y1],
                    "score": item["score"],
                }
                idx = len(self.proposals)
                self.proposals.append(proposal)
                for top in tile_origins(y0, y1, size, size * 2 // 3):
                    for left in tile_origins(x0, x1, size, size * 2 // 3):
                        self.tiles.append((idx, left, top))

    def image(self, name):
        return read_image(self.paths[name])

    def mask(self, idx):
        p = self.proposals[idx]
        return mu.decode(self.predictions[p["name"]][p["index"]]["rle"])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        idx, x, y = self.tiles[index]
        proposal = self.proposals[idx]
        image, _ = crop_square(
            self.image(proposal["name"]),
            x + self.size / 2,
            y + self.size / 2,
            self.size,
        )
        mask, _ = crop_square(
            self.mask(idx), x + self.size / 2, y + self.size / 2, self.size
        )
        image = torch.cat(
            (
                image_tensor(image.astype(np.float32) / 255),
                torch.from_numpy(mask[None].astype(np.float32)),
            )
        )
        return {"image": image, "tile": index}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tta", choices=["none", "flip", "d4"], default="none")
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if a.deterministic:
        from filament.runtime import configure_deterministic_inference

        configure_deterministic_inference()
    with Path(a.checkpoint).open("rb") as checkpoint_file:
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        checkpoint_file.seek(0)
        checkpoint_digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    config = ckpt["config"]
    base = Path(a.predictions)
    metadata = json.loads((base / "metadata.json").read_text())
    split = metadata["inference"]["split"]
    if (
        split != "test" and config["fold"] != metadata["model_config"]["fold"]
    ) or config["manifest_sha256"] != metadata["model_config"]["manifest_sha256"]:
        raise ValueError("Refiner and proposals must share the outer fold and manifest")
    predictions = json.loads((base / "predictions.json").read_text())
    dataset = RefinementInference(predictions, split, config["size"])
    model = (
        make_refiner(
            False,
            config.get("encoder", "resnet18"),
            config.get("detail_residual", False),
        )
        .cuda()
        .eval()
        .to(memory_format=torch.channels_last)
    )
    model.load_state_dict(ckpt["model"])
    loader = DataLoader(
        dataset,
        batch_size=a.batch,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    sums, weights, qualities = [], [], []
    for proposal in dataset.proposals:
        x0, y0, x1, y1 = proposal["box"]
        sums.append(np.zeros((y1 - y0, x1 - x0), np.float32))
        weights.append(np.zeros_like(sums[-1]))
        qualities.append([])
    window1d = np.maximum(np.hanning(config["size"]), 0.1).astype(np.float32)
    window = window1d[:, None] * window1d[None, :]
    from filament.performance import CUDAPrefetcher
    from filament.tta import refiner_tta

    for batch in CUDAPrefetcher(loader, channels_last=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, quality = refiner_tta(model, batch["image"], a.tta)
        probabilities = logits[:, 0].float().sigmoid().cpu().numpy()
        quality = quality[:, 0].float().sigmoid().cpu().numpy()
        for tile, prob, q in zip(batch["tile"].cpu().tolist(), probabilities, quality):
            idx, left, top = dataset.tiles[tile]
            x0, y0, x1, y1 = dataset.proposals[idx]["box"]
            ax0, ay0, ax1, ay1 = (
                max(x0, left),
                max(y0, top),
                min(x1, left + config["size"]),
                min(y1, top + config["size"]),
            )
            dst = np.s_[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
            src = np.s_[ay0 - top : ay1 - top, ax0 - left : ax1 - left]
            sums[idx][dst] += prob[src] * window[src]
            weights[idx][dst] += window[src]
            qualities[idx].append(float(q))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probabilities").mkdir(exist_ok=True)
    variants = {
        threshold: {name: [] for name in predictions}
        for threshold in [0.25, 0.4, 0.5, 0.6, 0.75]
    }
    for idx, proposal in enumerate(dataset.proposals):
        if np.any(weights[idx] == 0):
            raise ValueError("Incomplete tiling coverage")
        probability = sums[idx] / weights[idx]
        x0, y0, x1, y1 = proposal["box"]
        np.savez_compressed(
            out / "probabilities" / f"{proposal['name']}_{proposal['index']}.npz",
            probability=probability.astype(np.float16),
            box=proposal["box"],
            quality=np.mean(qualities[idx]),
            score=proposal["score"],
        )
        for threshold, result in variants.items():
            mask = probability > threshold
            if mask.sum() >= 64:
                result[proposal["name"]].append(
                    {
                        "rle": encode_crop_mask(mask, x0, y0),
                        "score": proposal["score"],
                        "quality": float(np.mean(qualities[idx])),
                    }
                )
    results = []
    for result in variants.values():
        for name, candidates in result.items():
            result[name] = filter_instances(candidates, score_threshold=0, min_area=64)
    if split == "valid":
        rows = [r for r in read_manifest()["images"] if r["fold"] == config["fold"]]
        for threshold, pred in variants.items():
            ev = Evaluator()
            for row in rows:
                for ann in row["annotators"]:
                    ev.update(
                        row["image_id"],
                        ann["id"],
                        ann["rles"],
                        [x["rle"] for x in pred[row["image_id"]]],
                    )
            results.append({"threshold": threshold, **ev.result()})
        best = max(results, key=lambda x: x["official"]["pq"])
        (out / "validation.json").write_text(json.dumps(results, indent=2))
        (out / "predictions.json").write_text(json.dumps(variants[best["threshold"]]))
        print(json.dumps(best), flush=True)
    metadata.update(
        refiner_config=config,
        refiner_checkpoint=a.checkpoint,
        refiner_epoch=ckpt["epoch"],
        refiner_checkpoint_sha256=checkpoint_digest,
        refiner_tta=a.tta,
        tiles=len(dataset),
        proposals=len(dataset.proposals),
        proposal_directory=str(base.resolve()),
        proposal_sha256=hashlib.sha256(
            (base / "predictions.json").read_bytes()
        ).hexdigest(),
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
