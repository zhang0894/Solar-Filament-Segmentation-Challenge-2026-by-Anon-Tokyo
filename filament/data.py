"""Mask-only supervised data with observation-grouped validation."""

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def read_manifest():
    return json.loads((ROOT / "data/manifest.json").read_text())


def load_masks(annotation_id, size=1024):
    with np.load(ROOT / f"data/cache_{size}/masks/{annotation_id}.npz") as data:
        return np.unpackbits(
            data["packed"], axis=-1, count=int(data["shape"][-1])
        ).astype(np.uint8)


def augment(image, masks):
    # Whole-image transforms retain coherent instance identities.
    k = random.randrange(4)
    image = np.rot90(image, k)
    masks = np.rot90(masks, k, axes=(-2, -1))
    if random.random() < 0.5:
        image = image[:, ::-1]
        masks = masks[:, :, ::-1]
    image = np.ascontiguousarray(image).astype(np.float32) / 255
    if random.random() < 0.8:
        image = np.clip(
            image * random.uniform(0.85, 1.15) + random.uniform(-0.04, 0.04), 0, 1
        )
        image = image ** random.uniform(0.8, 1.2)
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 0.8))
    if random.random() < 0.15:
        image = np.clip(image + np.random.normal(0, 0.006, image.shape), 0, 1)
    return image.astype(np.float32), np.ascontiguousarray(masks)


def image_tensor(image, contrast_input=False):
    # Match ImageNet pretraining without discarding original H-alpha intensity.
    if contrast_input:
        # Three image-derived intensity views; no dates, stations or labels.
        # A small background grid keeps illumination correction inexpensive.
        small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        background = cv2.resize(
            cv2.GaussianBlur(small, (0, 0), 8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        local = np.clip(0.5 * image / (background + 0.02), 0, 1)
        equalized = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            .apply(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
            .astype(np.float32)
            / 255
        )
        x = np.stack((image, local, equalized))
    else:
        x = np.repeat(image[None], 3, axis=0)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(x))


def component_weights(union):
    """Reduce large-object domination while preserving mean foreground weight."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(union, dtype=np.uint8), connectivity=8
    )
    table = np.ones(count, np.float32)
    if count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        weights = np.clip(np.sqrt(areas.mean() / areas), 0.25, 4.0)
        weights /= np.dot(weights, areas) / areas.sum()
        table[1:] = weights
    return table[labels]


def training_crop(image, target, annotation, tile_size, output_size):
    """Mix uniform crops and mask-derived centers without reading held-out labels."""
    from pycocotools import mask as mu

    height, width = image.shape
    size = min(height, width, round(tile_size * random.uniform(0.85, 1.15)))
    if random.random() < 0.5 and annotation["rles"]:
        x, y, w, h = mu.toBbox(random.choice(annotation["rles"]))
        cx = x + w / 2 + random.uniform(-0.25, 0.25) * size
        cy = y + h / 2 + random.uniform(-0.25, 0.25) * size
        x0 = max(0, min(width - size, round(cx - size / 2)))
        y0 = max(0, min(height - size, round(cy - size / 2)))
    else:
        x0, y0 = random.randint(0, width - size), random.randint(0, height - size)
    image = cv2.resize(
        image[y0 : y0 + size, x0 : x0 + size],
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    target = np.stack(
        [
            cv2.resize(
                t[y0 : y0 + size, x0 : x0 + size],
                (output_size, output_size),
                interpolation=cv2.INTER_NEAREST,
            )
            for t in target
        ]
    )
    return image, target


class FilamentDataset(Dataset):
    def __init__(
        self,
        fold=0,
        train=True,
        size=1024,
        kind="semantic",
        repeat=1,
        limit=0,
        fast_cache=True,
        component_balanced=False,
        contrast_input=False,
        tile_size=0,
    ):
        manifest = read_manifest()
        self.cache_size = manifest["cache_size"]
        self.rows = [
            r
            for r in manifest["images"]
            if ((r["fold"] != fold) if train else (r["fold"] == fold))
        ]
        if limit:
            self.rows = self.rows[:limit]
        self.train, self.size, self.kind, self.repeat = train, size, kind, repeat
        self.component_balanced = component_balanced
        self.contrast_input = contrast_input
        self.tile_size = tile_size
        if tile_size and (not train or kind != "semantic" or not 0 < tile_size <= 2048):
            raise ValueError("Native crop training requires semantic training data")
        self.fast_metadata = None
        self.fast_images = self.fast_targets = None
        directory = ROOT / f"data/cache_{2048 if tile_size else size}/fast"
        if fast_cache and (directory / "metadata.json").exists():
            metadata = json.loads((directory / "metadata.json").read_text())
            expected = hashlib.sha256(
                (ROOT / "data/manifest.json").read_bytes()
            ).hexdigest()
            if metadata["manifest_sha256"] != expected:
                raise ValueError("Fast cache was built from a different manifest")
            self.fast_metadata = metadata
            self.fast_images = np.load(directory / "images.npy", mmap_mode="r")
            if kind == "semantic":
                self.fast_targets = np.load(directory / "targets.npy", mmap_mode="r")
        if tile_size and self.fast_targets is None:
            raise ValueError("Native crop training requires the verified 2048 cache")

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        path = ROOT / f"data/cache_{self.cache_size}/images/{row['image_id']}.png"
        if self.size > self.cache_size:
            path = (
                ROOT
                / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
                / row["filename"]
            )
        image = (
            self.fast_images[self.fast_metadata["images"][row["image_id"]]]
            if self.fast_images is not None
            else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        )
        if self.kind == "image":
            if image.shape != (self.size, self.size):
                image = cv2.resize(
                    image, (self.size, self.size), interpolation=cv2.INTER_AREA
                )
            return {
                "image": image_tensor(
                    image.astype(np.float32) / 255, self.contrast_input
                ),
                "image_id": row["image_id"],
            }
        ann = random.choice(row["annotators"]) if self.train else row["annotators"][0]
        if self.fast_targets is not None:
            target = self.fast_targets[self.fast_metadata["annotations"][ann["id"]]]
            if self.fast_metadata.get("targets_packed", False):
                target = np.unpackbits(target, axis=-1, count=image.shape[-1])
            if self.tile_size:
                image, target = training_crop(
                    image, target, ann, self.tile_size, self.size
                )
            if self.train:
                image, target = augment(image, target)
            else:
                image = image.astype(np.float32) / 255
            if self.component_balanced:
                target = np.concatenate(
                    (target, component_weights(target[0])[None]), axis=0
                )
            return {
                "image": image_tensor(image, self.contrast_input),
                "image_id": row["image_id"],
                "target": torch.from_numpy(np.array(target, dtype=np.float32)),
            }
        if self.size > self.cache_size:
            from pycocotools import mask as mu

            masks = mu.decode(ann["rles"]).transpose(2, 0, 1)
        else:
            masks = load_masks(ann["id"], self.cache_size)
        if image.shape != (self.size, self.size):
            image = cv2.resize(
                image, (self.size, self.size), interpolation=cv2.INTER_AREA
            )
        if masks.shape[-1] != self.size:
            masks = np.stack(
                [
                    cv2.resize(
                        m, (self.size, self.size), interpolation=cv2.INTER_NEAREST
                    )
                    for m in masks
                ]
            )
        masks = masks[masks.sum(axis=(1, 2)) > 0]
        if self.train:
            image, masks = augment(image, masks)
        else:
            image = image.astype(np.float32) / 255
        item = {
            "image": image_tensor(image, self.contrast_input),
            "image_id": row["image_id"],
        }
        if self.kind == "semantic":
            union = (
                masks.max(axis=0)
                if len(masks)
                else np.zeros((self.size, self.size), np.uint8)
            )
            boundary = np.zeros_like(union)
            kernel = np.ones((3, 3), np.uint8)
            for mask in masks:
                boundary |= cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            item["target"] = torch.from_numpy(
                np.stack(
                    [
                        union,
                        boundary,
                        *(
                            [component_weights(union)]
                            if self.component_balanced
                            else []
                        ),
                    ]
                ).astype(np.float32)
            )
        else:
            # Torchvision consumes binary uint8 masks. Keeping this dtype cuts
            # pinned-memory and PCIe target traffic by 4x compared with float32.
            dtype = np.uint8 if self.kind == "rcnn" else np.float32
            item["masks"] = torch.from_numpy(masks.astype(dtype))
            item["labels"] = torch.zeros(len(masks), dtype=torch.long)
        return item


def instance_collate(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
        "masks": [x["masks"] for x in batch],
        "labels": [x["labels"] for x in batch],
    }


def worker_init(_):
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
