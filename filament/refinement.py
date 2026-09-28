"""Native-resolution instance-conditioned refinement and rejection.

The refiner receives an H-alpha crop and the current instance mask. Its target
is derived from the same fold's allowed training masks; it never sees held-out
observation labels during training.
"""

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from torch.utils.data import Dataset

from filament.data import ROOT, augment, image_tensor, read_manifest
from filament.metrics import pairwise_iou


@lru_cache(maxsize=16)
def native_image(filename):
    cached = native_image_cache()
    if cached is not None:
        images, indices = cached
        return images[indices[Path(filename).stem]]
    return cv2.imread(
        str(ROOT / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images" / filename),
        cv2.IMREAD_GRAYSCALE,
    )


@lru_cache(maxsize=1)
def native_image_cache():
    directory = ROOT / "data/cache_2048/fast"
    if not (directory / "metadata.json").exists():
        return None
    metadata = json.loads((directory / "metadata.json").read_text())
    expected = hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest()
    if metadata["manifest_sha256"] != expected:
        raise ValueError("Native image cache has incompatible provenance")
    return np.load(directory / "images.npy", mmap_mode="r"), metadata["images"]


def crop_square(array, cx, cy, size, fill=0):
    h, w = array.shape[-2:]
    x0, y0 = round(cx - size / 2), round(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    output = np.full((*array.shape[:-2], size, size), fill, dtype=array.dtype)
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if ax1 > ax0 and ay1 > ay0:
        output[..., ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = array[
            ..., ay0:ay1, ax0:ax1
        ]
    return output, (x0, y0)


def corrupt_mask(mask):
    mask = mask.copy()
    kernel = np.ones((random.choice([3, 5, 7]),) * 2, np.uint8)
    mode = random.randrange(4)
    if mode == 0:
        mask = cv2.dilate(mask, kernel)
    elif mode == 1:
        mask = cv2.erode(mask, kernel)
    elif mode == 2:
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)
        mask = cv2.warpAffine(
            mask,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            mask.shape[::-1],
            flags=cv2.INTER_NEAREST,
        )
    else:
        scale = random.choice([2, 4])
        small = cv2.resize(
            mask,
            (mask.shape[1] // scale, mask.shape[0] // scale),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(small, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return mask


class RefinementDataset(Dataset):
    def __init__(self, prediction_dir, fold=0, crop_size=384, samples=6000):
        metadata = json.loads((Path(prediction_dir) / "metadata.json").read_text())
        expected_hash = hashlib.sha256(
            (ROOT / "data/manifest.json").read_bytes()
        ).hexdigest()
        if (
            metadata["model_config"]["fold"] != fold
            or metadata["inference"]["split"] != "train"
            or metadata["model_config"]["manifest_sha256"] != expected_hash
        ):
            raise ValueError(
                "Refinement requires predictions for the same outer training fold"
            )
        self.rows = {
            r["image_id"]: r for r in read_manifest()["images"] if r["fold"] != fold
        }
        self.predictions = json.loads(
            (Path(prediction_dir) / "predictions.json").read_text()
        )
        if set(self.predictions) != set(self.rows):
            raise ValueError(
                "Refinement predictions must cover only and all outer training observations"
            )
        self.candidates = [
            (name, i)
            for name, items in self.predictions.items()
            for i in range(len(items))
        ]
        self.positive_ids = list(self.rows)
        self.samples, self.crop_size = samples, crop_size
        # Cache match indices, not decoded masks. Every annotator is kept separate.
        self.matches = {}
        for name, row in self.rows.items():
            pred_rles = [p["rle"] for p in self.predictions[name]]
            self.matches[name] = []
            for ann in row["annotators"]:
                ious = pairwise_iou(ann["rles"], pred_rles)
                self.matches[name].append(
                    (
                        ious.argmax(0) if len(pred_rles) else np.array([], int),
                        ious.max(0) if len(pred_rles) else np.array([]),
                    )
                )

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        synthetic = random.random() < 0.25 or not self.candidates
        if synthetic:
            name = random.choice(self.positive_ids)
            row = self.rows[name]
            ann = random.choice(row["annotators"])
            rle = random.choice(ann["rles"])
            truth = mu.decode(rle).astype(np.uint8)
            coarse = truth
            match_iou = 1.0
        else:
            name, candidate_index = random.choice(self.candidates)
            row = self.rows[name]
            annotator_index = random.randrange(len(row["annotators"]))
            ann = row["annotators"][annotator_index]
            match_indices, match_ious = self.matches[name][annotator_index]
            rle = self.predictions[name][candidate_index]["rle"]
            coarse = mu.decode(rle).astype(np.uint8)
            match_iou = float(match_ious[candidate_index])
            truth = (
                mu.decode(ann["rles"][int(match_indices[candidate_index])]).astype(
                    np.uint8
                )
                if match_iou > 0.1
                else np.zeros_like(coarse)
            )
        x, y, w, h = mu.toBbox(rle)
        if max(w, h) > self.crop_size * 0.75:
            ys, xs = np.nonzero(coarse)
            chosen = random.randrange(len(xs))
            cx, cy = xs[chosen], ys[chosen]
        else:
            cx, cy = x + w / 2, y + h / 2
        cx += random.uniform(-self.crop_size * 0.12, self.crop_size * 0.12)
        cy += random.uniform(-self.crop_size * 0.12, self.crop_size * 0.12)
        image, _ = crop_square(native_image(row["filename"]), cx, cy, self.crop_size)
        coarse, _ = crop_square(coarse, cx, cy, self.crop_size)
        truth, _ = crop_square(truth, cx, cy, self.crop_size)
        if synthetic or random.random() < 0.25:
            coarse = corrupt_mask(coarse)
        image, masks = augment(image, np.stack((coarse, truth)))
        image = torch.cat(
            (image_tensor(image), torch.from_numpy(masks[0:1].astype(np.float32))),
            dim=0,
        )
        return {
            "image": image,
            "target": torch.from_numpy(masks[1:2].astype(np.float32)),
            "quality": torch.tensor(float(match_iou > 0.3 and masks[1].any())),
        }


class DetailResidualRefiner(torch.nn.Module):
    """Add a native-pixel correction to a contextual encoder-decoder."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.detail = torch.nn.Sequential(
            torch.nn.Conv2d(5, 16, 3, padding=1),
            torch.nn.GroupNorm(4, 16),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1, groups=16),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 1, 1),
        )
        torch.nn.init.zeros_(self.detail[-1].weight)
        torch.nn.init.zeros_(self.detail[-1].bias)

    def forward(self, image):
        logits, quality = self.base(image)
        correction = self.detail(torch.cat((image, logits.sigmoid()), dim=1))
        return logits + correction, quality


def make_refiner(pretrained=True, encoder="resnet18", detail_residual=False):
    import segmentation_models_pytorch as smp

    base = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=4,
        classes=1,
        decoder_channels=(128, 64, 32, 16, 16),
        aux_params={"pooling": "avg", "classes": 1},
    )
    return DetailResidualRefiner(base) if detail_residual else base


def refinement_loss(logits, quality, batch, quality_weight=0.1):
    from torch.nn import functional as F

    logits = logits.float()
    target = batch["target"].float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dice = (
        1
        - (
            (2 * (probability * target).sum((1, 2, 3)) + 1)
            / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        ).mean()
    )
    quality_loss = F.binary_cross_entropy_with_logits(
        quality[:, 0].float(), batch["quality"]
    )
    return bce + dice + quality_weight * quality_loss
