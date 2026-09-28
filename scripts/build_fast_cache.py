"""CPU-only cache construction: shared mmap images and semantic targets.

This removes JPEG/PNG/NPZ decoding and per-instance morphology from each epoch.
The cache is derived solely from the existing fold manifest and training masks.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

from filament.data import ROOT, load_masks, read_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--pack-targets", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    manifest = read_manifest()
    size = args.size or manifest["cache_size"]
    directory = ROOT / f"data/cache_{size}/fast"
    directory.mkdir(exist_ok=True, parents=True)
    rows = manifest["images"]
    annotations = [a for row in rows for a in row["annotators"]]
    image_path = directory / "images.partial.npy"
    target_path = directory / "targets.partial.npy"
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(len(rows), size, size)
    )
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            len(annotations),
            2,
            size,
            (size + 7) // 8 if args.pack_targets else size,
        ),
    )

    def prepare_image(pair):
        i, row = pair
        if size == manifest["cache_size"]:
            image = cv2.imread(
                str(ROOT / f"data/cache_{size}/images/{row['image_id']}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
        else:
            image = cv2.imread(
                str(
                    ROOT
                    / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
                    / row["filename"]
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        images[i] = image

    def prepare_target(pair):
        i, ann = pair
        if size == manifest["cache_size"]:
            masks = load_masks(ann["id"], size)
        else:
            native = mu.decode(ann["rles"]).transpose(2, 0, 1)
            masks = np.stack(
                [
                    cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                    for m in native
                ]
            )
        union = masks.max(axis=0)
        boundary = np.zeros_like(union)
        for mask in masks:
            boundary |= cv2.morphologyEx(
                mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            )
        target = np.stack((union, boundary))
        targets[i] = np.packbits(target, axis=-1) if args.pack_targets else target

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(prepare_image, enumerate(rows)))
        list(pool.map(prepare_target, enumerate(annotations)))
    images.flush()
    targets.flush()
    image_path.replace(directory / "images.npy")
    target_path.replace(directory / "targets.npy")
    metadata = {
        "version": 1,
        "manifest_sha256": hashlib.sha256(
            (ROOT / "data/manifest.json").read_bytes()
        ).hexdigest(),
        "images": {r["image_id"]: i for i, r in enumerate(rows)},
        "annotations": {a["id"]: i for i, a in enumerate(annotations)},
        "size": size,
        "targets_packed": args.pack_targets,
    }
    temp = directory / "metadata.partial.json"
    temp.write_text(json.dumps(metadata))
    temp.replace(directory / "metadata.json")
    print(
        json.dumps(
            {
                "event": "fast_cache_ready",
                "images": len(rows),
                "annotations": len(annotations),
                "bytes": sum(p.stat().st_size for p in directory.glob("*.npy")),
            }
        )
    )


if __name__ == "__main__":
    main()
