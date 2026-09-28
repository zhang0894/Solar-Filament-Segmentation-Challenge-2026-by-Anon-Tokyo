"""Audit the official dataset and freeze grouped folds and mask-only targets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from sklearn.model_selection import StratifiedGroupKFold


def build_groups(rows):
    """Group nearby observations (<=72h gaps, <=14d span), then exact duplicates."""
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["image_id"])
    parent = list(range(len(rows)))

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        parent[find(a)] = find(b)

    last = None
    last_time = None
    start_time = None
    hash_first = {}
    for i in ordered:
        now = datetime.strptime(rows[i]["image_id"][:14], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        if (
            last is not None
            and (now - last_time).total_seconds() <= 72 * 3600
            and (now - start_time).days < 14
        ):
            join(i, last)
        else:
            start_time = now
        last, last_time = i, now
        h = rows[i]["sha256"]
        if h in hash_first:
            join(i, hash_first[h])
        hash_first[h] = i
    roots = {k: n for n, k in enumerate(sorted({find(i) for i in range(len(rows))}))}
    return [roots[find(i)] for i in range(len(rows))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / "data/MAGFiLO_1.0_Kaggle_2026"
    annotation_path = source / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data = json.loads(annotation_path.read_text())
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        # Deliberately ignore supplied bbox, spine, area and category metadata.
        annotations[ann["image_id"]].append(ann["segmentation"])
    by_file = defaultdict(list)
    for entry in data["images"]:
        by_file[entry["file_name"]].append(entry)
    cache = root / f"data/cache_{args.size}"
    (cache / "images").mkdir(parents=True, exist_ok=True)
    (cache / "masks").mkdir(exist_ok=True)

    def process(filename):
        entries = by_file[filename]
        path = source / "train/train_images" / filename
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Unexpected image shape: {filename}")
        image_id = path.stem
        cv2.imwrite(
            str(cache / "images" / f"{image_id}.png"),
            cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA),
        )
        row = {
            "image_id": image_id,
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "annotators": [],
        }
        for entry in entries:
            rles, masks, areas = [], [], []
            occupied = np.zeros((2048, 2048), np.uint8)
            overlap_pixels = 0
            for seg in annotations[entry["id"]]:
                if isinstance(seg, list):
                    rle = mu.merge(mu.frPyObjects(seg, 2048, 2048))
                elif isinstance(seg["counts"], list):
                    rle = mu.frPyObjects(seg, 2048, 2048)
                else:
                    rle = seg
                mask = mu.decode(rle).astype(np.uint8)
                overlap_pixels += int(np.count_nonzero(mask & occupied))
                occupied |= mask
                area = int(mask.sum())
                if area == 0:
                    raise ValueError(f"Empty annotated instance: {entry['id']}")
                counts = rle["counts"]
                rles.append(
                    {
                        "size": [2048, 2048],
                        "counts": counts.decode("ascii")
                        if isinstance(counts, bytes)
                        else counts,
                    }
                )
                small = cv2.resize(
                    mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST
                )
                masks.append(small)
                areas.append(area)
            stack = (
                np.stack(masks)
                if masks
                else np.zeros((0, args.size, args.size), np.uint8)
            )
            np.savez_compressed(
                cache / "masks" / f"{entry['id']}.npz",
                packed=np.packbits(stack, axis=-1),
                shape=np.array(stack.shape),
            )
            row["annotators"].append(
                {
                    "id": entry["id"],
                    "rles": rles,
                    "areas": areas,
                    "overlap_pixels": overlap_pixels,
                }
            )
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(process, sorted(by_file))):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(
                    json.dumps({"prepared": i + 1, "total": len(by_file)}), flush=True
                )
    groups = build_groups(rows)
    strata = [
        f"{(int(r['image_id'][:4]) - 2011) // 4}_{min(int(np.mean([len(a['rles']) for a in r['annotators']])) // 5, 3)}"
        for r in rows
    ]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260928)
    for fold, (_, valid) in enumerate(splitter.split(rows, strata, groups)):
        for i in valid:
            rows[i]["fold"] = fold
    for i, row in enumerate(rows):
        row["group"] = groups[i]
    test_files = sorted((source / "test/test_images").glob("*"))
    test_hashes = {
        hashlib.sha256(p.read_bytes()).hexdigest(): p.name for p in test_files
    }
    duplicates = [
        (r["filename"], test_hashes[r["sha256"]])
        for r in rows
        if r["sha256"] in test_hashes
    ]
    for fold in range(5):
        train = {r["group"] for r in rows if r["fold"] != fold}
        valid = {r["group"] for r in rows if r["fold"] == fold}
        assert train.isdisjoint(valid)
    summary = {
        "train_images": len(rows),
        "test_images": len(test_files),
        "annotator_entries": sum(len(r["annotators"]) for r in rows),
        "instances": sum(len(a["rles"]) for r in rows for a in r["annotators"]),
        "groups": len(set(groups)),
        "group_max_images": max(Counter(groups).values()),
        "fold_sizes": dict(Counter(r["fold"] for r in rows)),
        "stations": dict(Counter(r["image_id"][-2:] for r in rows)),
        "train_test_exact_duplicates": duplicates,
        "entries_with_overlapping_masks": sum(
            a["overlap_pixels"] > 0 for r in rows for a in r["annotators"]
        ),
        "area_quantiles": np.quantile(
            [v for r in rows for a in r["annotators"] for v in a["areas"]],
            [0, 0.1, 0.5, 0.9, 1],
        ).tolist(),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    }
    manifest = {
        "version": 1,
        "seed": 20260928,
        "cache_size": args.size,
        "summary": summary,
        "images": rows,
    }
    (root / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "data/audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
