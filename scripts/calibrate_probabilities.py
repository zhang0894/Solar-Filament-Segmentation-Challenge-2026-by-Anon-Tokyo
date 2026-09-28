"""Calibrate native-resolution masks using held-out probability maps only."""

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator
from filament.prediction import semantic_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directories", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.35, 0.5, 0.65, 0.8]
    )
    parser.add_argument("--areas", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--radii", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    roots = [Path(d) for d in args.directories]
    weights = np.array(args.weights or [1] * len(roots), dtype=float)
    if (
        len(weights) != len(roots)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid ensemble weights")
    weights /= weights.sum()
    manifest = read_manifest()
    rows = [r for r in manifest["images"] if r["fold"] == args.fold]
    expected_hash = hashlib.sha256(
        (ROOT / "data/manifest.json").read_bytes()
    ).hexdigest()
    for root in roots:
        metadata = json.loads((root / "metadata.json").read_text())
        if (
            metadata["model_config"]["fold"] != args.fold
            or metadata["inference"]["split"] != "valid"
            or metadata["model_config"]["manifest_sha256"] != expected_hash
        ):
            raise ValueError("All ensemble models must exclude this validation fold")
    probabilities = {}
    for row in rows:
        values = []
        for root in roots:
            with np.load(root / "probabilities" / f"{row['image_id']}.npz") as d:
                values.append(d["probability"].astype(np.float32))
        # Map mixed-resolution predictions to the common native image grid.
        import cv2

        size = max(v.shape[-1] for v in values)
        values = [
            cv2.resize(v, (size, size), interpolation=cv2.INTER_LINEAR)
            if v.shape[-1] != size
            else v
            for v in values
        ]
        probabilities[row["image_id"]] = sum(w * v for w, v in zip(weights, values))
    results = []
    best_predictions = None
    best_pq = -1
    for threshold, radius in itertools.product(args.thresholds, args.radii):

        def process(row, threshold=threshold, radius=radius):
            instances = semantic_instances(
                probabilities[row["image_id"]],
                threshold=threshold,
                min_area=min(args.areas),
                group_radius=radius,
            )
            for p in instances:
                p["area"] = int(mu.area(p["rle"]))
            return row["image_id"], instances

        with ThreadPoolExecutor(max_workers=8) as pool:
            predictions = dict(pool.map(process, rows))
        for area in args.areas:
            ev = Evaluator()
            for row in rows:
                rles = [
                    p["rle"] for p in predictions[row["image_id"]] if p["area"] >= area
                ]
                for ann in row["annotators"]:
                    ev.update(row["image_id"], ann["id"], ann["rles"], rles)
            result = {
                "threshold": threshold,
                "group_radius": radius,
                "min_area": area,
                **ev.result(),
            }
            results.append(result)
            if result["official"]["pq"] > best_pq:
                best_pq = result["official"]["pq"]
                best_predictions = {
                    name: [p for p in instances if p["area"] >= area]
                    for name, instances in predictions.items()
                }
                best_entries = ev.entries
        print(
            json.dumps(
                {"threshold": threshold, "group_radius": radius, "best_pq": best_pq}
            ),
            flush=True,
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["official"]["pq"], reverse=True)
    (output / "calibration.json").write_text(
        json.dumps(
            {
                "directories": args.directories,
                "weights": weights.tolist(),
                "fold": args.fold,
                "results": results,
            },
            indent=2,
        )
    )
    (output / "predictions.json").write_text(json.dumps(best_predictions))
    (output / "entries.json").write_text(json.dumps(best_entries))
    print(json.dumps(results[:4], indent=2))


if __name__ == "__main__":
    main()
