"""Test confidence-seeded grouping of faint, disconnected filament parts."""

import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from filament.data import read_manifest
from filament.metrics import Evaluator
from filament.prediction import semantic_instances


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    root = Path(a.directory)
    meta = json.loads((root / "metadata.json").read_text())
    if meta["inference"]["split"] != "valid":
        raise ValueError("Calibrate only on held-out data")
    rows = [
        r
        for r in read_manifest()["images"]
        if r["fold"] == meta["model_config"]["fold"]
    ]
    probabilities = {}
    for row in rows:
        name = row["image_id"]
        with np.load(root / "probabilities" / f"{name}.npz") as d:
            probabilities[name] = d["probability"].astype(np.float32)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    best, results = -1, []
    for threshold, connect in itertools.product([0.35, 0.5], [0.1, 0.2, 0.35]):
        settings = {
            "threshold": threshold,
            "connect_threshold": connect,
            "seed_threshold": 0.65,
            "seed_min_area": 64,
            "group_radius": 8,
        }

        def process(row, settings=settings):
            name = row["image_id"]
            return name, semantic_instances(
                probabilities[name], min_area=64, **settings
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            predictions = dict(pool.map(process, rows))
        for area in [64, 256]:
            ev = Evaluator()
            filtered = {
                name: [q for q in items if mu.area(q["rle"]) >= area]
                for name, items in predictions.items()
            }
            for row in rows:
                for ann in row["annotators"]:
                    ev.update(
                        row["image_id"],
                        ann["id"],
                        ann["rles"],
                        [q["rle"] for q in filtered[row["image_id"]]],
                    )
            result = {**settings, "min_area": area, **ev.result()}
            results.append(result)
            if result["official"]["pq"] > best:
                best = result["official"]["pq"]
                (output / "predictions.json").write_text(json.dumps(filtered))
                (output / "entries.json").write_text(json.dumps(ev.entries))
                (output / "metadata.json").write_text(
                    json.dumps(
                        meta
                        | {
                            "hysteresis": settings | {"min_area": area},
                            "evaluation": ev.result(),
                        },
                        indent=2,
                    )
                )
        print(json.dumps({**settings, "best_pq": best}), flush=True)
    results.sort(key=lambda r: r["official"]["pq"], reverse=True)
    (output / "calibration.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
