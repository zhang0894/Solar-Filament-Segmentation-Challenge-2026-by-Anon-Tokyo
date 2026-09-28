"""Validate removal of disconnected query-mask artifacts using frozen masks."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from filament.data import read_manifest
from filament.metrics import Evaluator
from filament.prediction import disjoint_instances, filter_instances, semantic_instances


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    original = json.loads(Path(a.predictions).read_text())
    rows = {r["image_id"]: r for r in read_manifest()["images"] if r["fold"] == a.fold}
    if set(original) != set(rows):
        raise ValueError("Wrong held-out image coverage")
    results = []
    best = -1
    for mode in ["largest", "split"]:

        def process(name, mode=mode):
            candidates = []
            for item in original[name]:
                mask = mu.decode(item["rle"]).astype(np.float32)
                components = semantic_instances(
                    mask, threshold=0.5, min_area=64, group_radius=8
                )
                if mode == "largest" and components:
                    components = [max(components, key=lambda c: mu.area(c["rle"]))]
                candidates.extend(c | {"score": item["score"]} for c in components)
            return name, filter_instances(candidates, score_threshold=0, min_area=64)

        with ThreadPoolExecutor(max_workers=6) as pool:
            cleaned = dict(pool.map(process, original))
        for threshold in [0.6, 0.75, 0.85, 0.9]:
            for area in [64, 256, 512]:
                selected = {
                    name: disjoint_instances(
                        [
                            p
                            for p in items
                            if p["score"] >= threshold and mu.area(p["rle"]) >= area
                        ],
                        min_area=area,
                    )
                    for name, items in cleaned.items()
                }
                ev = Evaluator()
                for name, row in rows.items():
                    for ann in row["annotators"]:
                        ev.update(
                            name,
                            ann["id"],
                            ann["rles"],
                            [p["rle"] for p in selected[name]],
                        )
                record = {
                    "mode": mode,
                    "threshold": threshold,
                    "min_area": area,
                    **ev.result(),
                }
                results.append(record)
                if record["official"]["pq"] > best:
                    best = record["official"]["pq"]
                    best_predictions = selected
                    best_entries = ev.entries
        print(json.dumps({"mode": mode, "best_pq": best}), flush=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["official"]["pq"], reverse=True)
    (out / "calibration.json").write_text(
        json.dumps({"config": vars(a), "results": results}, indent=2)
    )
    (out / "predictions.json").write_text(json.dumps(best_predictions))
    (out / "entries.json").write_text(json.dumps(best_entries))
    print(json.dumps(results[:3], indent=2))


if __name__ == "__main__":
    main()
