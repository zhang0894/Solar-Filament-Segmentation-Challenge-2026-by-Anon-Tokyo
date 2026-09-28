"""Held-out test of whether an independent detector recovers missing instances."""

import argparse
import hashlib
import json
from pathlib import Path

from pycocotools import mask as mu

from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator
from filament.prediction import disjoint_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [r for r in read_manifest()["images"] if r["fold"] == args.fold]
    expected = {r["image_id"] for r in rows}
    digest = hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest()
    data = []
    for directory in [args.primary, args.secondary]:
        base = Path(directory)
        meta = json.loads((base / "metadata.json").read_text())
        if meta["inference"]["split"] != "valid":
            raise ValueError("This probe only accepts held-out predictions")
        if (
            meta["model_config"]["fold"] != args.fold
            or meta["model_config"]["manifest_sha256"] != digest
        ):
            raise ValueError("Prediction provenance does not match the held-out fold")
        pred = json.loads((base / "predictions.json").read_text())
        if set(pred) != expected:
            raise ValueError("Incomplete held-out predictions")
        data.append(pred)
    primary, secondary = data
    additions = {}
    for name in primary:
        occupied_rles = [p["rle"] for p in primary[name]]
        occupied = mu.merge(occupied_rles) if occupied_rles else None
        additions[name] = []
        for candidate in secondary[name]:
            area = int(mu.area(candidate["rle"]))
            intersection = (
                float(mu.area(mu.merge([occupied, candidate["rle"]], intersect=True)))
                if occupied is not None
                else 0
            )
            # At most 20% of the supplementary instance may already be covered.
            # This is a conservative recall probe, not a mask replacement rule.
            if intersection / max(area, 1) <= 0.2:
                additions[name].append(candidate | {"area": area})
    results = []
    best = None
    for threshold in [0.6, 0.75, 0.9, 1.01]:
        for min_area in [64, 256]:
            evaluator = Evaluator()
            predicted = {}
            n_added = 0
            for row in rows:
                name = row["image_id"]
                candidates = [
                    p | {"original_score": p["score"], "score": p["score"] + 2}
                    for p in primary[name]
                ]
                extra = [
                    p
                    for p in additions[name]
                    if p["score"] >= threshold and p["area"] >= min_area
                ]
                result = disjoint_instances(candidates + extra, min_area=64)
                for p in result:
                    if "original_score" in p:
                        p["score"] = p.pop("original_score")
                n_added += len(result) - len(primary[name])
                predicted[name] = result
                for ann in row["annotators"]:
                    evaluator.update(
                        name, ann["id"], ann["rles"], [p["rle"] for p in result]
                    )
            score = {
                "threshold": threshold,
                "min_area": min_area,
                "max_existing_coverage": 0.2,
                "added_instances": n_added,
                **evaluator.result(),
            }
            results.append(score)
            if best is None or score["official"]["pq"] > best["official"]["pq"]:
                best, best_predictions, best_entries = (
                    score,
                    predicted,
                    evaluator.entries,
                )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "results": results,
                "best": best,
                "note": "Development-only selection; confirmation on another fold is required.",
            },
            indent=2,
        )
    )
    (out / "predictions.json").write_text(json.dumps(best_predictions))
    (out / "entries.json").write_text(json.dumps(best_entries))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
