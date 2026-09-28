"""Small, explicit validation-only search for size/confidence filtering."""

import argparse
import itertools
import json
from pathlib import Path

from pycocotools import mask as mu

from filament.data import read_manifest
from filament.metrics import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--areas", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--scores", type=float, nargs="+", default=[0, 0.75, 0.85, 0.9])
    p.add_argument("--score-field", choices=["score", "quality"], default="score")
    p.add_argument("--export-best", default="")
    args = p.parse_args()
    predictions = json.loads(Path(args.predictions).read_text())
    rows = [r for r in read_manifest()["images"] if r["fold"] == args.fold]
    if set(predictions) != {r["image_id"] for r in rows}:
        raise ValueError("Calibration predictions must cover exactly the held-out fold")
    for candidates in predictions.values():
        for candidate in candidates:
            candidate["area"] = int(mu.area(candidate["rle"]))
    results = []
    best_pq = -1
    for area, score in itertools.product(args.areas, args.scores):
        ev = Evaluator()
        for row in rows:
            pred = [
                p["rle"]
                for p in predictions[row["image_id"]]
                if p["area"] >= area and p[args.score_field] >= score
            ]
            for ann in row["annotators"]:
                ev.update(row["image_id"], ann["id"], ann["rles"], pred)
        result = {
            "min_area": area,
            "min_score": score,
            "score_field": args.score_field,
            **ev.result(),
        }
        results.append(result)
        if result["official"]["pq"] > best_pq:
            best_pq = result["official"]["pq"]
            best_predictions = {
                name: [
                    p
                    for p in items
                    if p["area"] >= area and p[args.score_field] >= score
                ]
                for name, items in predictions.items()
            }
            best_entries = ev.entries
    results.sort(key=lambda x: x["official"]["pq"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(
        json.dumps(
            {"source": args.predictions, "fold": args.fold, "results": results},
            indent=2,
        )
    )
    print(json.dumps(results[:5], indent=2))
    if args.export_best:
        directory = Path(args.export_best)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "predictions.json").write_text(json.dumps(best_predictions))
        (directory / "entries.json").write_text(json.dumps(best_entries))


if __name__ == "__main__":
    main()
