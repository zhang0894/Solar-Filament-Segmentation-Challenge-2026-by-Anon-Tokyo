"""Aggregate disjoint held-out folds under one frozen refinement recipe."""

import argparse
import hashlib
import json
from pathlib import Path

from pycocotools import mask as mu

from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipelines", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-min-area", type=int, default=256)
    p.add_argument("--require-all-folds", action="store_true")
    a = p.parse_args()
    manifest = read_manifest()
    digest = hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest()
    evaluators = {name: Evaluator() for name in ["coarse", "refined"]}
    confirmation = {name: Evaluator() for name in evaluators}
    seen, folds, recipes = set(), [], []
    for directory in a.pipelines:
        root = Path(directory)
        meta = json.loads((root / "final/metadata.json").read_text())
        config = meta["model_config"]
        fold = config["fold"]
        if meta["inference"]["split"] != "valid" or config["manifest_sha256"] != digest:
            raise ValueError(
                "OOF requires held-out predictions from the current manifest"
            )
        if fold in seen:
            raise ValueError("A validation fold was included twice")
        seen.add(fold)
        rows = [r for r in manifest["images"] if r["fold"] == fold]
        coarse = json.loads((root / "proposals/predictions.json").read_text())
        refined = json.loads((root / "final/predictions.json").read_text())
        expected = {r["image_id"] for r in rows}
        if set(coarse) != expected or set(refined) != expected:
            raise ValueError("Incomplete or contaminated held-out image coverage")
        recipe = json.loads((root / "pipeline_config.json").read_text())
        recipes.append({k: recipe[k] for k in ["segmentation", "refinement"]})
        if recipes[-1] != recipes[0]:
            raise ValueError("OOF pipelines must use one frozen postprocessing recipe")
        local = {name: Evaluator() for name in evaluators}
        for row in rows:
            name = row["image_id"]
            candidates = {
                "coarse": [
                    q["rle"]
                    for q in coarse[name]
                    if mu.area(q["rle"]) >= a.baseline_min_area
                ],
                "refined": [q["rle"] for q in refined[name]],
            }
            for ann in row["annotators"]:
                for kind, rles in candidates.items():
                    args = (name, ann["id"], ann["rles"], rles)
                    evaluators[kind].update(*args)
                    local[kind].update(*args)
                    # Fold zero was used to choose this recipe. The remaining
                    # folds provide a separate check of that development gain.
                    if fold != 0:
                        confirmation[kind].update(*args)
        folds.append(
            {
                "fold": fold,
                "images": len(rows),
                **{k: v.result() for k, v in local.items()},
            }
        )
    complete = seen == set(range(5))
    if a.require_all_folds and not complete:
        raise ValueError("All five folds are required")
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "complete_five_fold_oof": complete,
        "folds": folds,
        "recipe": recipes[0],
        "baseline_min_area": a.baseline_min_area,
        "all_available_folds": {k: v.result() for k, v in evaluators.items()},
        "confirmation_folds_excluding_development_fold0": {
            k: v.result() for k, v in confirmation.items()
        },
        "manifest_sha256": digest,
        "pipelines": a.pipelines,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    for kind, ev in evaluators.items():
        (output / f"{kind}_entries.json").write_text(json.dumps(ev.entries))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
