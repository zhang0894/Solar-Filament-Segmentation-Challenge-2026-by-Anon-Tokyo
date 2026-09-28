"""Ensemble cached probabilities; test parameters must be frozen on validation."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator
from filament.prediction import semantic_instances
from filament.submission import export_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directories", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-area", type=int, required=True)
    p.add_argument("--group-radius", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-probs", action="store_true")
    a = p.parse_args()
    cv2.setNumThreads(1)
    roots = [Path(d) for d in a.directories]
    metadata = [json.loads((root / "metadata.json").read_text()) for root in roots]
    splits = {m["inference"]["split"] for m in metadata}
    if len(splits) != 1 or not splits <= {"test", "valid"}:
        raise ValueError("Ensemble requires one common inference split")
    split = splits.pop()
    hashes = {m["model_config"]["manifest_sha256"] for m in metadata}
    expected_hash = hashlib.sha256(
        (ROOT / "data/manifest.json").read_bytes()
    ).hexdigest()
    if hashes != {expected_hash}:
        raise ValueError("Mixed data provenance")
    folds = {m["model_config"]["fold"] for m in metadata}
    if split == "valid" and len(folds) != 1:
        raise ValueError("All validation models must exclude the same fold")
    weights = np.array(a.weights or [1] * len(roots), float)
    if (
        len(weights) != len(roots)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid ensemble weights")
    weights /= weights.sum()
    if split == "test":
        names = sorted(
            p.stem
            for p in (ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob(
                "*.jpeg"
            )
        )
    else:
        rows = [r for r in read_manifest()["images"] if r["fold"] in folds]
        names = [r["image_id"] for r in rows]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.cache_probs:
        (out / "probabilities").mkdir(exist_ok=True)

    def process(name):
        values = []
        for root in roots:
            with np.load(root / "probabilities" / f"{name}.npz") as d:
                value = d["probability"].astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite probability")
            values.append(value)
        size = max(v.shape[-1] for v in values)
        value = sum(
            w
            * (
                cv2.resize(v, (size, size), interpolation=cv2.INTER_LINEAR)
                if v.shape[-1] != size
                else v
            )
            for w, v in zip(weights, values)
        )
        if a.cache_probs:
            np.savez_compressed(
                out / "probabilities" / f"{name}.npz",
                probability=value.astype(np.float16),
            )
        return name, semantic_instances(
            value,
            threshold=a.threshold,
            min_area=a.min_area,
            group_radius=a.group_radius,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        predictions = dict(pool.map(process, names))
    result = {
        "ensemble": vars(a),
        "sources": metadata,
        "model_config": metadata[0]["model_config"],
        "inference": {"split": split},
        "images": len(names),
    }
    if split == "test":
        result.update(export_submission(predictions, out / "submission.csv", names))
    else:
        ev = Evaluator()
        for row in rows:
            for ann in row["annotators"]:
                ev.update(
                    row["image_id"],
                    ann["id"],
                    ann["rles"],
                    [p["rle"] for p in predictions[row["image_id"]]],
                )
        result["evaluation"] = ev.result()
        (out / "entries.json").write_text(json.dumps(ev.entries))
    (out / "predictions.json").write_text(json.dumps(predictions))
    (out / "metadata.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["sources", "model_config"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
