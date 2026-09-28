"""Run a frozen semantic ensemble and same-proposal refiner ensemble."""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from filament.data import ROOT


def normalized_weights(items):
    weights = np.array([item.get("weight", 1.0) for item in items], dtype=np.float64)
    if (
        not len(weights)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid ensemble weights")
    return weights / weights.sum()


def validate_refiner_cache(cache, proposals, split):
    """Reject stale crops and held-out folds seen by a cached refiner."""
    meta = json.loads((cache / "metadata.json").read_text())
    source = json.loads((proposals / "metadata.json").read_text())
    digest = hashlib.sha256((proposals / "predictions.json").read_bytes()).hexdigest()
    if meta.get("proposal_sha256") != digest:
        raise ValueError("Cached refiner saw different proposals")
    if meta["inference"]["split"] != split:
        raise ValueError("Cached refiner has wrong inference split")
    config = meta["refiner_config"]
    expected = source["model_config"]
    if config["manifest_sha256"] != expected["manifest_sha256"]:
        raise ValueError("Cached refiner has different data provenance")
    if split == "valid" and config["fold"] != expected["fold"]:
        raise ValueError("Cached refiner trained on this validation fold")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    deterministic_args = ["--deterministic"] if config.get("deterministic") else []
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_config.json").write_text(json.dumps(config, indent=2))

    def run(module, args, label):
        print(json.dumps({"event": "stage_start", "stage": label}), flush=True)
        with (out / f"{label}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-u", "-m", module, *map(str, args)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(json.dumps({"event": "stage_complete", "stage": label}), flush=True)

    directories = []
    for i, item in enumerate(config["models"]):
        if "directory" in item:
            directory = Path(item["directory"])
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["inference"]["split"] != a.split:
                raise ValueError("Cached model has wrong inference split")
        else:
            directory = out / f"model_{i}"
            run(
                "scripts.predict",
                [
                    "--checkpoint",
                    item["checkpoint"],
                    "--split",
                    a.split,
                    "--output",
                    directory,
                    "--tta",
                    item.get("tta", "flip"),
                    "--batch",
                    item.get("batch", 2),
                    "--cache-probs",
                    *deterministic_args,
                ],
                f"global_{i}",
            )
        directories.append(str(directory))
    global_weights = normalized_weights(config["models"])
    seg = config["segmentation"]
    proposals = out / "proposals"
    run(
        "scripts.ensemble",
        [
            "--directories",
            *directories,
            "--weights",
            *global_weights,
            "--threshold",
            seg["threshold"],
            "--group-radius",
            seg.get("group_radius", 0),
            "--min-area",
            seg.get("min_area", 64),
            "--output",
            proposals,
        ],
        "proposals",
    )
    if not config.get("refiners"):
        print(json.dumps({"final_directory": str(proposals)}), flush=True)
        return
    caches = []
    for i, item in enumerate(config["refiners"]):
        if "directory" in item:
            cache = Path(item["directory"])
            validate_refiner_cache(cache, proposals, a.split)
            caches.append(cache)
            continue
        cache = out / f"refiner_{i}"
        run(
            "scripts.predict_refiner",
            [
                "--checkpoint",
                item["checkpoint"],
                "--predictions",
                proposals,
                "--output",
                cache,
                "--batch",
                item.get("batch", 16),
                "--tta",
                item.get("tta", "none"),
                *deterministic_args,
            ],
            f"refiner_{i}",
        )
        validate_refiner_cache(cache, proposals, a.split)
        caches.append(cache)
    weights = normalized_weights(config["refiners"])
    if len(caches) == 1:
        merged = caches[0]
    else:
        merged = out / "refiner_ensemble"
        (merged / "probabilities").mkdir(parents=True, exist_ok=True)
        metas = [json.loads((c / "metadata.json").read_text()) for c in caches]
        if len({m["proposal_sha256"] for m in metas}) != 1:
            raise ValueError("Refiners saw different proposals")
        files = sorted((caches[0] / "probabilities").glob("*.npz"))

        def combine(file):
            probability = None
            quality = 0.0
            box = None
            score = None
            for weight, cache in zip(weights, caches):
                with np.load(cache / "probabilities" / file.name) as d:
                    if box is not None and not np.array_equal(box, d["box"]):
                        raise ValueError("Refinement crops are misaligned")
                    box = d["box"]
                    score = float(d["score"])
                    value = d["probability"].astype(np.float32)
                    probability = (
                        weight * value
                        if probability is None
                        else probability + weight * value
                    )
                    quality += weight * float(d["quality"])
            np.savez_compressed(
                merged / "probabilities" / file.name,
                probability=probability.astype(np.float16),
                box=box,
                quality=quality,
                score=score,
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(combine, files))
        meta = metas[0] | {
            "refiner_ensemble": metas,
            "refiner_weights": weights.tolist(),
        }
        (merged / "metadata.json").write_text(json.dumps(meta, indent=2))
    ref = config["refinement"]
    run(
        "scripts.refine_cached",
        [
            "--cache",
            merged,
            "--proposals",
            proposals,
            "--mode",
            ref.get("mode", "mask"),
            "--threshold",
            ref.get("threshold", 0.5),
            "--blend",
            ref.get("blend", 0.5),
            "--min-quality",
            ref.get("min_quality", 0.5),
            "--min-area",
            ref.get("min_area", 64),
            "--output",
            out / "final",
        ],
        "export",
    )
    print(json.dumps({"final_directory": str(out / "final")}), flush=True)


if __name__ == "__main__":
    main()
