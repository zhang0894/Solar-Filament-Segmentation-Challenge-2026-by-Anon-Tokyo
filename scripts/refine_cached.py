"""Apply frozen refinement/quality settings to cached proposal crops."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator, encode_crop_mask
from filament.prediction import disjoint_instances, filter_instances
from filament.submission import export_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--proposals", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["mask", "quality"], default="mask")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-quality", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=64)
    p.add_argument("--blend", type=float, default=1.0)
    a = p.parse_args()
    if not 0 <= a.blend <= 1:
        p.error("blend must be in [0,1]")
    cv2.setNumThreads(1)
    cache, base = Path(a.cache), Path(a.proposals)
    metadata = json.loads((cache / "metadata.json").read_text())
    source = base / "predictions.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["proposal_sha256"]:
        raise ValueError("Cached crops do not belong to these exact proposals")
    predictions = json.loads(source.read_text())

    def process(name):
        candidates = []
        for index, original in enumerate(predictions[name]):
            with np.load(cache / "probabilities" / f"{name}_{index}.npz") as d:
                quality = float(d["quality"])
                if quality < a.min_quality:
                    continue
                if a.mode == "quality":
                    rle = original["rle"]
                else:
                    probability = d["probability"].astype(np.float32)
                    x0, y0, x1, y1 = d["box"].astype(int)
                    if a.blend < 1:
                        coarse = mu.decode(original["rle"])[y0:y1, x0:x1].astype(
                            np.float32
                        )
                        prior = cv2.GaussianBlur(coarse, (0, 0), 2)
                        probability = a.blend * probability + (1 - a.blend) * prior
                    rle = encode_crop_mask(probability > a.threshold, int(x0), int(y0))
            if int(mu.area(rle)) >= a.min_area:
                candidates.append(
                    {"rle": rle, "score": original["score"], "quality": quality}
                )
        return name, disjoint_instances(
            filter_instances(candidates, score_threshold=0, min_area=a.min_area),
            min_area=a.min_area,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        result = dict(pool.map(process, predictions))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata["refinement_inference"] = vars(a)
    metadata.update(
        images=len(result),
        instances=sum(map(len, result.values())),
        empty_images=[name for name, items in result.items() if not items],
    )
    split = metadata["inference"]["split"]
    if split == "valid":
        rows = [
            r
            for r in read_manifest()["images"]
            if r["fold"] == metadata["model_config"]["fold"]
        ]
        if set(result) != {r["image_id"] for r in rows}:
            raise ValueError("Wrong validation coverage")
        ev = Evaluator()
        for row in rows:
            for ann in row["annotators"]:
                ev.update(
                    row["image_id"],
                    ann["id"],
                    ann["rles"],
                    [p["rle"] for p in result[row["image_id"]]],
                )
        metadata["evaluation"] = ev.result()
        (out / "entries.json").write_text(json.dumps(ev.entries))
    elif split == "test":
        names = [
            p.stem
            for p in (ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob(
                "*.jpeg"
            )
        ]
        metadata.update(export_submission(result, out / "submission.csv", names))
    else:
        raise ValueError("Expected validation or test inference")
    (out / "predictions.json").write_text(json.dumps(result))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in ["evaluation", "csv_sha256", "instances"]
                if k in metadata
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
