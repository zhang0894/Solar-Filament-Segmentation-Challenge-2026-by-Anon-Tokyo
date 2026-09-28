"""Evaluate a fold-isolated proposal quality regressor before using it."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
from pycocotools import mask as mu
from sklearn.ensemble import ExtraTreesRegressor

from filament.candidate_features import FEATURE_NAMES, candidate_features
from filament.data import ROOT, read_manifest
from filament.metrics import Evaluator, pairwise_iou


def extract(directory, rows, training):
    metadata = json.loads((directory / "metadata.json").read_text())
    expected_split = "train" if training else "valid"
    if metadata["inference"]["split"] != expected_split:
        raise ValueError("Wrong prediction split")
    predictions = json.loads((directory / "predictions.json").read_text())
    if set(predictions) != set(rows):
        raise ValueError(
            "Predictions do not match the required outer-fold observations"
        )

    def process(name):
        row = rows[name]
        image = cv2.imread(
            str(
                ROOT
                / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
                / row["filename"]
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        pred = predictions[name]
        features = [candidate_features(image, p["rle"]) for p in pred]
        rewards = np.zeros(len(pred))
        for ann in row["annotators"]:
            ious = pairwise_iou(ann["rles"], [p["rle"] for p in pred])
            rewards += (ious * (ious > 0.5)).max(0) if len(pred) else np.zeros(0)
        rewards /= len(row["annotators"])
        return name, features, rewards

    index, X, y = [], [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, features, rewards in pool.map(process, predictions):
            index.extend((name, i) for i in range(len(features)))
            X.extend(features)
            y.extend(rewards)
    return predictions, index, np.asarray(X), np.asarray(y), metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    manifest = read_manifest()
    training = {r["image_id"]: r for r in manifest["images"] if r["fold"] != a.fold}
    valid = {r["image_id"]: r for r in manifest["images"] if r["fold"] == a.fold}
    _, _, X, y, train_meta = extract(Path(a.train), training, True)
    predictions, index, Xv, yv, val_meta = extract(Path(a.valid), valid, False)
    if any(m["model_config"]["fold"] != a.fold for m in [train_meta, val_meta]):
        raise ValueError("Proposal models must share the excluded fold")
    model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=15,
        max_depth=12,
        max_features=0.8,
        n_jobs=6,
        random_state=20260928,
    )
    model.fit(X, y)
    scores = model.predict(Xv)
    for (name, i), score in zip(index, scores):
        predictions[name][i].update(
            quality=float(score), area=int(mu.area(predictions[name][i]["rle"]))
        )
    results = []
    best = -1
    for threshold in [0, 0.1, 0.2, 0.3]:
        for area in [64, 256, 512]:
            filtered = {
                name: [
                    p for p in pred if p["quality"] >= threshold and p["area"] >= area
                ]
                for name, pred in predictions.items()
            }
            ev = Evaluator()
            for name, row in valid.items():
                for ann in row["annotators"]:
                    ev.update(
                        name, ann["id"], ann["rles"], [p["rle"] for p in filtered[name]]
                    )
            result = {"quality_threshold": threshold, "min_area": area, **ev.result()}
            results.append(result)
            if result["official"]["pq"] > best:
                best = result["official"]["pq"]
                best_predictions = filtered
                best_entries = ev.entries
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "quality_model.joblib")
    np.savez_compressed(
        out / "features.npz",
        train_X=X,
        train_y=y,
        valid_X=Xv,
        valid_y=yv,
        valid_score=scores,
    )
    results.sort(key=lambda r: r["official"]["pq"], reverse=True)
    (out / "calibration.json").write_text(
        json.dumps(
            {
                "config": vars(a),
                "features": FEATURE_NAMES,
                "importance": model.feature_importances_.tolist(),
                "results": results,
            },
            indent=2,
        )
    )
    (out / "predictions.json").write_text(json.dumps(best_predictions))
    (out / "entries.json").write_text(json.dumps(best_entries))
    print(json.dumps(results[:3], indent=2), flush=True)


if __name__ == "__main__":
    main()
