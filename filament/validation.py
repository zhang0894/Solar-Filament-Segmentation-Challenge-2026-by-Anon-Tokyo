"""Full-resolution, all-annotator evaluation for a frozen observation fold."""

import json
from concurrent.futures import ThreadPoolExecutor

import torch

from filament.metrics import Evaluator
from filament.performance import CUDAPrefetcher
from filament.prediction import (
    filter_instances,
    query_candidates_cpu,
    rcnn_candidates_cpu,
    semantic_instances,
)


@torch.no_grad()
def collect_predictions(
    model, loader, kind, channels_last=False, tile_size=0, model_size=1024
):
    model.eval()
    raw = {}
    for batch in CUDAPrefetcher(loader, channels_last=channels_last):
        image = batch["image"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if tile_size:
                from filament.tiling import tiled_logits

                output = tiled_logits(model, image, tile_size, model_size)[:, None]
            else:
                output = (
                    model(image) if kind == "semantic" else model(pixel_values=image)
                )
        if kind == "semantic":
            probabilities = output[:, 0].float().sigmoid().cpu().numpy()
            for name, probability in zip(batch["image_id"], probabilities):
                raw[name] = probability
        elif kind == "rcnn":
            for name, prediction in zip(batch["image_id"], output):
                raw[name] = rcnn_candidates_cpu(prediction)
        else:
            masks = output.masks_queries_logits.to(torch.float16).cpu().numpy()
            classes = output.class_queries_logits.float().cpu().numpy()
            for i, name in enumerate(batch["image_id"]):
                raw[name] = {"mask_logits": masks[i], "class_logits": classes[i]}
    return raw


def score_predictions(raw, rows, kind, thresholds=None, save_dir=None, workers=4):
    """CPU-only scoring, safe to overlap with the next GPU training epoch."""
    thresholds = thresholds or (
        [0.35, 0.5, 0.65] if kind == "semantic" else [0.2, 0.4, 0.6]
    )
    rows = {r["image_id"]: r for r in rows}
    if kind == "instance":

        def decode(name):
            value = raw[name]
            return name, query_candidates_cpu(value) if isinstance(
                value, dict
            ) else value

        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded = dict(pool.map(decode, raw))
        raw = decoded
    results = []
    all_predictions = {}
    for threshold in thresholds:
        ev = Evaluator()

        def process(name, threshold=threshold):
            if kind == "semantic":
                pred = semantic_instances(raw[name], threshold=threshold)
            else:
                pred = filter_instances(raw[name], score_threshold=threshold)
            return name, pred

        with ThreadPoolExecutor(max_workers=workers) as pool:
            predictions = dict(pool.map(process, raw))
        for name, pred in predictions.items():
            for ann in rows[name]["annotators"]:
                ev.update(name, ann["id"], ann["rles"], [p["rle"] for p in pred])
        result = {"threshold": threshold, **ev.result()}
        results.append(result)
        all_predictions[threshold] = (predictions, ev.entries)
    best = max(results, key=lambda x: x["official"]["pq"])
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "validation.json").write_text(json.dumps(results, indent=2))
        predictions, entries = all_predictions[best["threshold"]]
        (save_dir / "predictions.json").write_text(json.dumps(predictions))
        (save_dir / "entries.json").write_text(json.dumps(entries))
    return best, results


def validate(model, loader, kind, thresholds=None, save_dir=None):
    raw = collect_predictions(model, loader, kind)
    return score_predictions(raw, loader.dataset.rows, kind, thresholds, save_dir)
