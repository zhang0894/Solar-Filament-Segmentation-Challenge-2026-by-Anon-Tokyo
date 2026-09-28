"""Strict native-size CSV export with reproducible provenance."""

import csv
import hashlib

from filament.metrics import decode_mask


def export_submission(predictions, path, expected_ids):
    if set(predictions) != set(expected_ids):
        raise ValueError("Predictions must cover exactly the supplied test images")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for name in sorted(predictions):
            for j, pred in enumerate(predictions[name], 1):
                rle = pred["rle"]
                if rle["size"] != [2048, 2048] or not decode_mask(rle).any():
                    raise ValueError("Invalid submission instance")
                writer.writerow([f"{name}_{j}", rle["counts"]])
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != sum(map(len, predictions.values())) or len(
        {r["filament_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Submission serialization lost or duplicated instances")
    for row in rows:
        if not decode_mask(
            {"size": [2048, 2048], "counts": row["segmentation_rle"]}
        ).any():
            raise ValueError("Invalid serialized mask")
    return {
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instances": len(rows),
        "images": len(predictions),
        "empty_images": [k for k, v in predictions.items() if not v],
    }
