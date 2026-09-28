"""Independently validate every native mask and overlap before submission."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu


def audit(csv_path, expected_ids):
    grouped = defaultdict(list)
    seen = set()
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Incorrect submission columns")
        for row in reader:
            identifier = row["filament_id"]
            if identifier in seen:
                raise ValueError("Duplicate filament ID")
            seen.add(identifier)
            name, suffix = identifier.rsplit("_", 1)
            if name not in expected_ids or not suffix:
                raise ValueError("Unexpected observation ID")
            grouped[name].append(row["segmentation_rle"])
    for name, values in grouped.items():
        occupied = np.zeros((2048, 2048), dtype=bool)
        for counts in values:
            mask = mu.decode({"size": [2048, 2048], "counts": counts}).astype(bool)
            if mask.shape != occupied.shape or not mask.any():
                raise ValueError(f"Empty or invalid instance in {name}")
            if (mask & occupied).any():
                raise ValueError(f"Overlapping instances in {name}")
            occupied |= mask
    return {
        "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "test_observations": len(expected_ids),
        "observations_with_predictions": len(grouped),
        "empty_observations": sorted(set(expected_ids) - set(grouped)),
        "instances": len(seen),
        "overlapping_pixels": 0,
        "valid": True,
        "empty_policy": "No row for observations with zero predicted instances; no fabricated mask.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument(
        "--test-images", default="data/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = {p.stem for p in Path(args.test_images).glob("*.jpeg")}
    if not expected:
        raise ValueError("No test images found")
    result = audit(args.submission, expected)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
