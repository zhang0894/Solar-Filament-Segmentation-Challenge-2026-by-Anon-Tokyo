import csv

import numpy as np
import pytest

from filament.metrics import encode_mask
from scripts.audit_submission import audit


def write(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(records)


def test_audit_rejects_cross_instance_overlap(tmp_path):
    mask = np.zeros((2048, 2048), np.uint8)
    mask[5:10, 7:14] = 1
    rle = encode_mask(mask)["counts"]
    file = tmp_path / "submission.csv"
    write(file, [("image_1", rle), ("image_2", rle)])
    with pytest.raises(ValueError, match="Overlapping"):
        audit(file, {"image"})


def test_audit_accounts_for_empty_observations_without_fake_masks(tmp_path):
    mask = np.zeros((2048, 2048), np.uint8)
    mask[5:10, 7:14] = 1
    file = tmp_path / "submission.csv"
    write(file, [("image_1", encode_mask(mask)["counts"])])
    result = audit(file, {"image", "empty"})
    assert result["instances"] == 1
    assert result["empty_observations"] == ["empty"]
    assert result["overlapping_pixels"] == 0
