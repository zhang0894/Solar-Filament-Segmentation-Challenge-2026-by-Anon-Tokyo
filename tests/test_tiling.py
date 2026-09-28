import random

import numpy as np
import pytest
import torch

from filament.data import training_crop
from filament.metrics import encode_mask
from filament.tiling import tiled_logits


@pytest.mark.parametrize("mode", ["none", "flip", "d4"])
def test_tiles_preserve_asymmetric_coordinates_through_overlaps_and_edges(mode):
    image = torch.randn(2, 3, 23, 31)

    def model(x):
        return x[:, :1] - 0.5 * x[:, 1:2]

    actual = tiled_logits(model, image, 12, 12, mode)
    torch.testing.assert_close(actual, model(image)[:, 0], atol=1e-6, rtol=1e-5)


def test_resized_tile_predictions_have_no_empty_border_pixels():
    image = torch.zeros(1, 3, 27, 35)
    result = tiled_logits(lambda x: torch.ones_like(x[:, :1]) * 2, image, 16, 24)
    torch.testing.assert_close(result, torch.full_like(result, 2))


def test_crop_keeps_image_mask_alignment_and_does_not_modify_cached_arrays(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0 if a == 0.85 else 0.0)
    rng = np.random.default_rng(3)
    mask = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    image = mask * 255
    targets = np.stack([mask, mask])
    originals = image.copy(), targets.copy()
    for seed in range(10):
        random.seed(seed)
        crop, labels = training_crop(
            image, targets, {"rles": [encode_mask(mask)]}, 32, 32
        )
        assert crop.shape == (32, 32) and labels.shape == (2, 32, 32)
        np.testing.assert_array_equal(crop, labels[0] * 255)
        np.testing.assert_array_equal(labels[0], labels[1])
        assert set(np.unique(labels)).issubset({0, 1})
    np.testing.assert_array_equal(image, originals[0])
    np.testing.assert_array_equal(targets, originals[1])
