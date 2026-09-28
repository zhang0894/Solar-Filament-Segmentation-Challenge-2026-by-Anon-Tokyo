import pytest
import torch

from filament.tta import refiner_tta, semantic_tta


@pytest.mark.parametrize("mode", ["none", "flip", "d4"])
def test_semantic_tta_restores_asymmetric_image_coordinates(mode):
    image = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)

    def pointwise_model(x):
        return torch.stack((x[:, 0] - 0.25 * x[:, 1], x[:, 2]), dim=1)

    expected = pointwise_model(image)[:, 0]
    torch.testing.assert_close(semantic_tta(pointwise_model, image, mode), expected)


@pytest.mark.parametrize("mode", ["none", "flip", "d4"])
def test_refiner_tta_restores_masks_and_preserves_global_quality(mode):
    image = torch.arange(2 * 4 * 5 * 7, dtype=torch.float32).reshape(2, 4, 5, 7)

    def model(x):
        return x[:, :1] - x[:, 3:4], x.mean((1, 2, 3))[:, None]

    expected = model(image)
    actual = refiner_tta(model, image, mode)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
