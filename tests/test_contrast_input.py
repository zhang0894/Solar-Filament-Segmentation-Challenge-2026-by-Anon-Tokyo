import numpy as np
import torch

from filament.data import FilamentDataset, image_tensor
from scripts.predict import InferenceDataset


def test_contrast_validation_and_inference_use_identical_image_features():
    validation = FilamentDataset(
        fold=0, train=False, kind="image", size=1024, limit=1, contrast_input=True
    )[0]
    inference = InferenceDataset("valid", 0, 1024, contrast_input=True)[0]
    assert validation["image_id"] == inference["image_id"]
    torch.testing.assert_close(validation["image"], inference["image"], rtol=0, atol=0)


def test_contrast_preserves_original_intensity_and_handles_black_background():
    for value in [0.0, 0.5, 1.0]:
        image = np.full((64, 64), value, np.float32)
        raw, contrast = image_tensor(image), image_tensor(image, True)
        assert torch.isfinite(contrast).all()
        torch.testing.assert_close(raw[0], contrast[0], rtol=0, atol=0)
