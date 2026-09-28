from copy import deepcopy

import torch
from torch.nn import functional as F

from filament.highres_masks import HighResolutionMaskFeatures


def test_detail_head_starts_as_exact_bilinear_upsampling_and_can_learn():
    torch.manual_seed(32)
    head = HighResolutionMaskFeatures(channels=8)
    clone = deepcopy(head)
    image = torch.randn(2, 3, 32, 32)
    features = torch.randn(2, 8, 8, 8)
    result = head(image, features)
    expected = F.interpolate(
        features, size=(16, 16), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(clone(image, features), result, rtol=0, atol=0)
    result.square().mean().backward()
    assert torch.isfinite(head.project.weight.grad).all()
    assert head.project.weight.grad.abs().sum() > 0
