import numpy as np
import torch

from filament.data import component_weights
from filament.models import semantic_loss


def test_small_foreground_receives_more_gradient_without_reweighting_background():
    target = np.zeros((32, 32), np.uint8)
    target[3:5, 3:5] = 1
    target[15:27, 15:27] = 1
    weights = component_weights(target)
    assert np.all(weights[target == 0] == 1)
    np.testing.assert_allclose(weights[target != 0].mean(), 1, rtol=1e-6)
    labels = torch.from_numpy(
        np.stack([target, np.zeros_like(target), weights])[None].astype(np.float32)
    )
    logits = torch.zeros(1, 2, 32, 32, requires_grad=True)
    semantic_loss(logits, labels).backward()
    assert logits.grad[0, 0, 3, 3].abs() > 2 * logits.grad[0, 0, 20, 20].abs()
