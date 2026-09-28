import torch
from transformers import Mask2FormerConfig

from filament.dense_loss import DenseMaskLoss


def test_dense_supervision_penalizes_broad_masks_and_has_finite_gradient():
    criterion = DenseMaskLoss(Mask2FormerConfig(num_labels=1), {})
    target = torch.zeros(1, 64, 64)
    target[:, 16:48, 28:36] = 1
    exact = torch.full((1, 1, 16, 16), -7.0)
    exact[:, :, 4:12, 7:9] = 7
    exact.requires_grad_()
    broad = torch.full_like(exact, -7.0)
    broad[:, :, 2:14, 4:12] = 7
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    exact_loss = sum(criterion.loss_masks(exact, [target], indices, 1).values())
    broad_loss = sum(criterion.loss_masks(broad, [target], indices, 1).values())
    assert exact_loss < 0.02
    assert broad_loss > 0.5
    exact_loss.backward()
    assert torch.isfinite(exact.grad).all()
