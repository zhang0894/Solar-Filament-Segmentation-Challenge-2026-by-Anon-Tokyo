"""Unbiased dense supervision at the query-mask output resolution.

Area downsampling retains subpixel filament coverage. Matching may sample
foreground to locate tiny objects, but mask optimization covers every pixel.
"""

from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from filament.sparse_loss import SparseMatcher


class DenseMaskLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict, full_resolution=False):
        super().__init__(config, weight_dict)
        self.full_resolution = full_resolution
        self.matcher = SparseMatcher(
            cost_class=config.class_weight,
            cost_mask=config.mask_weight,
            cost_dice=config.dice_weight,
            num_points=config.train_num_points,
        )

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks):
        src = self._get_predictions_permutation_indices(indices)
        tgt = self._get_targets_permutation_indices(indices)
        pred = masks_queries_logits[src].float()
        if len(pred) == 0:
            zero = masks_queries_logits.sum() * 0
            return {"loss_mask": zero, "loss_dice": zero}
        target, _ = self._pad_images_to_max_in_batch(mask_labels)
        target = target[tgt].float()
        if self.full_resolution:
            pred = F.interpolate(
                pred[:, None],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        else:
            target = F.interpolate(target[:, None], size=pred.shape[-2:], mode="area")[
                :, 0
            ]
        bce = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            .flatten(1)
            .mean(1)
            .sum()
            / num_masks
        )
        prob = pred.sigmoid().flatten(1)
        truth = target.flatten(1)
        dice = (
            1 - (2 * (prob * truth).sum(1) + 1) / (prob.sum(1) + truth.sum(1) + 1)
        ).sum() / num_masks
        return {"loss_mask": bce, "loss_dice": dice}
