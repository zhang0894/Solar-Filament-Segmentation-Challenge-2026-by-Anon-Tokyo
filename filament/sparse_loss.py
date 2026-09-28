"""Optional foreground-aware Mask2Former supervision for very small objects.

All coordinates are derived exclusively from training instance masks. Matching
uses a shared point pool for all prediction/target pairs. Unmatched queries
retain the standard no-object classification loss.
"""

import torch
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    Mask2FormerLoss,
    dice_loss,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
    sigmoid_cross_entropy_loss,
)


@torch.no_grad()
def positive_coordinates(masks, count):
    """Sample actual foreground pixel centers; empty masks use uniform points."""
    n, h, w = masks.shape
    if n == 0:
        return masks.new_empty((0, count, 2))
    weights = masks.reshape(n, -1).float()
    empty = weights.sum(-1) == 0
    weights = weights + empty[:, None].to(weights.dtype)
    indices = torch.multinomial(weights, count, replacement=True)
    return torch.stack(((indices % w + 0.5) / w, (indices // w + 0.5) / h), -1)


class SparseMatcher(Mask2FormerHungarianMatcher):
    @torch.no_grad()
    def forward(
        self, masks_queries_logits, class_queries_logits, mask_labels, class_labels
    ):
        indices = []
        for logits, classes, targets, labels in zip(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels
        ):
            if len(targets) == 0:
                indices.append(
                    (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
                )
                continue
            # Each target gets foreground coverage; all query/target costs then
            # use the same coordinates to keep the assignment objective coherent.
            per_target = max(1, self.num_points // (4 * len(targets)))
            positive = positive_coordinates(targets, per_target).reshape(1, -1, 2)
            uniform = torch.rand(
                1, self.num_points - positive.shape[1], 2, device=logits.device
            )
            coords = torch.cat((uniform, positive), 1)
            pred_points = sample_point(
                logits[:, None].float(),
                coords.expand(len(logits), -1, -1),
                align_corners=False,
            ).squeeze(1)
            target_points = sample_point(
                targets[:, None].float(),
                coords.expand(len(targets), -1, -1),
                align_corners=False,
            ).squeeze(1)
            cost = self.cost_class * -classes.float().softmax(-1)[:, labels]
            cost = cost + self.cost_mask * pair_wise_sigmoid_cross_entropy_loss(
                pred_points, target_points
            )
            cost = cost + self.cost_dice * pair_wise_dice_loss(
                pred_points, target_points
            )
            if not torch.isfinite(cost).all():
                raise FloatingPointError("Nonfinite matching cost")
            row, col = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long),
                    torch.as_tensor(col, dtype=torch.long),
                )
            )
        return indices


class SparseMaskLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict):
        super().__init__(config, weight_dict)
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
        with torch.no_grad():
            n_positive = self.num_points // 4
            base = self.sample_points_using_uncertainty(
                pred[:, None],
                lambda x: self.calculate_uncertainty(x),
                self.num_points - n_positive,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            positive = positive_coordinates(target, n_positive)
            coords = torch.cat((base, positive), 1)
            labels = sample_point(target[:, None], coords, align_corners=False).squeeze(
                1
            )
        logits = sample_point(pred[:, None], coords, align_corners=False).squeeze(1)
        return {
            "loss_mask": sigmoid_cross_entropy_loss(logits, labels, num_masks),
            "loss_dice": dice_loss(logits, labels, num_masks),
        }
