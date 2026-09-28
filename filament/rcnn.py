"""COCO-initialized ROI instance model with mask-derived filament targets."""

from types import SimpleNamespace

import torch
from torch import nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import FrozenBatchNorm2d


def boxes_from_masks(masks):
    """Vectorized exclusive XYXY boxes, including one-pixel-wide filaments."""
    n, h, w = masks.shape
    if n == 0:
        return masks.new_empty((0, 4), dtype=torch.float32)
    rows, columns = masks.bool().any(2), masks.bool().any(1)
    yy = torch.arange(h, device=masks.device)[None]
    xx = torch.arange(w, device=masks.device)[None]
    return torch.stack(
        (
            torch.where(columns, xx, w).amin(1),
            torch.where(rows, yy, h).amin(1),
            torch.where(columns, xx, -1).amax(1) + 1,
            torch.where(rows, yy, -1).amax(1) + 1,
        ),
        1,
    ).float()


def freeze_backbone_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                for key in ["weight", "bias", "running_mean", "running_var"]:
                    getattr(frozen, key).copy_(getattr(child, key))
            setattr(module, name, frozen)
        else:
            freeze_backbone_norm(child)


class FilamentRCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
            box_detections_per_img=60,
            box_score_thresh=0.05,
            box_nms_thresh=0.7,
        )
        # Retain three anchors/location so the COCO RPN remains usable. Smaller
        # and more elongated anchors target the geometry of the training masks.
        self.model.rpn.anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.125, 1.0, 8.0),) * 5,
        )
        old = self.model.roi_heads.box_predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            old.cls_score.in_features, 2
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
        # A 28-pixel output erases thin branches in large filament boxes.
        # Pool at 28 instead of 14, producing 56-pixel masks with the same
        # pretrained convolutional head and a moderate memory increase.
        self.model.roi_heads.mask_roi_pool.output_size = (28, 28)
        freeze_backbone_norm(self.model.backbone)

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        # Dataset tensors already contain ImageNet-normalized grayscale. The
        # internal transform therefore uses identity channel normalization.
        size = pixel_values.shape[-1]
        self.model.transform.min_size = (size,)
        self.model.transform.max_size = size
        targets = None
        if mask_labels is not None:
            targets = [
                {
                    "boxes": boxes_from_masks(masks),
                    "masks": masks.to(torch.uint8),
                    "labels": torch.ones(
                        len(masks), dtype=torch.int64, device=masks.device
                    ),
                }
                for masks in mask_labels
            ]
        output = self.model(list(pixel_values.unbind()), targets)
        return (
            SimpleNamespace(loss=sum(output.values()), loss_dict=output)
            if self.training
            else output
        )
