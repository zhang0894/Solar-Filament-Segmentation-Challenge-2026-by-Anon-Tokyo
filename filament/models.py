"""Model construction; no credentials or dataset metadata enter the model."""

from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def make_model(
    kind="semantic",
    pretrained=True,
    sparse_points=False,
    dense_masks=False,
    highres_masks=False,
    encoder="tu-convnext_tiny.fb_in22k_ft_in1k",
):
    if sparse_points and dense_masks:
        raise ValueError("Choose one mask loss")
    if highres_masks and (kind != "instance" or not dense_masks):
        raise ValueError(
            "High-resolution query masks require the dense instance criterion"
        )
    if kind == "rcnn":
        from filament.rcnn import FilamentRCNN

        return FilamentRCNN(pretrained)
    if kind == "semantic":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=2,
        )
    if kind == "instance":
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        checkpoint = "facebook/mask2former-swin-base-coco-instance"
        cache_dir = ROOT / "weights/huggingface"
        config = Mask2FormerConfig.from_pretrained(checkpoint, cache_dir=cache_dir)
        config.num_labels = 1
        config.id2label = {0: "filament"}
        config.label2id = {"filament": 0}
        config.train_num_points = 12544
        if pretrained:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                checkpoint,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=True,
            )
        else:
            model = Mask2FormerForUniversalSegmentation(config)
        if sparse_points:
            from filament.sparse_loss import SparseMaskLoss

            model.criterion = SparseMaskLoss(config, model.weight_dict)
        if dense_masks:
            from filament.dense_loss import DenseMaskLoss

            model.criterion = DenseMaskLoss(
                config, model.weight_dict, full_resolution=highres_masks
            )
        if highres_masks:
            from filament.highres_masks import install_high_resolution_head

            install_high_resolution_head(model)
        return model
    raise ValueError(kind)


def semantic_loss(logits, target):
    logits, target = logits.float(), target.float()
    foreground = logits[:, :1]
    truth = target[:, :1]
    weights = target[:, 2:3] if target.shape[1] > 2 else None
    bce = F.binary_cross_entropy_with_logits(foreground, truth, weight=weights)
    prob = foreground.sigmoid()
    weighted_prob = prob if weights is None else prob * weights
    weighted_truth = truth if weights is None else truth * weights
    intersection = (weighted_prob * truth).sum(dim=(2, 3))
    dice = (
        1
        - (
            (2 * intersection + 1)
            / (weighted_prob.sum(dim=(2, 3)) + weighted_truth.sum(dim=(2, 3)) + 1)
        ).mean()
    )
    edge = F.binary_cross_entropy_with_logits(
        logits[:, 1:], target[:, 1:2], pos_weight=logits.new_tensor(3.0)
    )
    return bce + dice + 0.2 * edge


def make_optimizer(model, kind, lr=1e-4):
    backbone, decoder = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith(
            {
                "semantic": "encoder.",
                "instance": "model.pixel_level_module.encoder",
                "rcnn": "model.backbone",
            }[kind]
        )
        (backbone if is_backbone else decoder).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": lr * 0.2},
            {"params": decoder, "lr": lr},
        ],
        weight_decay=0.01,
        fused=all(p.is_cuda for p in backbone + decoder),
    )
