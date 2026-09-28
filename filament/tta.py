"""Invertible geometric test-time averaging for semantic logits."""

import torch


@torch.no_grad()
def semantic_tta(model, image, mode="none"):
    if mode in {"none", "flip"}:
        logits = model(image)[:, 0]
        if mode == "flip":
            # Keep the already validated three-view numeric protocol unchanged.
            for axis in [-1, -2]:
                logits += model(image.flip(axis))[:, 0].flip(axis)
            logits /= 3
        return logits
    if mode != "d4":
        raise ValueError(f"Unknown semantic TTA: {mode}")
    total = None
    for turns in range(4):
        rotated = torch.rot90(image, turns, (-2, -1))
        for flip in [False, True]:
            view = rotated.flip(-1) if flip else rotated
            logits = model(view.contiguous(memory_format=torch.channels_last))[
                :, 0
            ].float()
            if flip:
                logits = logits.flip(-1)
            logits = torch.rot90(logits, -turns, (-2, -1))
            total = logits if total is None else total + logits
    return total / 8


@torch.no_grad()
def refiner_tta(model, image, mode="none"):
    if mode == "none":
        return model(image)
    if mode not in {"flip", "d4"}:
        raise ValueError(f"Unknown refinement TTA: {mode}")
    views = (
        [(0, False), (0, True), (2, True)]
        if mode == "flip"
        else [(k, flip) for k in range(4) for flip in [False, True]]
    )
    total_mask, total_quality = None, None
    for turns, flip in views:
        transformed = torch.rot90(image, turns, (-2, -1))
        if flip:
            transformed = transformed.flip(-1)
        mask, quality = model(transformed.contiguous(memory_format=torch.channels_last))
        mask, quality = mask.float(), quality.float()
        if flip:
            mask = mask.flip(-1)
        mask = torch.rot90(mask, -turns, (-2, -1))
        total_mask = mask if total_mask is None else total_mask + mask
        total_quality = quality if total_quality is None else total_quality + quality
    return total_mask / len(views), total_quality / len(views)
