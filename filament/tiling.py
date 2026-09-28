"""Overlapping native-image inference for models trained on local crops."""

import math

import torch
from torch.nn import functional as F

from filament.tta import semantic_tta


def tile_starts(length, size):
    if not 0 < size <= length:
        raise ValueError("Tile size must fit the image")
    count = math.ceil((length - size) / max(1, size // 2)) + 1
    return [round(i * (length - size) / max(1, count - 1)) for i in range(count)]


@torch.no_grad()
def tiled_logits(model, image, tile_size, model_size, mode="none"):
    """Blend logits with positive edge weights; preserve every native pixel."""
    batch, _, height, width = image.shape
    ys, xs = tile_starts(height, tile_size), tile_starts(width, tile_size)
    window = torch.hann_window(
        tile_size, periodic=False, device=image.device, dtype=torch.float32
    ).clamp_min(0.05)
    weight = window[:, None] * window[None, :]
    total = torch.zeros((batch, height, width), device=image.device)
    normalizer = torch.zeros((height, width), device=image.device)
    for y in ys:
        for x in xs:
            tile = image[:, :, y : y + tile_size, x : x + tile_size]
            if tile_size != model_size:
                tile = F.interpolate(
                    tile, (model_size, model_size), mode="bilinear", align_corners=False
                )
            value = semantic_tta(
                model, tile.contiguous(memory_format=torch.channels_last), mode
            ).float()
            if value.shape[-2:] != (tile_size, tile_size):
                value = F.interpolate(
                    value[:, None],
                    (tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            total[:, y : y + tile_size, x : x + tile_size] += value * weight
            normalizer[y : y + tile_size, x : x + tile_size] += weight
    return total / normalizer
