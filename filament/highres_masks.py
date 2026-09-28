"""Optional stride-two mask features with an image-detail residual pathway."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class HighResolutionMaskFeatures(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels + 32, channels, 1),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
        )
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, image, features):
        detail = self.detail(image)
        base = F.interpolate(
            features, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        return base + self.project(self.mix(torch.cat((base, detail), dim=1)))


def high_resolution_pixel_forward(self, pixel_values, output_hidden_states=False):
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerPixelLevelModule,
    )

    output = Mask2FormerPixelLevelModule.forward(
        self, pixel_values, output_hidden_states=output_hidden_states
    )
    output.decoder_last_hidden_state = self.detail_head(
        pixel_values, output.decoder_last_hidden_state
    )
    return output


def install_high_resolution_head(model):
    module = model.model.pixel_level_module
    module.detail_head = HighResolutionMaskFeatures(model.config.mask_feature_size)
    module.forward = MethodType(high_resolution_pixel_forward, module)
