"""ResNet-18 classifier adapted for MRI tumor detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from ml.config import NUM_CLASSES


def build_model(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = models.resnet18(weights=weights)

    # MRI slices are often single-channel; replicate to RGB for transfer learning.
    old_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        3,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained_backbone:
        with torch.no_grad():
            backbone.conv1.weight.copy_(old_conv.weight)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, NUM_CLASSES)
    return backbone


def load_model(weights_path, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cpu")
    model = build_model(pretrained_backbone=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
