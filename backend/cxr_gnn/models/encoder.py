"""
models/encoder.py — Frozen ResNet18 feature extractor.

layer1+layer2 of an ImageNet-pretrained ResNet18. Output: 128-channel
feature map (layer2 output). All parameters frozen — GATv2 only consumes
these features, never trains this part.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


def build_encoder(device: torch.device) -> nn.Module:
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        logger.info("Loaded ImageNet-pretrained ResNet18.")
    except Exception as ex:
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        logger.warning("Pretrained weights unavailable (%s). Using random init.", ex)

    encoder = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,   # → (1, 128, H/8, W/8)
    ).to(device).eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder
