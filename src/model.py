"""Target / shadow model architecture.

CIFAR-style ResNet-18 with a 3x3 stride-1 conv1 and the initial maxpool
removed (matches the target model from task_template.py). 9 output classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

NUM_CLASSES = 9


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def load_target(model_path, map_location: str = "cpu") -> nn.Module:
    model = build_model()
    state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
