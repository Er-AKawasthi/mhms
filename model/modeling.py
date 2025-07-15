# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from math import sqrt
from scipy import ndimage

import models.configs as configs
from .lvc import LVCBlock
from vit_pytorch.vit import ViT

logger = logging.getLogger(__name__)

# ------------------------------------
# Utility Functions
# ------------------------------------

def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish
}

# ------------------------------------
# ViT-based Model for CMD
# ------------------------------------

class CmdVIT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=7, zero_head=False, vis=False):
        super(CmdVIT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        # Use vit-pytorch's ViT directly
        self.transformer = ViT(
            image_size=img_size,
            patch_size=config.patches.size[0],
            num_classes=num_classes,
            dim=config.hidden_size,
            depth=config.transformer.num_layers,
            heads=config.transformer.num_heads,
            mlp_dim=config.transformer.mlp_dim
        )

    def forward(self, x, labels=None):
        outputs = self.transformer(x)

        # Handle possible return types (logits only or logits + attn)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            attn_weights = outputs[1] if len(outputs) > 1 else None
            atten = outputs[2] if len(outputs) > 2 else None
        else:
            logits = outputs
            attn_weights = atten = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, atten
        else:
            return logits, attn_weights, atten

    def load_from(self, weights):
        logger.warning("load_from called, but vit-pytorch does not use .npz pretrained weights here. Ignored.")

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
