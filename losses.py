from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.alpha = alpha
        print(f"FOCAL LOSS, num classes {num_classes}")

    def _convert_target(self, target):

        target = torch.clip(target, 0, self.num_classes - 1)

        if self.num_classes == 2:
            new_target = (
                0.95 * F.one_hot(target, num_classes=self.num_classes) +
                0.05 * F.one_hot(1 - target, num_classes=self.num_classes)
            )
        else:
            new_target = 0.9 * F.one_hot(target, num_classes=self.num_classes)
            t0 = torch.clip(target - 1, 0, self.num_classes - 1)
            t1 = torch.clip(target + 1, 0, self.num_classes - 1)
            new_target += 0.05 * F.one_hot(t0, num_classes=self.num_classes)
            new_target += 0.05 * F.one_hot(t1, num_classes=self.num_classes)

        return new_target


    def forward(self, input, target):

        # new_target = self._convert_target(target)
        new_target = F.one_hot(target, num_classes=self.num_classes)
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = - new_target * log_prob * (1 - prob) ** self.gamma
        return self.alpha * loss.sum(dim=-1)
