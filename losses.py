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


class ZINB(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) loss.

    Args:
        x: observed gene expression (batch_size, n_genes)
        mu: predicted mean (batch_size, n_genes)
        theta: predicted dispersion (batch_size, n_genes)
        pi: predicted zero-inflation logits (batch_size, n_genes)  (logits! before sigmoid)
        eps: small number for numerical stability
        reduction: 'mean' or 'sum'

    Returns:
        Loss value
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, mu, theta, pi):

        """
        log_mu, log_theta, pi_logits = model(input)
        mu = torch.exp(log_mu)
        theta = torch.exp(log_theta)

        loss = zinb_loss(x_true, mu, theta, pi_logits)
        loss.backward()
        """
        softplus_pi = F.softplus(-pi)  # log(1 + exp(-pi)) for stability
        pi_prob = torch.sigmoid(pi)

        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + self.eps)
        t2 = (theta + self.eps) * torch.log(theta + self.eps) + (x + self.eps) * torch.log(mu + self.eps)
        t3 = (theta + x + self.eps) * torch.log(theta + mu + self.eps)
        nb_case = t1 + t2 - t3  # log NB likelihood

        zero_case = -softplus_pi + torch.log(pi_prob + (1.0 - pi_prob) * torch.exp(nb_case))

        nb_case = -softplus_pi - pi_prob + (1.0 - pi_prob) * torch.exp(nb_case)
        result = torch.where(x < 1e-8, zero_case, nb_case)

        if self.reduction == 'mean':
            return -result.mean()
        elif self.reduction == 'sum':
            return -result.sum()
        else:
            return -result