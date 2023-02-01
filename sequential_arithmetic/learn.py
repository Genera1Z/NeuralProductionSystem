import torch as pt
import torch.nn as nn


class NormalizedMse(nn.Module):

    def __init__(self, reduction='mean'):
        super(NormalizedMse, self).__init__()
        assert reduction == 'mean'
        self.reduction = reduction

    def forward(self, input, target, eps=1e-9):
        numerator = nn.functional.mse_loss(input, target, reduction=self.reduction)
        denumerator = nn.functional.mse_loss(target, pt.zeros_like(target), reduction=self.reduction)
        normalized_mse = numerator / (denumerator + eps)
        return normalized_mse
