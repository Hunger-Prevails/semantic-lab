import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, reduction, ignore_index, gamma = 2.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none', ignore_index = self.ignore_index)

        focal_loss = (1.0 - torch.exp(- ce_loss)) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise NotImplementedError()
