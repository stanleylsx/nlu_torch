# -*- coding: utf-8 -*-
# @Time : 2022/5/19 17:02
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : focal_loss.py
# @Software: VSCode
import torch
import torch.nn.functional as F
from configure import configure


class FocalLoss(torch.nn.Module):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, device, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        weight = configure['weight']
        self.alpha = torch.Tensor(weight).to(device) if weight else None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N, ]
        """
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        log_pt = log_pt.gather(1, targets.view(-1, 1))
        pt = pt.gather(1, targets.view(-1, 1))
        loss = -torch.mul(torch.pow((1 - pt), self.gamma), log_pt)
        if self.alpha is not None:
            self.alpha = self.alpha.gather(0, targets)
            loss = torch.mul(self.alpha, loss.t())
        loss = torch.squeeze(loss)
        if self.reduction == 'mean':
            loss = torch.mean(loss, dim=0)
        return loss
