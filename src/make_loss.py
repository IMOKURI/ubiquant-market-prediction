import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts)


# ====================================================
# Criterion
# ====================================================
def make_criterion(c):
    if c.params.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif c.params.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif c.params.criterion == "MSELoss":
        criterion = nn.MSELoss()
    elif c.params.criterion == "RMSELoss":
        criterion = RMSELoss()
    elif c.params.criterion == "LabelSmoothCrossEntropyLoss":
        criterion = LabelSmoothCrossEntropyLoss(
            smoothing=c.params.label_smoothing)
    elif c.params.criterion == "LabelSmoothBCEWithLogitsLoss":
        criterion = LabelSmoothBCEWithLogitsLoss(
            smoothing=c.params.label_smoothing)
    elif c.params.criterion == "MarginRankingLoss":
        criterion = nn.MarginRankingLoss(margin=c.params.margin)

    else:
        raise Exception("Invalid criterion.")
    return criterion


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch/blob/main/label_smothing_cross_entropy_loss.py
class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes),
                            device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# https://www.kaggle.com/felipebihaiek/torch-continued-from-auxiliary-targets-smoothing
class LabelSmoothBCEWithLogitsLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothBCEWithLogitsLoss._smooth(targets, self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# ====================================================
# Optimizer
# ====================================================
def make_optimizer(c, model):
    if c.params.optimizer == "Adam":
        optimizer = Adam(
            model.parameters(), lr=c.params.lr, weight_decay=c.params.weight_decay
        )
    elif c.params.optimizer == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=c.params.lr, weight_decay=c.params.weight_decay
        )
    else:
        raise Exception("Invalid optimizer.")
    return optimizer


# ====================================================
# Scheduler
# ====================================================
def make_scheduler(c, optimizer, ds):
    num_data = len(ds)
    num_steps = (
        num_data // (c.params.batch_size *
                     c.params.gradient_acc_step) * c.params.epoch + 5
    )

    if c.params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=num_steps, T_mult=1, eta_min=c.params.min_lr, last_epoch=-1
        )
    elif c.params.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=c.params.min_lr, last_epoch=-1
        )
    elif c.params.scheduler == "CosineAnnealingWarmupRestarts":
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_steps,
            max_lr=c.params.lr,
            min_lr=c.params.min_lr,
            warmup_steps=(num_steps // 10),
        )

    else:
        raise Exception("Invalid scheduler.")
    return scheduler
