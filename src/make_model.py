import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.params.model == "ump_1":
        model = BaseModel(c, model_path)
    else:
        raise Exception("Invalid model.")

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    if device:
        model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model


def load_model(c, device):
    models = []

    for training in c.params.pretrained:
        c.params.model = training.model
        model = make_model(c, device, training.dir)
        models.append(model)

    return models


def swish(x):
    return x * torch.sigmoid(x)


class BaseModel(nn.Module):
    def __init__(self, c, pretrained=True):
        super().__init__()
        self.amp = c.settings.amp
        self.dim = 300
        self.layers = 4

        self.bn_1 = nn.BatchNorm1d(self.dim)

        self.fcs = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.layers)])

        self.head = nn.Linear(self.dim, 1)

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.bn_1(x)
            for fc in self.fcs:
                x = swish(fc(x))

            x = self.head(x)

        return x
