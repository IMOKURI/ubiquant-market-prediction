import joblib
import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.params.model == "ump_1":
        model = MLPModel(c)
    elif c.params.model == "ump_1dcnn":
        model = OneDCNNModel(c)
    else:
        raise Exception("Invalid model.")

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    if device:
        model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model


def load_model(c, device, pretrained=None):
    if pretrained is None:
        pretrained = c.params.pretrained
    models = []

    for training in pretrained:
        c.params.model = training.model

        if training.model == "lightgbm":
            model = joblib.load(os.path.join(training.dir, "lightgbm.pkl"))
        else:
            model = make_model(c, device, training.dir)

        models.append(model)

    return models


def swish(x):
    return x * torch.sigmoid(x)


class MLPModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.dim = c.params.model_input
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


# https://www.kaggle.com/c/lish-moa/discussion/202256
# https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
class OneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.params.model_input

        self.hidden_size = 1024
        self.ch_1 = 128
        self.ch_2 = 384
        self.ch_3 = 384

        self.ch_po_1 = int(self.hidden_size / self.ch_1 / 2)
        self.ch_po_2 = int(self.hidden_size / self.ch_1 / 2 / 2) * self.ch_3

        self.expand = nn.Sequential(
            nn.BatchNorm1d(self.input), nn.utils.weight_norm(nn.Linear(self.input, self.hidden_size)), nn.CELU(0.06)
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(self.ch_1),
            nn.Dropout(0.1),
            nn.utils.weight_norm(
                nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=False), dim=None
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=self.ch_po_1),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.1),
            nn.utils.weight_norm(
                nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            nn.utils.weight_norm(
                nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            nn.utils.weight_norm(
                nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None
            ),
            nn.ReLU(),
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.ch_po_2),
            nn.Dropout(0.1),
            nn.utils.weight_norm(nn.Linear(self.ch_po_2, 1)),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.conv2(x) * x

            x = self.max_po_c2(x)
            x = self.flt(x)

            x = self.head(x)

        return x
