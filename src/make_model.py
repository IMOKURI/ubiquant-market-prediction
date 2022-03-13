import logging
import os

import joblib
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
    elif c.params.model == "ump_lstm":
        model = LSTMModel(c)
    elif c.params.model == "ump_transformer":
        model = TransformerModel(c)
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
        try:
            c.params.model = training.model
            c.params.model_input = training.model_input
            c.params.n_class = training.n_class
        except Exception:
            pass

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

        self.bn_1 = nn.BatchNorm1d(self.dim)

        self.fc_1 = nn.Linear(self.dim, self.dim)
        self.fc_2 = nn.Linear(self.dim, self.dim // 2)
        self.fc_3 = nn.Linear(self.dim // 2, self.dim // 4)
        self.fc_4 = nn.Linear(self.dim // 4, self.dim // 8)

        self.head = nn.Linear(self.dim // 8, 1)

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.bn_1(x)

            x = swish(self.fc_1(x))
            x = swish(self.fc_2(x))
            x = swish(self.fc_3(x))
            x = swish(self.fc_4(x))

            x = self.head(x).squeeze(1)

        return x


class LSTMModel(nn.Module):
    """
    LSTM を DataParallel で使うときの注意。
    hidden と cell の shape の batch_size が GPU数で分割された値になってしまう。
    https://discuss.pytorch.org/t/dataparallel-lstm-gru-wrong-hidden-batch-size-8-gpus/6701

    Cell state: Long term memory of the model, only part of LSTM models
    Hidden state: Working memory, part of LSTM and RNN models
    """

    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp

        self.batch_size = c.params.batch_size
        self.window_size = c.params.model_window  # Sequence for LSTM
        self.num_feature = len(c.params.feature_set) - 1
        self.input_size = c.params.model_input
        self.input_size_by_feat = self.input_size // self.num_feature
        self.hidden_size = 300

        self.bn_1 = nn.BatchNorm1d(self.window_size)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)

        # num_gpu = len(c.settings.gpus.split(","))
        # hidden = torch.zeros(1, self.batch_size // num_gpu, self.hidden_size)
        # cell = torch.zeros(1, self.batch_size // num_gpu, self.hidden_size)
        # self.hidden_cell = (hidden, cell)

        # self.head = nn.Linear(self.hidden_size, 1)
        self.head = nn.Linear(self.hidden_size * 2, 1)  # bidirectional

    # def forward(self, x, h_c=None):
    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = x.view(-1, self.num_feature, self.window_size, self.input_size_by_feat)
            x = x.permute(0, 2, 1, 3).reshape(-1, self.window_size, self.input_size)
            x = self.bn_1(x)

            # if h_c is None:
            #     x, h_c = self.lstm(x)
            # else:
            #     x, h_c = self.lstm(x, h_c)
            x, _ = self.rnn(x)

            x = self.head(x).view(-1, self.window_size)

        return x  # , h_c


class TransformerModel(nn.Module):
    """
    シングルGPUで実行する必要がある。
    （現状、マルチGPUの設定 DataParallel がたぶんよろしくない）
    """
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp

        self.batch_size = c.params.batch_size
        self.window_size = c.params.model_window  # Sequence
        self.num_feature = len(c.params.feature_set) - 1
        self.input_size = c.params.model_input
        self.input_size_by_feat = self.input_size // self.num_feature
        self.num_head = 9

        self.bn_1 = nn.BatchNorm1d(self.window_size)

        # batch_first=False なので注意。(pytorch のバージョンが古い)
        self.transformer = nn.Transformer(d_model=self.input_size, nhead=self.num_head)

        self.head = nn.Linear(self.input_size, 1)

    # def forward(self, x, h_c=None):
    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            batch_size = x.size(0)
            x = x.view(-1, self.num_feature, self.window_size, self.input_size_by_feat)
            x = x.permute(0, 2, 1, 3).reshape(batch_size, self.window_size, self.input_size)
            x = self.bn_1(x).permute(1, 0, 2)

            src = x[:-1, :, :]
            tgt = x[-1:, :, :]

            x = self.transformer(src, tgt).squeeze(0)

            x = self.head(x).view(batch_size)

        return x


def weight_norm(layer, dim=None, enabled=True):
    return nn.utils.weight_norm(layer, dim=dim) if enabled else layer


# https://www.kaggle.com/c/lish-moa/discussion/202256
# https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
class OneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.params.model_input

        self.hidden_size = 1024
        self.ch_1 = 128
        self.ch_2 = 256
        self.ch_3 = 256

        self.ch_po_1 = int(self.hidden_size / self.ch_1 / 2)
        self.ch_po_2 = int(self.hidden_size / self.ch_1 / 2 / 2) * self.ch_3

        self.expand = nn.Sequential(
            nn.BatchNorm1d(self.input), weight_norm(nn.Linear(self.input, self.hidden_size)), nn.CELU(0.06)
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(self.ch_1),
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=False)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=self.ch_po_1),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(),
            nn.BatchNorm1d(self.ch_2),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True)),
            nn.ReLU(),
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.ch_po_2),
            nn.Dropout(0.1),
            nn.utils.weight_norm(nn.Linear(self.ch_po_2, c.params.n_class)),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.conv2(x) * x

            x = self.max_po_c2(x)
            x = self.flt(x)

            x = self.head(x).squeeze(1)

        return x
