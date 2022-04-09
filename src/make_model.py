import io
import logging
import os
import zipfile
from typing import Any

import joblib
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor

log = logging.getLogger(__name__)


def make_model(c, device=None, model_path=None):
    if c.params.model == "ump_ad_ae":
        model = AnomalyDetectionAutoEncoderModel(c)
    elif c.params.model == "ump_1":
        model = MLPModel(c)
    elif c.params.model == "ump_1_tf":
        model = MLPModelTF(c)
    elif c.params.model == "ump_1dcnn":
        model = OneDCNNModel(c)
    elif c.params.model == "ump_1dcnn_small":
        model = SmallOneDCNNModel(c)
    elif c.params.model == "ump_lstm":
        model = LSTMModel(c)
    elif c.params.model == "ump_lstm_tf":
        model = LSTMModelTF(c)
    elif c.params.model == "ump_transformer":
        model = TransformerModel(c)
    elif c.params.model == "ump_1dcnn_lstm":
        model = OneDCNNLSTMModel(c)
    else:
        raise Exception("Invalid model.")

    if c.settings.multi_gpu:
        model = nn.DataParallel(model)
    if device:
        model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    return model


def make_model_xgboost(c, ds=None, model_path=None):

    xgb_params = dict(
        n_estimators=10000,
        learning_rate=0.05,
        eval_metric='rmse',
        random_state=c.params.seed,
        tree_method='gpu_hist',
    )  # type: dict[str, Any]

    # if ds is not None:
    #     num_data = len(ds)
    #     num_steps = num_data // (c.params.batch_size * c.params.gradient_acc_step) * c.params.epoch + 5
    #
    #     xgb_params["scheduler_params"] = dict(T_0=num_steps, T_mult=1, eta_min=c.params.min_lr, last_epoch=-1)
    #     xgb_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    clf = xgb.XGBRegressor(**xgb_params)

    if model_path is not None:
        clf.load_model(model_path)

    return clf


def make_model_tabnet(c, ds=None, model_path=None):

    tabnet_params = dict(
        # n_d=8,
        # n_a=8,
        # n_steps=3,
        # n_independent=1,  # 2 is better CV than 1, but need more time
        # n_shared=1,  # same above
        # gamma=1.3,
        # lambda_sparse=0,
        # cat_dims=[len(np.unique(train_cat[:, i])) for i in range(train_cat.shape[1])],
        # cat_emb_dim=[1] * train_cat.shape[1],
        # cat_idxs=features_cat_index,
        # optimizer_fn=torch.optim.Adam,
        # optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type="entmax",
        seed=c.params.seed,
        verbose=10,
    )  # type: dict[str, Any]

    if ds is not None:
        num_data = len(ds)
        num_steps = num_data // (c.params.batch_size * c.params.gradient_acc_step) * c.params.epoch + 5

        tabnet_params["scheduler_params"] = dict(T_0=num_steps, T_mult=1, eta_min=c.params.min_lr, last_epoch=-1)
        tabnet_params["scheduler_fn"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    clf = TabNetRegressor(**tabnet_params)

    if model_path is not None:
        clf.load_model(model_path)

    return clf


def load_model(c, device, pretrained=None):
    if pretrained is None:
        pretrained = c.params.pretrained
    models = []

    for training in pretrained:
        try:
            c.params.model = training.model
            c.params.n_class = training.n_class
            c.params.model_input = training.model_input
            c.params.feature_set = training.feature_set
        except Exception:
            pass

        if training.model == "lightgbm":
            model = joblib.load(os.path.join(training.dir, "lightgbm.pkl"))
        elif training.model == "tabnet":
            tabnet_zip = io.BytesIO()
            with zipfile.ZipFile(tabnet_zip, "w") as z:
                z.write(os.path.join(training.dir, "model_params.json"), arcname="model_params.json")
                z.write(os.path.join(training.dir, "network.pt"), arcname="network.pt")
            model = make_model_tabnet(c, model_path=tabnet_zip)
        else:
            model = make_model(c, device, training.dir)

        models.append(model)

    return models


def swish(x):
    return x * torch.sigmoid(x)


class AnomalyDetectionAutoEncoderModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.dim = c.params.model_input

        self.encoder = nn.Sequential(
            nn.Linear(self.dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 100),
        )

        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, self.dim),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.encoder(x)
            out = self.decoder(x)

        return out


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


class MLPModelTF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.dim = c.params.model_input

        self.bn_1 = nn.BatchNorm1d(self.dim)

        self.fc_1 = nn.Linear(self.dim, self.dim)
        self.fc_2 = nn.Linear(self.dim, self.dim // 2)
        self.fc_3 = nn.Linear(self.dim // 2, self.dim // 4)
        self.fc_4 = nn.Linear(self.dim // 4, self.dim // 8)
        self.fc_5 = nn.Linear(self.dim // 8, 1)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.bn_1(x)

            x = swish(self.fc_1(x))
            x = swish(self.fc_2(x))
            x = swish(self.fc_3(x))
            x = swish(self.fc_4(x))

            x = self.fc_5(x).squeeze(1)

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


class LSTMModelTF(nn.Module):
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
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'head' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

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
        self.ch_2 = 384
        self.ch_3 = 384

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


# https://www.kaggle.com/ghostcxs/fork-prediction-including-spatial-info-with-conv1d
class SmallOneDCNNModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp
        self.input = c.params.model_input

        self.hidden_size = 256
        self.ch_1 = 1
        self.ch_2 = 16
        self.ch_3 = 64
        self.head_size_1 = int(self.hidden_size / 4 / 4 / 2) * self.ch_3  # 512
        self.head_size_2 = 128
        self.head_size_3 = 32

        self.expand = nn.Sequential(
            nn.Linear(self.input, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_2, self.ch_2, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),
        )

        self.flt = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(self.head_size_1, self.head_size_1),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(self.head_size_1, self.head_size_2),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(self.head_size_2, self.head_size_3),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(self.head_size_3, c.params.n_class),
        )

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = self.expand(x).view(x.size(0), self.ch_1, -1)

            x = self.conv1(x)
            x = self.flt(x)

            x = self.head(x).squeeze(1)

        return x


# https://confit.atlas.jp/guide/event-img/jsai2018/3Pin1-44/public/pdf?type=in
class OneDCNNLSTMModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.amp = c.settings.amp

        # LSTM Parameters
        self.batch_size = c.params.batch_size
        self.window_size = c.params.model_window  # Sequence for LSTM
        self.num_feature = len(c.params.feature_set) - 1
        self.input_size = c.params.model_input
        self.input_size_by_feat = self.input_size // self.num_feature
        self.lstm_hidden_size = 256

        # 1D CNN Parameters
        self.conv_hidden_size = 256
        self.ch_1 = 1
        self.ch_2 = 16
        self.ch_3 = 64
        self.head_size_1 = self.lstm_hidden_size * 2  # bidirectional
        self.head_size_2 = self.head_size_1 // 2
        self.head_size_3 = self.head_size_2 // 2

        self.bn_1 = nn.BatchNorm1d(self.window_size)

        self.expand = nn.Sequential(
            nn.Linear(self.input_size, self.conv_hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.ch_1, self.ch_2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_2, self.ch_2, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_2),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_2, self.ch_3, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=4, padding=0, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),

            nn.Conv1d(self.ch_3, self.ch_3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(self.ch_3),
            nn.LeakyReLU(),

            nn.Flatten(),
        )

        self.rnn = nn.LSTM(self.head_size_1, self.lstm_hidden_size, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(self.head_size_1, self.head_size_2),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(self.head_size_2, self.head_size_3),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(self.head_size_3, 1),
        )

    # def forward(self, x, h_c=None):
    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            x = x.view(-1, self.num_feature, self.window_size, self.input_size_by_feat)
            x = x.permute(0, 2, 1, 3).reshape(-1, self.window_size, self.input_size)

            x = self.bn_1(x)

            xs = []
            for n in range(self.window_size):
                xs.append(self.conv1(self.expand(x[:, n]).view(x.size(0), self.ch_1, -1)))
            x = torch.stack(xs, dim=1)

            x, _ = self.rnn(x)

            x = self.head(x).view(-1, self.window_size)

        return x  # , h_c

