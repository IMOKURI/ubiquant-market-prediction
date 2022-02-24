from typing import Any, Dict

import numpy as np
from nptyping import NDArray

from .features.helper import *
from .streampy import StreamPy
from .utils import in_kaggle


class Investment:
    def __init__(
        self,
        investment_id: int,
        features: StreamPy,
        features_ma_short: StreamPy,
        features_ma_long: StreamPy,
        features_macd: StreamPy,
        targets: StreamPy,
        pseudo_targets: StreamPy,
    ):
        self.investment_id = investment_id
        self.features = features
        self.features_ma_short = features_ma_short  # moving average in short term
        self.features_ma_long = features_ma_long  # moving average in long term
        self.features_macd = features_macd  # moving average convergence divergence (短期と長期の移動平均の差)
        self.targets = targets
        self.pseudo_targets = pseudo_targets

    @classmethod
    def empty(cls, investment_id: int, dtype: type = np.float32, default_value: float = 0.0):
        feature_cols = [f"f_{n}" for n in range(300)]
        features = StreamPy.empty(feature_cols, dtype, default_value)

        feature_ma_short_cols = [f"f_ma_short_{n}" for n in range(300)]
        features_ma_short = StreamPy.empty(feature_ma_short_cols, dtype, default_value)

        feature_ma_long_cols = [f"f_ma_long_{n}" for n in range(300)]
        features_ma_long = StreamPy.empty(feature_ma_long_cols, dtype, default_value)

        feature_macd_cols = [f"f_macd_{n}" for n in range(300)]
        features_macd = StreamPy.empty(feature_macd_cols, dtype, default_value)

        target_cols = ["target"]
        targets = StreamPy.empty(target_cols, dtype, default_value)

        pseudo_target_cols = ["pseudo_target"]
        pseudo_targets = StreamPy.empty(pseudo_target_cols, dtype, default_value)

        return Investment(
            investment_id, features, features_ma_short, features_ma_long, features_macd, targets, pseudo_targets
        )

    def last_n(self, n: int) -> "Investment":
        return Investment(
            self.investment_id,
            self.features.last_n(n),
            self.features_ma_short.last_n(n),
            self.features_ma_long.last_n(n),
            self.features_macd.last_n(n),
            self.targets.last_n(n),
            self.pseudo_targets.last_n(n),
        )


class Investments:
    def __init__(self):
        self.investments = {}  # type: Dict[int, Investment]

    def __getitem__(self, investment_id: int) -> Investment:
        if investment_id not in self.investments:
            self.investments[investment_id] = Investment.empty(investment_id)
        return self.investments[investment_id]

    def extend(self, row: NDArray[(Any,), Any]):
        """
        Args:
            row (NDArray[(Any, ), Any]): 以下のスキーマを期待している
                Test data (302 col): row_id, investment_id, f_x, ...
                Train data (304 + α col): row_id, time_id, investment_id, target, f_x, ... , fold
        """
        # time_id を使うかどうか
        # row[0] = int(row[0].split("_")[0])

        if in_kaggle() or row.shape[0] == 302:
            investment_id = int(row[1])
            self[investment_id].features.extend(row[2:302].astype(np.float32).reshape(1, -1))

        else:
            investment_id = int(row[2])
            self[investment_id].features.extend(row[4:304].astype(np.float32).reshape(1, -1))
            self[investment_id].targets.extend(row[3:4].astype(np.float32).reshape(1, -1))

        last_20 = self[investment_id].features.last_n(20)
        ma_short = np.zeros((300,), dtype=self[investment_id].features.dtype)
        ma_long = np.zeros((300,), dtype=self[investment_id].features.dtype)
        macd = np.zeros((300,), dtype=self[investment_id].features.dtype)
        for n, col in enumerate(last_20.T):
            ma_short[n] = nanmean(col[-5:])
            ma_long[n] = nanmean(col[-20:])
            macd[n] = ma_short[n] - ma_long[n]

        self[investment_id].features_ma_short.extend(ma_short.reshape(1, -1))
        self[investment_id].features_ma_long.extend(ma_long.reshape(1, -1))
        self[investment_id].features_macd.extend(macd.reshape(1, -1))

    def extend_post(self, row: NDArray[(Any,), Any]):
        """
        Args:
            row (NDArray[(Any, ), Any]): 以下のスキーマを期待している
                Train/Test data (2 col): row_id, target
        """

        self[int(row[0].split("_")[1])].pseudo_targets.extend(row[1:2].astype(np.float32).reshape(1, -1))
