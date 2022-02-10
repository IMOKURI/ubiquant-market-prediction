from typing import Any, Dict

import numpy as np
from nptyping import NDArray

from .streampy import StreamPy
from .utils import in_kaggle


class Investment:
    def __init__(self, investment_id: int, features: StreamPy, targets: StreamPy, pseudo_targets: StreamPy):
        self.investment_id = investment_id
        self.features = features
        self.targets = targets
        self.pseudo_targets = pseudo_targets

    @classmethod
    def empty(cls, investment_id: int, dtype: type = np.float32, default_value: float = 0.0):
        feature_cols = [f"f_{n}" for n in range(300)]
        features = StreamPy.empty(feature_cols, dtype, default_value)

        target_cols = ["target"]
        targets = StreamPy.empty(target_cols, dtype, default_value)

        pseudo_target_cols = ["pseudo_target"]
        pseudo_targets = StreamPy.empty(pseudo_target_cols, dtype, default_value)

        return Investment(investment_id, features, targets, pseudo_targets)

    def last_n(self, n: int) -> "Investment":
        return Investment(
            self.investment_id, self.features.last_n(n), self.targets.last_n(n), self.pseudo_targets.last_n(n)
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
            self[int(row[1])].features.extend(row[2:302].astype(np.float32).reshape(1, -1))

        else:
            self[int(row[2])].features.extend(row[4:304].astype(np.float32).reshape(1, -1))
            self[int(row[2])].targets.extend(row[3:4].astype(np.float32).reshape(1, -1))

    def extend_post(self, row: NDArray[(Any,), Any]):
        """
        Args:
            row (NDArray[(Any, ), Any]): 以下のスキーマを期待している
                Train/Test data (2 col): row_id, target
        """

        self[int(row[0].split("_")[1])].pseudo_targets.extend(row[1:2].astype(np.float32).reshape(1, -1))
