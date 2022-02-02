import numpy as np
from typing import Dict, Any

from nptyping import NDArray

from .streampy import StreamPy


class Investment:
    def __init__(self, investment_id: int, features: StreamPy, targets: StreamPy):
        self.investment_id = investment_id
        self.features = features
        self.targets = targets

    @classmethod
    def empty(cls, investment_id: int, dtype: type = np.float32, default_value: float = 0.0):
        feature_cols = [f"f_{n}" for n in range(300)]
        features = StreamPy.empty(feature_cols, dtype, default_value)

        target_cols = ["target"]
        targets = StreamPy.empty(target_cols, dtype, default_value)

        return Investment(investment_id, features, targets)

    def last_n(self, n: int) -> "Investment":
        return Investment(self.investment_id, self.features.last_n(n), self.targets.last_n(n))


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
                Train data (304 col): row_id, time_id, investment_id, target, f_x, ...
        """
        row[0] = int(row[0].split("_")[0])

        if row.shape[0] == 302:
            self[row[1]].features.extend(row[2:].reshape(1, -1))

        else:
            self[row[2]].features.extend(row[4:].reshape(1, -1))
            self[row[2]].targets.extend(row[3:4].reshape(1, -1))
