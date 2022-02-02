import numpy as np
from typing import Any

from nptyping import NDArray

from .investment import Investments


class Store:
    def __init__(self, investments: Investments):
        self.investments = investments

    @classmethod
    def empty(cls) -> "Store":
        investments = Investments()

        return cls(investments)

    def append(self, row: NDArray[(Any,), Any]):
        np.nan_to_num(row, copy=False)

        # TODO: 最終的に catch_everything_in_kaggle をいれていく
        self.investments.extend(row)
