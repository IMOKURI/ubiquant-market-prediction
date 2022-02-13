import gc
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
from nptyping import NDArray
from omegaconf.dictconfig import DictConfig
from sklearn.neighbors import NearestNeighbors

from .investment import Investments
from .utils import catch_everything_in_kaggle


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Store:
    def __init__(
        self,
        investments: Investments,
        training_array: Optional[NDArray[(Any, Any), Any]] = None,
        nearest_neighbors: Optional[NearestNeighbors] = None,
    ):
        self.investments = investments
        self.training_array = training_array
        self.nearest_neighbors = nearest_neighbors

    @classmethod
    def empty(cls) -> "Store":
        investments = Investments()

        return cls(investments)

    @classmethod
    def train(cls, c: DictConfig) -> "Store":
        instance = cls.empty()

        if os.path.exists(os.path.join(c.settings.dirs.input_minimal, "train_min.npy")):
            instance.training_array = np.load(os.path.join(c.settings.dirs.input_minimal, "train_min.npy"))

        if os.path.exists(os.path.join(c.settings.dirs.preprocess, "nearest_neighbors.pkl")):
            instance.nearest_neighbors = pickle.load(
                open(os.path.join(c.settings.dirs.preprocess, "nearest_neighbors.pkl"), "rb")
            )

        # if os.path.exists(os.path.join(c.settings.dirs.feature, "train_low_mem.parquet")):
        #     df = pd.read_parquet(os.path.join(c.settings.dirs.feature, "train_low_mem.parquet"))
        # elif os.path.exists(os.path.join(c.settings.dirs.input, "train.f")):
        #     df = pd.read_feather(os.path.join(c.settings.dirs.input, "train.f"))
        # else:
        #     df = pd.read_csv(os.path.join(c.settings.dirs.input, "train.csv"))
        #
        # for row in tqdm(df.values):
        #     instance.append(row)
        #
        # del df
        # gc.collect()

        return instance

    def append(self, row: NDArray[(Any,), Any]):
        # row の中のデータは dtype: object であることに注意。

        # ここではなく、特徴量生成後、にやる
        # ↑ の前に dtype を変更しないといけない気がする。
        # row = np.nan_to_num(row)

        # TODO: 最終的に catch_everything_in_kaggle をいれていく
        # with catch_everything_in_kaggle():
        self.investments.extend(row)

    def append_post(self, row: NDArray[(Any,), Any]):
        # row の中のデータは dtype: object であることに注意。

        # TODO: 最終的に catch_everything_in_kaggle をいれていく
        # with catch_everything_in_kaggle():
        self.investments.extend_post(row)
