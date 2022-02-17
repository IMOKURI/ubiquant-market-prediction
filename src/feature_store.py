import gc
import os
import pickle
from typing import Any, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
from nptyping import NDArray
from omegaconf.dictconfig import DictConfig
from ppca import PPCA
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

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
        # sampling_array: Optional[NDArray[(Any, Any), Any]] = None,
        # scalers: Optional[List[Union[StandardScaler, PowerTransformer]]] = None,
        # pca: Optional[Union[PCA, PPCA]] = None,
        nearest_neighbors: Optional[NearestNeighbors] = None,
    ):
        self.investments = investments
        self.training_array = training_array
        # self.sampling_array = sampling_array
        # self.scalers = scalers
        # self.pca = pca
        self.nearest_neighbors = nearest_neighbors

    @classmethod
    def empty(cls) -> "Store":
        investments = Investments()

        return cls(investments)

    @classmethod
    def train(cls, c: DictConfig) -> "Store":
        instance = cls.empty()

        training_array_path = os.path.join(c.settings.dirs.input_minimal, "train_min.npy")
        # sampling_array_path = os.path.join(c.settings.dirs.input_minimal, f"sampling_pca{c.params.pca_n_components}.npy")

        # standard_scaler_0_path = os.path.join(c.settings.dirs.preprocess, "standard_scaler_f_0.pkl")
        # pca_path = os.path.join(c.settings.dirs.preprocess, f"pca_{c.params.pca_n_components}.pkl")
        nearest_neighbors_path = os.path.join(c.settings.dirs.preprocess, "faiss_ivfpq.index")

        if os.path.exists(training_array_path):
            instance.training_array = np.load(training_array_path)

        # if os.path.exists(sampling_array_path):
        #     instance.sampling_array = np.load(sampling_array_path)

        # if os.path.exists(standard_scaler_0_path):
        #     scalers = []
        #     for n in range(300):
        #         scaler = pickle.load(open(standard_scaler_0_path.replace("0", str(n)), "rb"))
        #         scalers.append(scaler)
        #     instance.scalers = scalers

        # if os.path.exists(pca_path):
        #     instance.pca = pickle.load(open(pca_path, "rb"))

        if os.path.exists(nearest_neighbors_path):
            # instance.nearest_neighbors = pickle.load(open(nearest_neighbors_path, "rb"))
            index_cpu = faiss.read_index(nearest_neighbors_path)
            instance.nearest_neighbors = faiss.index_cpu_to_all_gpus(index_cpu)

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
