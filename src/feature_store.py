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


class Store:
    def __init__(
        self,
        investments: Investments,
        training_features: Optional[NDArray[(Any, Any), Any]] = None,
        training_targets: Optional[NDArray[(Any, Any), Any]] = None,
        # sampling_array: Optional[NDArray[(Any, Any), Any]] = None,
        # scalers: Optional[List[Union[StandardScaler, PowerTransformer]]] = None,
        # pca: Optional[Union[PCA, PPCA]] = None,
        nearest_neighbors: Optional[NearestNeighbors] = None,
    ):
        self.investments = investments
        self.training_features = training_features
        self.training_targets = training_targets
        # self.sampling_array = sampling_array
        # self.scalers = scalers
        # self.pca = pca
        self.nearest_neighbors = nearest_neighbors

    @classmethod
    def empty(cls) -> "Store":
        investments = Investments()

        return cls(investments)

    @classmethod
    def train(cls, c: DictConfig, training_fold: Optional[int] = None) -> "Store":
        """
        TODO: c.params.preprocess によって、ロードするものを変更する。
        """
        instance = cls.empty()

        # sampling_array_path = os.path.join(c.settings.dirs.input_minimal, f"sampling_pca{c.params.pca_n_components}.npy")

        # standard_scaler_0_path = os.path.join(c.settings.dirs.preprocess, "standard_scaler_f_0.pkl")
        # pca_path = os.path.join(c.settings.dirs.preprocess, f"pca_{c.params.pca_n_components}.pkl")

        if training_fold is None:
            training_features_path = os.path.join(c.settings.dirs.input_minimal, "training_features.npy")
            training_targets_path = os.path.join(c.settings.dirs.input_minimal, "training_targets.npy")
            nearest_neighbors_path = os.path.join(c.settings.dirs.preprocess, "faiss_ivfpq.index")
        else:
            training_features_path = os.path.join(
                c.settings.dirs.input_minimal, f"training_features_{training_fold}.npy"
            )
            training_targets_path = os.path.join(c.settings.dirs.input_minimal, f"training_targets_{training_fold}.npy")
            nearest_neighbors_path = os.path.join(c.settings.dirs.preprocess, f"faiss_ivfpq_{training_fold}.index")

        if os.path.exists(training_features_path):
            instance.training_features = np.load(training_features_path)

        if os.path.exists(training_targets_path):
            instance.training_targets = np.load(training_targets_path)

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

            try:
                devices = c.settings.gpus.split(",")
                resources = [faiss.StandardGpuResources() for _ in devices]
                instance.nearest_neighbors = faiss.index_cpu_to_gpu_multiple_py(resources, index_cpu, gpus=devices)
            except Exception:
                instance.nearest_neighbors = faiss.index_cpu_to_all_gpus(index_cpu)

        return instance

    def update(self, array: NDArray[(Any, Any), Any]):
        for row in array:
            # row の中のデータは dtype: object であることに注意。

            # TODO: 最終的に catch_everything_in_kaggle をいれていく
            # with catch_everything_in_kaggle():
            self.investments.extend(row)

    def update_post(self, array: NDArray[(Any, Any), Any]):
        for row in array:
            # row の中のデータは dtype: object であることに注意。

            # TODO: 最終的に catch_everything_in_kaggle をいれていく
            # with catch_everything_in_kaggle():
            self.investments.extend_post(row)
