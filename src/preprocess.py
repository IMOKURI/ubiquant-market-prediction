# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle
import time
from functools import wraps
from typing import Callable

import faiss
import numpy as np
import pandas as pd
from ppca import PPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .utils import timeSince

log = logging.getLogger(__name__)


# TODO:
# デコレーターをもうちょっとなんとかしたい (アンチパターンな気がする)
# - 引数の関数 func の引数が、 func の中では使われていない


def load_or_fit(func: Callable):
    """
    前処理を行うクラスがすでに保存されていれば、それをロードする。
    保存されていなければ、 func で生成、学習する。
    与えられたデータを、学習済みクラスで前処理する。

    Args:
        func (Callable): 前処理を行うクラスのインスタンスを生成し、学習する関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1]) if args[1] is not None else None

        if path is not None and os.path.exists(path):
            instance = pickle.load(open(path, "rb"))

        else:
            instance = func(*args, **kwargs)

            if path is not None:
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                pickle.dump(instance, open(path, "wb"), protocol=4)

        return instance

    return wrapper


def load_or_transform(func: Callable):
    """
    前処理されたデータがすでに存在すれば、それをロードする。
    存在しなければ、 func で生成する。生成したデータは保存しておく。

    Args:
        func (Callable): 前処理を行う関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.input_minimal, args[1])

        if os.path.exists(path):
            array = np.load(path)

        else:
            array = func(*args, **kwargs)

            if isinstance(array, np.ndarray):
                os.makedirs(c.settings.dirs.input_minimal, exist_ok=True)
                np.save(os.path.splitext(path)[0], array)

        return array

    return wrapper


def preprocess(c, df: pd.DataFrame):
    if not c.params.preprocess:
        return df

    training_features = save_training_features(c, "training_features.npy", df)
    save_training_targets(c, "training_targets.npy", df)

    # Normalization
    if "standard_scaler" in c.params.preprocess:
        scaled = apply_standard_scaler(c, "standard_scaled.npy", training_features)
    elif "power_transformer" in c.params.preprocess:
        scaled = apply_power_transformer(c, "power_transformer.npy", training_features)
    else:
        scaled = training_features

    # PCA
    pca_cols = [f"pca_{n}" for n in range(c.params.pca_n_components)]
    if "pca" in c.params.preprocess:
        df[pca_cols] = apply_pca(c, f"pca_{c.params.pca_n_components}.npy", scaled)
    elif "ppca" in c.params.preprocess:
        df[pca_cols] = apply_ppca(c, f"ppca.npy", scaled)

    # Nearest Neighbors
    if "nearest_neighbors" in c.params.preprocess:
        sampling_array = sampling(c, f"sampling_pca{c.params.pca_n_components}.npy", df)
        _ = apply_nearest_neighbors(c, sampling_array)
    elif "faiss_ivfpq" in c.params.preprocess:
        _ = apply_faiss_nearest_neighbors(c, "faiss_ivfpq.index", training_features)

    if "remove_china_shock" in c.params.preprocess:
        df = df[(df["time_id"] < 300) | (df["time_id"] > 550)].reset_index(drop=True)

    return df


@load_or_transform
def save_training_features(c, out_path, df: pd.DataFrame) -> np.ndarray:
    cols = [f"f_{n}" for n in range(300)]
    return df[cols].values


@load_or_transform
def save_training_targets(c, out_path, df: pd.DataFrame) -> np.ndarray:
    return df[c.params.label_name].values


@load_or_transform
def apply_standard_scaler(c, out_path, array: np.ndarray) -> np.ndarray:
    log.info("Apply standard scaler.")
    scaled = np.zeros_like(array, dtype=np.float32)

    for n, data in enumerate(array.T):
        instance = fit_scaler(c, f"standard_scaler_f_{n}.pkl", data.reshape(-1, 1), StandardScaler)
        scaled[:, n] = instance.transform(data.reshape(-1, 1)).squeeze()

    return scaled


@load_or_transform
def apply_power_transformer(c, out_path, array: np.ndarray) -> np.ndarray:
    log.info("Apply power transformer.")
    scaled = np.zeros_like(array, dtype=np.float32)

    for n, data in enumerate(array.T):
        instance = fit_scaler(c, f"power_transformer_f_{n}.pkl", data.reshape(-1, 1), PowerTransformer)
        scaled[:, n] = instance.transform(data.reshape(-1, 1)).squeeze()

    return scaled


@load_or_transform
def apply_pca(c, out_path, array: np.ndarray):
    log.info("Apply PCA.")
    instance = fit_scaler(
        c, f"pca_{c.params.pca_n_components}.pkl", array, PCA, n_components=c.params.pca_n_components, tol=1e-4
    )
    return instance.transform(array)


@load_or_transform
def apply_ppca(c, out_path, array: np.ndarray):
    log.info("Apply PPCA.")
    instance = fit_scaler(c, f"ppca.pkl", array, PPCA)
    return instance.transform(array)


@load_or_transform
def sampling(c, out_path, df: pd.DataFrame):
    log.info("Sampling.")

    pca_cols = [f"pca_{n}" for n in range(c.params.pca_n_components)]
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[c.params.label_name], bins=num_bins, labels=False)

    sampling_rate = 0.001
    _, sampling_df = train_test_split(df, test_size=sampling_rate, stratify=df["bins"], random_state=c.params.seed)
    return sampling_df[pca_cols].values


def apply_nearest_neighbors(c, array: np.ndarray):
    log.info("Apply Nearest Neighbors.")
    instance = fit_scaler(c, f"nearest_neighbors_pca{c.params.pca_n_components}.pkl", array, NearestNeighbors)
    return instance


def apply_faiss_nearest_neighbors(c, out_path, array: np.ndarray):
    if os.path.exists(os.path.join(c.settings.dirs.preprocess, out_path)):
        return None

    log.info("Apply Faiss Nearest Neighbors.")
    scaler = fit_scaler(c, None, array, FaissKNeighbors)
    index_cpu = faiss.index_gpu_to_cpu(scaler.index)
    faiss.write_index(index_cpu, os.path.join(c.settings.dirs.preprocess, out_path))
    return scaler


@load_or_fit
def fit_scaler(c, scaler_path, data: np.ndarray, scaler_class: type, **kwargs):
    scaler = scaler_class(**kwargs)
    scaler.fit(data)
    return scaler


# https://gist.github.com/j-adamczyk/74ee808ffd53cd8545a49f185a908584
# https://www.ariseanalytics.com/activities/report/20210304/
# https://rest-term.com/archives/3414/
class FaissKNeighbors:
    def __init__(self, k=5):
        res = faiss.StandardGpuResources()

        dim = 300  # input dim
        nlist = 7000  # 4 * sqrt(num_data)
        m = 20  # choice from [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96] and dim ≡ 0 (mod m)
        nbits = 8  # fixed for GPU
        metric = faiss.METRIC_L2

        config = faiss.GpuIndexIVFPQConfig()
        config.usePrecomputedTables = True

        self.index = faiss.GpuIndexIVFPQ(res, dim, nlist, m, nbits, metric, config)
        self.k = k

    def fit(self, X):
        self.index.train(np.ascontiguousarray(X, dtype=np.float32))
        self.index.add(np.ascontiguousarray(X, dtype=np.float32))

    def kneighbors(self, X, k: int = 0, nprobe: int = 100, return_distance: bool = True):
        if k <= 0:
            k = self.k

        self.index.nprobe = nprobe

        distances, indices = self.index.search(np.ascontiguousarray(X, dtype=np.float32), k=k)

        if return_distance:
            return distances, indices
        else:
            return indices
