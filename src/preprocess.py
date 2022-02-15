# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle
import time
from functools import wraps

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


def cached_preprocess(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1])
        data = args[2]

        if os.path.exists(path):
            instance = pickle.load(open(path, "rb"))

        else:
            instance = func(*args, **kwargs)
            os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
            try:
                pickle.dump(instance, open(path, "wb"), protocol=4)
            except TypeError:
                pass

        return instance.transform(data)

    return wrapper


def cached_preprocessed_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.input_minimal, args[1])

        if os.path.exists(path):
            array = np.load(path)

        else:
            array = func(*args, **kwargs)
            os.makedirs(c.settings.dirs.input_minimal, exist_ok=True)
            np.save(os.path.splitext(path)[0], array)

        return array

    return wrapper


def preprocess(c, df: pd.DataFrame):
    if not c.params.preprocess:
        return df

    if "standard_scaler" in c.params.preprocess:
        scaled = apply_standard_scaler(c, "standard_scaled.npy", df)

    elif "power_transformer" in c.params.preprocess:
        scaled = apply_power_transformer(c, "power_transformer.npy", df)

    else:
        return df

    if "pca" in c.params.preprocess:
        pca_array = apply_pca(c, f"pca_{c.params.pca_n_components}.npy", scaled)

    elif "ppca" in c.params.preprocess:
        pca_array = apply_ppca(c, f"ppca.npy", scaled)

    else:
        return df

    pca_cols = [f"pca_{n}" for n in range(c.params.pca_n_components)]
    df[pca_cols] = pca_array

    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[c.params.label_name], bins=num_bins, labels=False)

    sampling_rate = 0.001
    _, sampling_df = train_test_split(df, test_size=sampling_rate, stratify=df["bins"], random_state=c.params.seed)

    if "nearest_neighbors" in c.params.preprocess:
        neighbors = apply_nearest_neighbors(
            c,
            f"nearest_neighbors_pca{c.params.pca_n_components}_sample{str(sampling_rate).replace('.', '_')}.npy",
            sampling_df[pca_cols].values,
        )
    elif "faiss_nearest_neighbors" in c.params.preprocess:
        neighbors = apply_faiss_nearest_neighbors(
            c,
            f"faiss_nearest_neighbors_pca{c.params.pca_n_components}_sample{str(sampling_rate).replace('.', '_')}.npy",
            sampling_df[pca_cols].values,
        )

    return df


@cached_preprocessed_data
def apply_standard_scaler(c, out_path, df: pd.DataFrame):
    log.info("Apply standard scaler.")
    cols = [f"f_{n}" for n in range(300)]
    scaled = np.zeros((len(df), len(cols)), dtype=np.float32)

    start = time.time()
    for n, data in enumerate(df[cols].values.T):
        scaled[:, n] = fit_scaler(c, f"standard_scaler_f_{n}.pkl", data.reshape(-1, 1), StandardScaler).squeeze()

        if n % (len(cols) // 10) == 0 or n == (len(cols) - 1):
            log.info(f"  Preprocess: [{n}/{len(cols)}] Elapsed {timeSince(start, float(n + 1) / len(cols)):s}")

    return scaled


@cached_preprocessed_data
def apply_power_transformer(c, out_path, df: pd.DataFrame):
    log.info("Apply power transformer.")
    cols = [f"f_{n}" for n in range(300)]
    scaled = np.zeros((len(df), len(cols)), dtype=np.float32)

    start = time.time()
    for n, data in enumerate(df[cols].values.T):
        scaled[:, n] = fit_scaler(c, f"power_transformer_f_{n}.pkl", data.reshape(-1, 1), PowerTransformer).squeeze()

        if n % (len(cols) // 10) == 0 or n == (len(cols) - 1):
            log.info(f"  Preprocess: [{n}/{len(cols)}] Elapsed {timeSince(start, float(n + 1) / len(cols)):s}")

    return scaled


@cached_preprocessed_data
def apply_pca(c, out_path, array: np.ndarray):
    log.info("Apply PCA.")
    return fit_scaler(
        c, f"pca_{c.params.pca_n_components}.pkl", array, PCA, n_components=c.params.pca_n_components, tol=1e-4
    )


@cached_preprocessed_data
def apply_ppca(c, out_path, array: np.ndarray):
    log.info("Apply PPCA.")
    return fit_scaler(c, f"ppca.pkl", array, PPCA)


@cached_preprocessed_data
def apply_nearest_neighbors(c, out_path, array: np.ndarray):
    log.info("Apply Nearest Neighbors.")
    return fit_scaler(c, f"nearest_neighbors_pca{c.params.pca_n_components}.pkl", array, NearestNeighbors)


@cached_preprocessed_data
def apply_faiss_nearest_neighbors(c, out_path, array: np.ndarray):
    # TODO: faiss nn の保存 write_index
    log.info("Apply Faiss Nearest Neighbors.")
    return fit_scaler(c, f"faiss_nearest_neighbors_pca{c.params.pca_n_components}.pkl", array, FaissKNeighbors)


@cached_preprocess
def fit_scaler(c, scaler_path, data: np.ndarray, scaler_class: type, **kwargs):
    scaler = scaler_class(**kwargs)
    scaler.fit(data)
    return scaler


class FaissKNeighbors:
    def __init__(self, k=5):
        self.res = faiss.StandardGpuResources()
        self.flat_config = faiss.GpuIndexFlatConfig()

        self.index = None
        self.k = k

    def fit(self, X):
        self.index = faiss.GpuIndexFlatL2(self.res, X.shape[1], self.flat_config)
        self.index.add(np.ascontiguousarray(X, dtype=np.float32))

    def kneighbors(self, X, n_neighbors: int = 0, return_distance: bool = True):
        if n_neighbors <= 0:
            n_neighbors = self.k

        distances, indices = self.index.search(np.ascontiguousarray(X, dtype=np.float32), k=n_neighbors)

        if return_distance:
            return distances, indices
        else:
            return indices
