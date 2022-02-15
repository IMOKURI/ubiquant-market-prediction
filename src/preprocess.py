# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from ppca import PPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .utils import timeSince

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame):
    scaled_df = None
    pca_df = None

    if not c.params.preprocess:
        return df, scaled_df, pca_df

    if any([scaler in c.params.preprocess for scaler in ["standard", "power"]]):
        scaled_df = apply_scaler(c, df)

    if "pca" in c.params.preprocess:
        if "standard" in c.params.preprocess and scaled_df is not None:
            pca_df, _ = fit_transform_pca(c, scaled_df)
        else:
            log.warning("You should use Standard Scaler before applying PCA.")
            pca_df, _ = fit_transform_pca(c, df)

    if "neighbors" in c.params.preprocess:
        if pca_df is not None:
            _ = fit_nearest_neighbors(c, pca_df)
        else:
            log.warning("You should use PCA before applying Nearest Neighbors.")
            _ = fit_nearest_neighbors(c, df, pca=False)

    return df, scaled_df, pca_df


def apply_scaler(c, df: pd.DataFrame):
    log.info("Apply scaler.")
    cols = [f"f_{n}" for n in range(300)]
    scaled_df = df.copy()

    start = time.time()
    for n, data in enumerate(df[cols].values.T):

        if "standard" in c.params.preprocess:
            new_data, _ = fit_transform_scaler(c, StandardScaler, f"standard_scaler_f_{n}", data.reshape(-1, 1))

        elif "power" in c.params.preprocess:
            new_data, _ = fit_transform_scaler(c, PowerTransformer, f"power_transformer_f_{n}", data.reshape(-1, 1))

        else:
            raise Exception("Invalid preprocess method.")

        scaled_df[f"f_{n}"] = new_data.squeeze()

        if n % (len(cols) // 10) == 0 or n == (len(cols) - 1):
            log.info(f"  Preprocess: [{n}/{len(cols)}] Elapsed {timeSince(start, float(n + 1) / len(cols)):s}")

    return scaled_df


def fit_transform_scaler(c, scaler_class: type, col: str, data: np.ndarray):
    scaler_path = os.path.join(c.settings.dirs.preprocess, f"{col}.pkl")
    if os.path.exists(scaler_path):
        log.debug("Load pretrained scaler.")
        scaler = pickle.load(open(scaler_path, "rb"))
        new_data = scaler.transform(data)

    else:
        log.debug("Fit and save scaler.")
        scaler = scaler_class()
        new_data = scaler.fit_transform(data)

        os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
        pickle.dump(scaler, open(scaler_path, "wb"))

    return new_data, scaler


def fit_transform_pca(c, df: pd.DataFrame):
    log.info("Apply PCA.")
    f_cols = [f"f_{n}" for n in range(300)]
    data = df[f_cols].values

    pca_path = os.path.join(c.settings.dirs.preprocess, f"pca_{c.params.pca_n_components}.pkl")
    pca_data_path = os.path.join(c.settings.dirs.input_minimal, f"pca_{c.params.pca_n_components}.npy")

    if os.path.exists(pca_path):
        log.debug("Load pretrained PCA.")
        pca = pickle.load(open(pca_path, "rb"))

    else:
        log.debug("Fit and save PCA.")
        pca = PCA(n_components=c.params.pca_n_components, tol=1e-4)
        # pca = PPCA()
        pca.fit(data)

        os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
        pickle.dump(pca, open(pca_path, "wb"))

    if os.path.exists(pca_data_path):
        new_data = np.load(pca_data_path)
    else:
        new_data = pca.transform(data)
        np.save(os.path.splitext(pca_data_path)[0], new_data)

    cols = [f"pca_{n}" for n in range(c.params.pca_n_components)]
    pca_df = pd.DataFrame(new_data, columns=cols)

    pca_df[c.params.label_name] = df[c.params.label_name]

    return pca_df, pca


def fit_nearest_neighbors(c, df: pd.DataFrame, pca: bool = True):
    log.info("Fit nearest neighbors.")

    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[c.params.label_name], bins=num_bins, labels=False)

    if pca:
        cols = [f"pca_{n}" for n in range(c.params.pca_n_components)]
        nn_path = os.path.join(c.settings.dirs.preprocess, f"nearest_neighbors_pca{c.params.pca_n_components}.pkl")
    else:
        cols = [f"f_{n}" for n in range(300)]
        nn_path = os.path.join(c.settings.dirs.preprocess, "nearest_neighbors.pkl")

    sampling_array_path = os.path.join(c.settings.dirs.input_minimal, "sampling_array_0_001.npy")

    if os.path.exists(sampling_array_path):
        sampling_array = np.load(sampling_array_path)
    else:
        _, sampling_df = train_test_split(df, test_size=0.001, stratify=df["bins"])
        sampling_array = sampling_df[cols].values
        np.save(os.path.splitext(sampling_array_path)[0], sampling_array)

    if os.path.exists(nn_path):
        log.debug("Load pretrained nearest neighbors.")
        nn = pickle.load(open(nn_path, "rb"))

    else:
        log.debug("Fit and save nearest neighbors.")
        nn = NearestNeighbors()
        nn.fit(sampling_array)

        os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
        pickle.dump(nn, open(nn_path, "wb"), protocol=4)

    return nn
