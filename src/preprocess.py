# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

from .utils import timeSince

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame):
    nearest_neighbors = None

    if not c.params.preprocess:
        return df, nearest_neighbors

    if any([scaler in c.params.preprocess for scaler in ["standard", "power"]]):
        df = apply_scaler(c, df)

    if "neighbors":
        ...

    return df, nearest_neighbors


def apply_scaler(c, df: pd.DataFrame):
    cols = [f"f_{n}" for n in range(300)]

    start = time.time()
    for n, data in enumerate(df[cols].values.T):

        if "standard" in c.params.preprocess:
            new_data, _ = scaler_fit_transform(c, StandardScaler, f"standard_scaler_f_{n}", data.reshape(-1, 1))

        elif "power" in c.params.preprocess:
            new_data, _ = scaler_fit_transform(c, PowerTransformer, f"power_transformer_f_{n}", data.reshape(-1, 1))

        else:
            raise Exception("Invalid preprocess method.")

        df[f"f_{n}"] = new_data.squeeze()

        if n % (len(cols) // 10) == 0 or n == (len(cols) - 1):
            log.info(f"  Preprocess: [{n}/{len(cols)}] Elapsed {timeSince(start, float(n + 1) / len(cols)):s}")

    return df


def scaler_fit_transform(c, scaler_class: type, col: str, data: np.ndarray):
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
