# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame):
    cols = [f"f_{n}" for n in range(300)]
    data = df[cols].values

    if c.params.preprocess == "standard":
        new_data, _ = fit_transform(c, StandardScaler, data)

    elif c.params.preprocess == "power":
        new_data, _ = fit_transform(c, PowerTransformer, data)

    else:
        raise Exception("Invalid preprocess method.")

    df[cols] = new_data
    return df


def fit_transform(c, scaler_class: type, data: np.ndarray):
    scaler_path = os.path.join(c.settings.dirs.preprocess, f"{c.params.preprocess}.pkl")
    if os.path.exists(scaler_path):
        log.info("Load pretrained scaler.")
        scaler = pickle.load(open(scaler_path, "rb"))
        new_data = scaler.transform(data)

    else:
        log.info("Fit and save scaler.")
        scaler = scaler_class()
        new_data = scaler.fit_transform(data)

        os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
        pickle.dump(scaler, open(scaler_path, "wb"))

    return new_data, scaler
