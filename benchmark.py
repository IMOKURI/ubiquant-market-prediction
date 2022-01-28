import sys

sys.path.append("../inputs")

import logging

import pandas as pd
import ubiquant

import src.utils as utils
from src.feature_store import Store
from src.time_series_api import TimeSeriesAPI

log = logging.getLogger(__name__)


def bench():
    log.info("Training data")
    with utils.timer("store.empty"):
        store = Store.empty()

    train = pd.read_feather("../inputs/train.f")
    small_train = train[train["investment_id"] < 10]

    iter_train = TimeSeriesAPI(small_train[:20])

    for train_df, sample_prediction_df in iter_train:
        with utils.timer("store.append"):
            for _, row in train_df.iterrows():
                store.append(row)

        iter_train.predict(sample_prediction_df)


    log.info("Test data")
    with utils.timer("store.empty"):
        store = Store.empty()

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    for test_df, sample_prediction_df in iter_test:
        with utils.timer("store.append"):
            for _, row in test_df.iterrows():
                store.append(row)

        env.predict(sample_prediction_df)


if __name__ == "__main__":
    utils.basic_logger()
    bench()
