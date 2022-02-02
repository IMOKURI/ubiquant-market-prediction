import sys

sys.path.append("../inputs")

# isort: split

import logging

import pandas as pd
import ubiquant

import src.utils as utils
from src.feature_store import Store
from src.time_series_api import TimeSeriesAPI

log = logging.getLogger(__name__)


def bench():
    train = pd.read_feather("../inputs/train.f")
    train = train[train["investment_id"] < 100]

    log.info("Training data")
    with utils.timer("store.empty"):
        store = Store.empty()

    log.info(f"Num of train data: {len(train)}")
    iter_train = TimeSeriesAPI(train)

    with utils.timer("store.append"):
        for train_df, sample_prediction_df in iter_train:
            for row in train_df.values:
                store.append(row)

            iter_train.predict(sample_prediction_df)

    log.info("Test data")
    with utils.timer("store.empty"):
        store = Store.empty()

    env = ubiquant.make_env()
    iter_test = env.iter_test()

    with utils.timer("store.append"):
        for test_df, sample_prediction_df in iter_test:
            for row in test_df.values:
                store.append(row)

            env.predict(sample_prediction_df)


if __name__ == "__main__":
    utils.basic_logger()
    bench()
